"""SwarmClient — the Kaggle/worker side of the software interconnect.

The loop is deliberately tiny:

    pull weights  ->  train on one micro-batch  ->  push only the gradients

The worker never holds an optimizer or a second copy of the model, so its peak
memory is one model + one batch. That's why a swarm of free instances can
collectively fine-tune a model bigger than any one of them could *train* alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import httpx
import torch

from swarm import protocol as P
from swarm.config import ModelConfig
from swarm.data import build_dataset
from swarm.model import TinyGPT, build_model


@dataclass
class PushOutcome:
    ok: bool          # True if the server applied/accepted the gradient
    stale: bool       # True if rejected as stale (caller should re-pull)
    status_code: int
    body: dict


class SwarmClient:
    def __init__(
        self,
        server_url: str,
        worker_id: str,
        batch_size: int = 16,
        shard: int = 0,
        num_shards: int = 1,
        token: str = "",
        timeout: float = 60.0,
        seed: Optional[int] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.token = token
        self._http = httpx.Client(timeout=timeout)
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

        # Adopt the *server's* model config so architecture/vocab are guaranteed
        # identical — no reliance on env vars matching across machines.
        server_cfg = self._fetch_config()
        self.model_cfg: ModelConfig = ModelConfig.from_dict(server_cfg["model"])
        self.dataset = build_dataset(
            block_size=self.model_cfg.block_size, shard=shard, num_shards=num_shards
        )
        if self.dataset.vocab_size != self.model_cfg.vocab_size:
            raise RuntimeError(
                f"vocab mismatch: server={self.model_cfg.vocab_size}, "
                f"local corpus={self.dataset.vocab_size}. Use the same corpus everywhere."
            )
        self.model: TinyGPT = build_model(self.model_cfg)
        self.version: int = -1

    # ---- HTTP ---------------------------------------------------------------
    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {}
        if self.token:
            h[P.H_AUTH] = P.bearer(self.token)
        if extra:
            h.update(extra)
        return h

    def _fetch_config(self) -> dict:
        r = self._http.get(self.server_url + P.EP_CONFIG)
        r.raise_for_status()
        return r.json()

    def pull(self) -> int:
        """Download global weights into the local model; returns the model version."""
        r = self._http.get(self.server_url + P.EP_WEIGHTS, headers=self._headers())
        r.raise_for_status()
        tensors = P.deserialize_tensors(r.content)
        self.model.load_state_dict(tensors, strict=True)
        self.version = int(r.headers[P.H_MODEL_VERSION])
        return self.version

    def push(self, grads: Dict[str, torch.Tensor], loss: float) -> PushOutcome:
        body = P.serialize_tensors(grads)
        headers = self._headers(
            {
                P.H_MODEL_VERSION: str(self.version),
                P.H_LOSS: f"{loss:.6f}",
                P.H_WORKER_ID: self.worker_id,
                "Content-Type": P.CONTENT_TYPE,
            }
        )
        r = self._http.post(self.server_url + P.EP_GRADIENTS, content=body, headers=headers)
        if r.status_code == 409:
            return PushOutcome(ok=False, stale=True, status_code=409, body=_safe_json(r))
        r.raise_for_status()
        return PushOutcome(ok=True, stale=False, status_code=r.status_code, body=r.json())

    # ---- compute ------------------------------------------------------------
    def train_step(self) -> Tuple[Dict[str, torch.Tensor], float]:
        """Compute gradients of one micro-batch w.r.t. the freshly pulled weights."""
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        x, y = self.dataset.get_batch(self.batch_size, generator=self._gen)
        _, loss = self.model(x, y)
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in self.model.named_parameters()}
        return grads, float(loss.item())

    def run(self, steps: int, verbose: bool = True) -> None:
        """Full worker loop: (pull -> train -> push) x steps, re-pulling on staleness."""
        for i in range(steps):
            self.pull()
            grads, loss = self.train_step()
            outcome = self.push(grads, loss)
            if outcome.stale:
                if verbose:
                    print(f"[{self.worker_id}] step {i}: stale, re-pulling")
                continue
            if verbose:
                b = outcome.body
                tag = "STEP" if b.get("applied") else "buffered"
                print(
                    f"[{self.worker_id}] iter {i}: loss={loss:.4f} -> {tag} "
                    f"v={b.get('version')} pending={b.get('pending')}"
                )

    def close(self) -> None:
        self._http.close()


def _safe_json(r: httpx.Response) -> dict:
    try:
        return r.json()
    except Exception:
        return {"detail": r.text}
