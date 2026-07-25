"""SwarmClient — the Kaggle/worker side of the software interconnect.

The base loop is deliberately tiny:

    pull weights  ->  train on one micro-batch  ->  push only the gradients

Two options change the communication/computation trade-off:

* ``compression`` — Top-K sparsify the gradient (with error feedback) before
  upload, cutting the payload by 50x or more. See :mod:`swarm.compression`.
* ``local_steps`` — run N local optimizer steps and upload the *accumulated
  parameter delta* instead of a single-batch gradient. The server feeds that
  delta to its own optimizer, which is exactly FedAvg-with-a-server-optimizer
  ("FedOpt", Reddi et al. 2021). N local steps means 1/N as many round trips,
  which matters far more than FLOPs when the interconnect is HTTP.

The worker never holds an optimizer state for the *global* model and never a
second replica, so its peak memory is one model + one batch. That is why a swarm
of free instances can collectively train a model none of them could train alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import httpx
import torch

from swarm import compression as C
from swarm import protocol as P
from swarm.config import ModelConfig
from swarm.data import build_dataset
from swarm.model import TinyGPT, build_model


@dataclass
class PushOutcome:
    ok: bool          # True if the server accepted the gradient
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
        compression: str = C.NONE,
        compress_ratio: float = 0.01,
        local_steps: int = 1,
        local_lr: float = 1e-3,
        byzantine: float = 0.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.token = token
        self.local_steps = max(int(local_steps), 1)
        self.local_lr = local_lr
        # Red-team switch: when > 0 this worker uploads garbage of that magnitude
        # instead of a real gradient. Exists so you can verify your own swarm's
        # Byzantine-robust aggregation actually holds (see scripts/simulate_swarm.py
        # --byzantine). With rule="mean" one such worker destroys the model.
        self.byzantine = float(byzantine)
        if compression not in C.MODES:
            raise ValueError(f"compression must be one of {C.MODES}, got {compression!r}")
        self.compression = compression
        self.compress_ratio = compress_ratio
        # Error-feedback buffer: gradient mass dropped by Top-K, carried forward.
        self._residual: Dict[str, torch.Tensor] = {}
        self.last_ratio: float = 1.0

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
        wire = grads
        if self.compression != C.NONE:
            wire = C.compress(
                grads,
                mode=self.compression,
                ratio=self.compress_ratio,
                residual=self._residual,
            )
            _, _, self.last_ratio = C.compression_report(grads, wire)

        body = P.serialize_tensors(wire)
        headers = self._headers(
            {
                P.H_MODEL_VERSION: str(self.version),
                P.H_LOSS: f"{loss:.6f}",
                P.H_WORKER_ID: self.worker_id,
                P.H_COMPRESSION: self.compression,
                P.H_LOCAL_STEPS: str(self.local_steps),
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
        return grads, loss.item()

    def train_local(self) -> Tuple[Dict[str, torch.Tensor], float]:
        """Run ``local_steps`` local SGD steps; return the parameter delta as a
        pseudo-gradient plus the mean local loss.

        ``delta = theta_start - theta_end`` points in the same direction as an
        accumulated gradient, so the server's optimizer can consume it unchanged.
        """
        if self.byzantine > 0:
            grads = {
                n: torch.randn_like(p) * self.byzantine
                for n, p in self.model.named_parameters()
            }
            return grads, float("nan")

        if self.local_steps == 1:
            return self.train_step()

        start = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        opt = torch.optim.SGD(self.model.parameters(), lr=self.local_lr)
        self.model.train()
        total = 0.0
        for _ in range(self.local_steps):
            opt.zero_grad(set_to_none=True)
            x, y = self.dataset.get_batch(self.batch_size, generator=self._gen)
            _, loss = self.model(x, y)
            loss.backward()
            opt.step()
            total += loss.item()

        delta = {
            n: (start[n] - p.detach()) for n, p in self.model.named_parameters()
        }
        return delta, total / self.local_steps

    def run(self, steps: int, verbose: bool = True) -> None:
        """Full worker loop: (pull -> train -> push) x steps, re-pulling on staleness."""
        for i in range(steps):
            self.pull()
            grads, loss = self.train_local()
            outcome = self.push(grads, loss)
            if outcome.stale:
                if verbose:
                    print(f"[{self.worker_id}] round {i}: stale, re-pulling")
                continue
            if verbose:
                b = outcome.body
                tag = "STEP" if b.get("applied") else "buffered"
                extra = f" comp={self.last_ratio:.0f}x" if self.compression != C.NONE else ""
                print(
                    f"[{self.worker_id}] round {i}: loss={loss:.4f} -> {tag} "
                    f"v={b.get('version')} pending={b.get('pending')}{extra}"
                )

    def close(self) -> None:
        self._http.close()


def _safe_json(r: httpx.Response) -> dict:
    try:
        return r.json()
    except Exception:
        return {"detail": r.text}
