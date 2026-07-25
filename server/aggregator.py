"""GradientAggregator — averages worker gradients and steps the global model.

This is the "software interconnect". Two modes:

* **sync**  — buffer ``world_size`` gradient sets computed against the *current*
  global version, average them element-wise, run one optimizer step. The averaged
  update is exactly what you'd get from one large batch = sum of the workers'
  micro-batches. This is gradient accumulation, but over HTTP instead of NVLink.

* **async** — apply every gradient as soon as it lands, scaled down by how stale it
  is (``1 / (1 + age)``). Higher throughput, noisier convergence, no barrier.

All model mutation happens under a single lock, so concurrent uploads from many
workers are serialized and can never corrupt the weights.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from server import robust
from server.state import GlobalState


class StaleGradientError(Exception):
    """Raised when an upload is too far behind the current global version."""

    def __init__(self, worker_version: int, global_version: int, tolerance: int):
        self.worker_version = worker_version
        self.global_version = global_version
        self.tolerance = tolerance
        super().__init__(
            f"gradient at v{worker_version} is stale "
            f"(global v{global_version}, tolerance {tolerance})"
        )


class InvalidGradientError(Exception):
    """Raised when an upload's tensor keys don't match the model parameters."""


@dataclass
class SubmitResult:
    accepted: bool
    applied: bool          # True iff an optimizer step was taken
    version: int           # global version after this submission
    step: int
    pending: int           # gradients buffered toward the next step (sync mode)
    last_loss: float


class GradientAggregator:
    def __init__(self, state: GlobalState):
        self.state = state
        self.cfg = state.swarm_cfg
        self._lock = threading.Lock()
        # sync-mode buffer of gradient dicts awaiting averaging.
        self._buffer: List[Dict[str, torch.Tensor]] = []
        self._buffer_losses: List[float] = []
        self._expected_keys = set(state.param_names)

    # ---- public API ---------------------------------------------------------
    def submit(
        self,
        grads: Dict[str, torch.Tensor],
        worker_version: int,
        loss: Optional[float] = None,
    ) -> SubmitResult:
        self._validate_keys(grads)
        with self._lock:
            age = self.state.version - worker_version
            if age < 0 or age > self.cfg.staleness_tolerance:
                raise StaleGradientError(
                    worker_version, self.state.version, self.cfg.staleness_tolerance
                )
            if self.cfg.mode == "async":
                return self._submit_async(grads, age, loss)
            return self._submit_sync(grads, loss)

    def status(self) -> dict:
        with self._lock:
            norm = self._weight_norm_locked()
            return {
                "mode": self.cfg.mode,
                "rule": self.cfg.rule,
                "version": self.state.version,
                "step": self.state.step,
                "pending": len(self._buffer),
                "world_size": self.cfg.world_size,
                "last_loss": self.state.last_loss,
                "num_params": self.state.model.num_params(),
                # L2 norm of all weights. The honest liveness signal: a poisoned
                # update blows this up or turns it NaN long before the reported
                # loss reflects it. None means the model has diverged.
                "weight_norm": norm,
                "healthy": norm is not None,
            }

    def _weight_norm_locked(self) -> Optional[float]:
        with torch.no_grad():
            total = 0.0
            for p in self.state.model.parameters():
                total += float(p.detach().float().pow(2).sum())
                if not math.isfinite(total):
                    return None
        norm = math.sqrt(total)
        return norm if math.isfinite(norm) else None

    # ---- sync ---------------------------------------------------------------
    def _submit_sync(
        self, grads: Dict[str, torch.Tensor], loss: Optional[float]
    ) -> SubmitResult:
        self._buffer.append(grads)
        if loss is not None:
            self._buffer_losses.append(loss)

        if len(self._buffer) < self.cfg.world_size:
            return SubmitResult(
                accepted=True,
                applied=False,
                version=self.state.version,
                step=self.state.step,
                pending=len(self._buffer),
                last_loss=self.state.last_loss,
            )

        averaged = robust.aggregate(self._buffer, self.cfg.rule, self.cfg.trim_ratio)
        finite = [l for l in self._buffer_losses if math.isfinite(l)]
        mean_loss = sum(finite) / len(finite) if finite else None
        self._apply_and_step(averaged, mean_loss)
        self._buffer.clear()
        self._buffer_losses.clear()
        return SubmitResult(
            accepted=True,
            applied=True,
            version=self.state.version,
            step=self.state.step,
            pending=0,
            last_loss=self.state.last_loss,
        )

    # ---- async --------------------------------------------------------------
    def _submit_async(
        self, grads: Dict[str, torch.Tensor], age: int, loss: Optional[float]
    ) -> SubmitResult:
        scale = 1.0 / (1.0 + max(age, 0))
        scaled = {k: v * scale for k, v in grads.items()}
        self._apply_and_step(scaled, loss)
        return SubmitResult(
            accepted=True,
            applied=True,
            version=self.state.version,
            step=self.state.step,
            pending=0,
            last_loss=self.state.last_loss,
        )

    # ---- shared step --------------------------------------------------------
    def _apply_and_step(
        self, grads: Dict[str, torch.Tensor], loss: Optional[float]
    ) -> None:
        """Load grads onto params, clip, optimizer.step(), bump version. (lock held)"""
        model = self.state.model
        opt = self.state.optimizer
        opt.zero_grad(set_to_none=True)
        for name, p in model.named_parameters():
            g = grads.get(name)
            p.grad = g.to(p.device, p.dtype) if g is not None else None
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        self.state.version += 1
        self.state.step += 1
        # A hostile or broken worker can report NaN/inf. Never let that reach the
        # state: it is not JSON-representable and would 500 /status for everyone.
        if loss is not None and math.isfinite(loss):
            self.state.last_loss = float(loss)
        self._maybe_checkpoint()

    def _maybe_checkpoint(self) -> None:
        every = self.cfg.checkpoint_every
        if every and self.state.step % every == 0 and self.cfg.checkpoint_path:
            try:
                self.state.save_checkpoint(self.cfg.checkpoint_path)
            except OSError:
                # Persistence is best-effort; never let a disk hiccup kill training.
                pass

    # ---- helpers ------------------------------------------------------------
    def _validate_keys(self, grads: Dict[str, torch.Tensor]) -> None:
        keys = set(grads.keys())
        if keys != self._expected_keys:
            missing = self._expected_keys - keys
            extra = keys - self._expected_keys
            raise InvalidGradientError(
                f"gradient keys mismatch (missing={sorted(missing)[:3]}, "
                f"extra={sorted(extra)[:3]})"
            )
