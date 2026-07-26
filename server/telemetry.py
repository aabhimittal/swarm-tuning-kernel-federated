"""Swarm telemetry — who contributed what, and how the global model is moving.

A federated swarm is opaque by default: gradients arrive from anonymous
ephemeral machines and vanish. This module keeps the small amount of state that
makes the swarm legible — a per-worker contribution ledger, a rolling loss
history for the dashboard chart, and cumulative bandwidth accounting so the
value of compression is measurable rather than asserted.

All of it is in-memory and bounded; a Space restart resets it.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class WorkerStats:
    worker_id: str
    accepted: int = 0
    stale: int = 0
    rejected: int = 0
    steps_triggered: int = 0          # pushes that completed a global step
    local_steps: int = 0              # total local optimizer steps contributed
    wire_bytes: int = 0               # bytes actually uploaded
    dense_bytes: int = 0              # bytes an uncompressed upload would have cost
    last_loss: Optional[float] = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        saved = self.dense_bytes - self.wire_bytes
        return {
            "worker_id": self.worker_id,
            "accepted": self.accepted,
            "stale": self.stale,
            "rejected": self.rejected,
            "steps_triggered": self.steps_triggered,
            "local_steps": self.local_steps,
            "wire_bytes": self.wire_bytes,
            "dense_bytes": self.dense_bytes,
            "bytes_saved": max(saved, 0),
            "compression_ratio": (
                round(self.dense_bytes / self.wire_bytes, 2) if self.wire_bytes else 1.0
            ),
            "last_loss": self.last_loss,
            "age_s": round(time.time() - self.first_seen, 1),
            "idle_s": round(time.time() - self.last_seen, 1),
        }


class SwarmTelemetry:
    """Thread-safe ledger of worker contributions and global loss history."""

    def __init__(self, history_size: int = 300, max_workers: int = 500):
        self._lock = threading.Lock()
        self._workers: Dict[str, WorkerStats] = {}
        self._history: Deque[Tuple[int, float]] = deque(maxlen=history_size)
        self._max_workers = max_workers
        self._started = time.time()
        self._total_wire = 0
        self._total_dense = 0

    # ---- recording ---------------------------------------------------------
    def record_push(
        self,
        worker_id: Optional[str],
        *,
        outcome: str,               # "accepted" | "stale" | "rejected"
        loss: Optional[float] = None,
        wire_bytes: int = 0,
        dense_bytes: int = 0,
        applied: bool = False,
        local_steps: int = 1,
    ) -> None:
        wid = (worker_id or "anonymous")[:64]
        with self._lock:
            w = self._workers.get(wid)
            if w is None:
                # Bound the ledger so a hostile client can't exhaust memory by
                # inventing a new worker id on every request.
                if len(self._workers) >= self._max_workers:
                    self._evict_stalest_locked()
                w = self._workers[wid] = WorkerStats(worker_id=wid)

            w.last_seen = time.time()
            w.wire_bytes += wire_bytes
            w.dense_bytes += dense_bytes
            self._total_wire += wire_bytes
            self._total_dense += dense_bytes

            if outcome == "accepted":
                w.accepted += 1
                w.local_steps += max(local_steps, 1)
                if loss is not None:
                    w.last_loss = loss
                if applied:
                    w.steps_triggered += 1
            elif outcome == "stale":
                w.stale += 1
            else:
                w.rejected += 1

    def record_step(self, version: int, loss: Optional[float]) -> None:
        # NaN/inf are not JSON-representable; drop them rather than poison /history.
        if loss is None or not math.isfinite(loss):
            return
        with self._lock:
            self._history.append((version, float(loss)))

    def _evict_stalest_locked(self) -> None:
        oldest = min(self._workers.values(), key=lambda w: w.last_seen)
        self._workers.pop(oldest.worker_id, None)

    # ---- reading -----------------------------------------------------------
    def workers(self) -> List[dict]:
        with self._lock:
            rows = [w.as_dict() for w in self._workers.values()]
        # Most productive first — this is the leaderboard.
        rows.sort(key=lambda r: (r["accepted"], -r["idle_s"]), reverse=True)
        return rows

    def history(self) -> List[dict]:
        with self._lock:
            return [{"version": v, "loss": l} for v, l in self._history]

    def summary(self) -> dict:
        with self._lock:
            active = sum(1 for w in self._workers.values() if time.time() - w.last_seen < 60)
            return {
                "known_workers": len(self._workers),
                "active_workers": active,
                "uptime_s": round(time.time() - self._started, 1),
                "wire_bytes": self._total_wire,
                "dense_bytes": self._total_dense,
                "bytes_saved": max(self._total_dense - self._total_wire, 0),
                "compression_ratio": (
                    round(self._total_dense / self._total_wire, 2) if self._total_wire else 1.0
                ),
            }
