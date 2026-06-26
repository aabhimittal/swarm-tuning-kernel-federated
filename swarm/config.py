"""Configuration for the model and the swarm, driven by environment variables.

Both the server and the workers build their model from :class:`ModelConfig`, so
the architecture is identical everywhere. Keeping these in one place means a node
can be configured purely through env vars (handy on Hugging Face Spaces and
Kaggle, where you set "Variables"/"Secrets" rather than editing code).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the TinyGPT model shared across the swarm.

    The defaults are intentionally small so the whole demo trains on a free CPU.
    Bump ``n_layer``/``n_embd``/``block_size`` (via env vars) to simulate a model
    "larger than a single free GPU could hold" — the swarm still trains it because
    each worker only ever holds one copy and processes one micro-batch at a time.
    """

    vocab_size: int = 0  # filled in from the dataset at build time
    block_size: int = _env_int("SWARM_BLOCK_SIZE", 64)
    n_layer: int = _env_int("SWARM_N_LAYER", 4)
    n_head: int = _env_int("SWARM_N_HEAD", 4)
    n_embd: int = _env_int("SWARM_N_EMBD", 128)
    dropout: float = _env_float("SWARM_DROPOUT", 0.0)

    def with_vocab(self, vocab_size: int) -> "ModelConfig":
        return ModelConfig(
            vocab_size=vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class SwarmConfig:
    """Runtime configuration for the parameter server / aggregation."""

    # "sync"  -> buffer WORLD_SIZE gradients, average, take one optimizer step.
    # "async" -> apply every gradient as it arrives (scaled by staleness).
    mode: str = _env_str("SWARM_AGG_MODE", "sync").lower()
    world_size: int = _env_int("SWARM_WORLD_SIZE", 4)
    lr: float = _env_float("SWARM_LR", 3e-4)
    weight_decay: float = _env_float("SWARM_WEIGHT_DECAY", 0.0)
    grad_clip: float = _env_float("SWARM_GRAD_CLIP", 1.0)  # <=0 disables clipping
    # Reject gradients computed against a model version this far behind the global
    # version. 0 = require exact version match (strict). Higher tolerates staleness.
    staleness_tolerance: int = _env_int("SWARM_STALENESS_TOLERANCE", 0)
    # Optional shared secret. If set, /gradients requires "Authorization: Bearer <token>".
    token: str = _env_str("SWARM_TOKEN", "")
    # Where the server persists the global checkpoint (HF Spaces persistent disk = /data).
    checkpoint_path: str = _env_str("SWARM_CHECKPOINT", "global_model.safetensors")
    # Save a checkpoint every N optimizer steps (0 disables periodic saving).
    checkpoint_every: int = _env_int("SWARM_CHECKPOINT_EVERY", 25)

    def validate(self) -> "SwarmConfig":
        if self.mode not in ("sync", "async"):
            raise ValueError(f"SWARM_AGG_MODE must be 'sync' or 'async', got {self.mode!r}")
        if self.world_size < 1:
            raise ValueError("SWARM_WORLD_SIZE must be >= 1")
        return self
