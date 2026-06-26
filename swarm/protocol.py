"""Wire protocol for the software interconnect.

Tensors (weights or gradients) travel as **safetensors** bytes — a safe,
zero-copy, no-pickle format. That matters because the parameter server is a public
Hugging Face Space: accepting pickled tensors from the internet would be a remote
code execution hole, whereas safetensors can only ever produce tensors.

Lightweight metadata (model version, worker id, batch loss) travels in HTTP
headers rather than inside the payload, so the server can route/validate an upload
before deserializing it.
"""

from __future__ import annotations

from typing import Dict

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

# ---- Endpoints ----
EP_HEALTH = "/health"
EP_CONFIG = "/config"
EP_WEIGHTS = "/weights"
EP_GRADIENTS = "/gradients"
EP_STATUS = "/status"
EP_CHECKPOINT = "/checkpoint"

# ---- Headers ----
H_MODEL_VERSION = "X-Model-Version"
H_WORKER_ID = "X-Worker-Id"
H_LOSS = "X-Loss"
H_AUTH = "Authorization"

CONTENT_TYPE = "application/octet-stream"


def _to_transport(t: torch.Tensor) -> torch.Tensor:
    """Detach to a contiguous CPU tensor suitable for safetensors serialization."""
    return t.detach().to("cpu").contiguous()


def serialize_tensors(tensors: Dict[str, torch.Tensor]) -> bytes:
    """Serialize a ``{name: tensor}`` mapping (state_dict or gradients) to bytes."""
    payload = {k: _to_transport(v) for k, v in tensors.items()}
    return st_save(payload)


def deserialize_tensors(data: bytes) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`serialize_tensors`. Returns CPU tensors."""
    return st_load(bytes(data))


def bearer(token: str) -> str:
    return f"Bearer {token}"
