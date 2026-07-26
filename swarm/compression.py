"""Gradient compression — the bandwidth kernel.

On a real GPU cluster the interconnect is NVLink at ~600 GB/s. Here it is HTTP
over the public internet at maybe 10 MB/s, so **bytes on the wire are the
bottleneck**, not FLOPs. A worker that spends 200 ms computing a gradient and
8 s uploading it is 97% idle.

The fix is standard in distributed training but rarely built from scratch:

* **Top-K sparsification** — only the ``ratio`` fraction of gradient entries with
  the largest magnitude are transmitted (as index/value pairs). At ratio=0.01
  that is a ~50x reduction in payload.
* **Error feedback** — the entries we *dropped* are not discarded; they are kept
  in a local residual buffer and added to the next step's gradient. A small
  consistent gradient direction therefore accumulates until it becomes large
  enough to be sent. Without this, sparsification biases training and
  convergence stalls; with it, Top-K is near-lossless in practice.
  (Seide et al. 2014; Lin et al., "Deep Gradient Compression", 2018.)
* **int8 quantization** — the surviving values are optionally quantized to 8-bit
  with a per-tensor scale, for another ~4x on top.

The server reconstructs a dense gradient before aggregating, so compression is
entirely transparent to :class:`~server.aggregator.GradientAggregator`.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

# Wire-format suffixes. safetensors payloads are flat {name: tensor} maps, so a
# compressed tensor is split across a few synthetic keys.
IDX = "::idx"
VAL = "::val"
SCALE = "::scale"

# Values for the X-Compression header.
NONE = "none"
TOPK = "topk"
TOPK_INT8 = "topk-int8"
MODES = (NONE, TOPK, TOPK_INT8)


def _numel(shape: Sequence[int]) -> int:
    return int(math.prod(shape)) if len(shape) else 1


def topk_count(numel: int, ratio: float) -> int:
    """How many entries survive Top-K for a tensor of ``numel`` elements."""
    if ratio >= 1.0:
        return numel
    return max(1, min(numel, int(numel * ratio)))


def compress(
    tensors: Mapping[str, torch.Tensor],
    mode: str = TOPK,
    ratio: float = 0.01,
    residual: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Compress ``{name: dense_gradient}`` into a sparse wire payload.

    ``residual`` (error feedback) is read *and updated in place* when provided:
    dropped mass is carried into the next call so no gradient signal is lost,
    only delayed.
    """
    if mode == NONE:
        return dict(tensors)
    if mode not in MODES:
        raise ValueError(f"unknown compression mode {mode!r}; expected one of {MODES}")

    quantize = mode == TOPK_INT8
    out: Dict[str, torch.Tensor] = {}

    for name, grad in tensors.items():
        g = grad.detach().to(torch.float32)
        if residual is not None:
            prev = residual.get(name)
            if prev is not None:
                g = g + prev

        flat = g.reshape(-1)
        k = topk_count(flat.numel(), ratio)
        # Select by magnitude; sorted=False is cheaper and order is irrelevant.
        _, idx = torch.topk(flat.abs(), k, sorted=False)
        vals = flat[idx]

        if residual is not None:
            # Everything we did NOT send stays behind for the next step.
            carry = flat.clone()
            carry[idx] = 0.0
            residual[name] = carry.view_as(grad)

        if quantize:
            peak = vals.abs().max()
            scale = (peak / 127.0) if peak > 0 else torch.tensor(1.0)
            out[name + VAL] = torch.round(vals / scale).clamp_(-127, 127).to(torch.int8)
            out[name + SCALE] = scale.reshape(1).to(torch.float32)
        else:
            out[name + VAL] = vals

        out[name + IDX] = idx.to(torch.int64)

    return out


def decompress(
    payload: Mapping[str, torch.Tensor],
    shapes: Mapping[str, torch.Size],
    mode: str = TOPK,
) -> Dict[str, torch.Tensor]:
    """Rebuild dense gradients from a sparse payload.

    Shapes come from the server's own model, so they never travel on the wire.
    """
    if mode == NONE:
        return dict(payload)
    if mode not in MODES:
        raise ValueError(f"unknown compression mode {mode!r}; expected one of {MODES}")

    out: Dict[str, torch.Tensor] = {}
    for name, shape in shapes.items():
        try:
            idx = payload[name + IDX]
            val = payload[name + VAL]
        except KeyError as e:  # pragma: no cover - surfaced as a 400 by the app
            raise KeyError(f"compressed payload missing entry for {name!r}: {e}") from e

        scale = payload.get(name + SCALE)
        if scale is not None:
            val = val.to(torch.float32) * float(scale.reshape(-1)[0])

        flat = torch.zeros(_numel(shape), dtype=torch.float32)
        flat[idx.to(torch.int64)] = val.to(torch.float32)
        out[name] = flat.view(shape)

    return out


def payload_bytes(payload: Mapping[str, torch.Tensor]) -> int:
    """Total tensor bytes in a payload (excludes safetensors' small header)."""
    return sum(t.numel() * t.element_size() for t in payload.values())


def compression_report(
    dense: Mapping[str, torch.Tensor], wire: Mapping[str, torch.Tensor]
) -> Tuple[int, int, float]:
    """(dense_bytes, wire_bytes, ratio) — ratio is how many times smaller the wire is."""
    d = payload_bytes(dense)
    w = payload_bytes(wire)
    return d, w, (d / w if w else 1.0)
