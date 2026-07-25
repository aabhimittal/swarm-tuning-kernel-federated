"""Byzantine-robust aggregation rules — the trust kernel.

A swarm is open by construction: anyone holding the token can push gradients.
Plain averaging has a breakdown point of *zero* — one worker sending a single
huge tensor drags the global mean arbitrarily far and destroys the model. That
is a real threat here, not a theoretical one, because the parameter server is a
public URL.

These rules trade a little statistical efficiency for a non-zero breakdown point:

* ``mean``          — standard accumulation. Fastest, zero robustness.
* ``median``        — coordinate-wise median. Tolerates up to 50% malicious
                      workers; each coordinate is decided by the majority.
* ``trimmed_mean``  — drop the ``trim_ratio`` largest and smallest values per
                      coordinate, average the rest. Robust *and* smoother than
                      the median. (Yin et al., 2018.)
* ``krum``          — pick the single gradient closest to its nearest
                      neighbours, i.e. the most "consensual" update, and discard
                      the rest entirely. Strongest against colluding outliers.
                      (Blanchard et al., NeurIPS 2017.)

All rules take a list of ``{name: tensor}`` gradient dicts and return one dict.
"""

from __future__ import annotations

from typing import Dict, List

import torch


def _stack(buffer: List[Dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    """(num_workers, *param_shape) stack of one parameter across the swarm."""
    return torch.stack([g[key] for g in buffer], dim=0)


def mean(buffer: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: _stack(buffer, k).mean(dim=0) for k in buffer[0]}


def median(buffer: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: _stack(buffer, k).median(dim=0).values for k in buffer[0]}


def trimmed_mean(
    buffer: List[Dict[str, torch.Tensor]], trim_ratio: float = 0.25
) -> Dict[str, torch.Tensor]:
    n = len(buffer)
    trim = int(n * trim_ratio)
    # Need at least one survivor after trimming both ends.
    if trim == 0 or n - 2 * trim < 1:
        return mean(buffer)
    out: Dict[str, torch.Tensor] = {}
    for k in buffer[0]:
        stacked = _stack(buffer, k)
        ordered, _ = torch.sort(stacked, dim=0)
        out[k] = ordered[trim : n - trim].mean(dim=0)
    return out


def krum(
    buffer: List[Dict[str, torch.Tensor]], trim_ratio: float = 0.25
) -> Dict[str, torch.Tensor]:
    """Select the most consensual gradient (single-Krum)."""
    n = len(buffer)
    f = int(n * trim_ratio)  # assumed number of Byzantine workers
    # Krum scores need n - f - 2 neighbours; fall back when the swarm is small.
    neighbours = n - f - 2
    if neighbours < 1:
        return median(buffer) if n >= 3 else mean(buffer)

    flat = torch.stack([
        torch.cat([g[k].reshape(-1) for k in sorted(g)]) for g in buffer
    ])
    dist = torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0)).squeeze(0) ** 2
    dist.fill_diagonal_(float("inf"))  # exclude self-distance
    # Sum of squared distances to the closest `neighbours` peers.
    closest, _ = torch.sort(dist, dim=1)
    scores = closest[:, :neighbours].sum(dim=1)
    winner = int(torch.argmin(scores))
    return {k: v.clone() for k, v in buffer[winner].items()}


def aggregate(
    buffer: List[Dict[str, torch.Tensor]], rule: str = "mean", trim_ratio: float = 0.25
) -> Dict[str, torch.Tensor]:
    """Combine buffered gradients using the configured rule."""
    if not buffer:
        raise ValueError("cannot aggregate an empty buffer")
    if len(buffer) == 1:
        return {k: v.clone() for k, v in buffer[0].items()}
    if rule == "mean":
        return mean(buffer)
    if rule == "median":
        return median(buffer)
    if rule == "trimmed_mean":
        return trimmed_mean(buffer, trim_ratio)
    if rule == "krum":
        return krum(buffer, trim_ratio)
    raise ValueError(f"unknown aggregation rule {rule!r}")
