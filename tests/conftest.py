import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swarm.config import ModelConfig, SwarmConfig  # noqa: E402
from server.state import GlobalState  # noqa: E402


def tiny_model_cfg(vocab_size: int = 17) -> ModelConfig:
    """A minimal architecture for fast unit tests."""
    return ModelConfig(
        vocab_size=vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0
    )


def make_state(mode: str = "sync", world_size: int = 2, tolerance: int = 0) -> GlobalState:
    swarm_cfg = SwarmConfig(
        mode=mode,
        world_size=world_size,
        staleness_tolerance=tolerance,
        checkpoint_path="",       # no disk persistence in tests
        checkpoint_every=0,
    ).validate()
    return GlobalState(tiny_model_cfg(), swarm_cfg)


def zero_grads(state: GlobalState):
    return {n: torch.zeros_like(p) for n, p in state.model.named_parameters()}


def const_grads(state: GlobalState, value: float):
    return {n: torch.full_like(p, value) for n, p in state.model.named_parameters()}
