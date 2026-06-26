"""Holds the global model + optimizer and handles checkpoint persistence."""

from __future__ import annotations

import os
from typing import Dict

import torch
from safetensors.torch import load_file, save_file

from swarm.config import ModelConfig, SwarmConfig
from swarm.data import build_dataset
from swarm.model import TinyGPT, build_model


class GlobalState:
    """The single source of truth for the swarm: weights, optimizer, version."""

    def __init__(self, model_cfg: ModelConfig, swarm_cfg: SwarmConfig):
        self.model_cfg = model_cfg
        self.swarm_cfg = swarm_cfg
        self.model: TinyGPT = build_model(model_cfg)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=swarm_cfg.lr,
            weight_decay=swarm_cfg.weight_decay,
        )
        self.version: int = 0  # bumped on every applied optimizer step
        self.step: int = 0
        self.last_loss = None  # Optional[float]; None until the first step
        self._param_names = [n for n, _ in self.model.named_parameters()]

    @classmethod
    def bootstrap(cls, model_cfg: ModelConfig, swarm_cfg: SwarmConfig) -> "GlobalState":
        """Build state from the bundled corpus's vocab, restoring a checkpoint if present."""
        ds = build_dataset(block_size=model_cfg.block_size)
        cfg = model_cfg.with_vocab(ds.vocab_size)
        state = cls(cfg, swarm_cfg)
        if swarm_cfg.checkpoint_path and os.path.exists(swarm_cfg.checkpoint_path):
            state.load_checkpoint(swarm_cfg.checkpoint_path)
        return state

    @property
    def param_names(self):
        return list(self._param_names)

    def weights_state_dict(self) -> Dict[str, torch.Tensor]:
        """Full model state_dict (params + buffers) for workers to load.

        Tensors are ``clone()``d so that tied weights (``wte.weight`` is tied to
        ``lm_head.weight``) no longer share storage — safetensors refuses to
        serialize tensors that alias the same memory.
        """
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def named_grad_tensors(self) -> Dict[str, torch.Tensor]:
        return {n: p for n, p in self.model.named_parameters()}

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_file(self.weights_state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        loaded = load_file(path)
        self.model.load_state_dict(loaded, strict=True)
