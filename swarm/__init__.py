"""Shared core for federated swarm tuning.

This package is imported by *both* the parameter server and the workers so that
the model architecture, data pipeline, and wire protocol are guaranteed to be
identical on every node of the swarm.
"""

from swarm.config import ModelConfig, SwarmConfig

__all__ = ["ModelConfig", "SwarmConfig"]
