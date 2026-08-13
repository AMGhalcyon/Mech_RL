"""Hydra structured config for training hyperparameters.

This module provides a Hydra-compatible dataclass for PPO training
hyperparameters used by the training entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainConfig:
    """Structured config for PPO training hyperparameters.

    Matches the fields used by Stable-Baselines3's PPO constructor.
    """

    # RL algorithm
    algorithm: str = "PPO"

    # Total training timesteps
    total_timesteps: int = 100_000

    # Learning rate
    learning_rate: float = 0.0003

    # Batch size
    batch_size: int = 64

    # Number of steps to run for each environment per update
    n_steps: int = 2048

    # Discount factor
    gamma: float = 0.99

    # GAE (Generalized Advantage Estimation) lambda
    gae_lambda: float = 0.95

    # PPO clip range
    clip_range: float = 0.2

    # Entropy coefficient for the loss calculation
    ent_coef: float = 0.0

    # Value function coefficient for the loss calculation
    vf_coef: float = 0.5

    # Maximum norm for gradient clipping
    max_grad_norm: float = 0.5

    # Policy network architecture (shared between pi and vf)
    policy_kwargs: dict[str, Any] | None = None

    # Random seed for reproducibility
    seed: int = 42

    # Device: "auto", "cpu", "cuda"
    device: str = "auto"

    def __post_init__(self) -> None:
        """Set default policy_kwargs if not provided."""
        if self.policy_kwargs is None:
            object.__setattr__(self, "policy_kwargs", dict(net_arch=[64, 64]))


__all__ = ["TrainConfig"]
