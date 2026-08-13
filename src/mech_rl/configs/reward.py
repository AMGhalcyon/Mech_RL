"""Hydra structured config for RewardParams.

This module provides a Hydra-compatible dataclass that converts to the
validated RewardParams Pydantic model at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from mech_rl.domain.parameters import RewardParams


@dataclass
class RewardConfig:
    """Structured config for reward function coefficients.

    Mirrors RewardParams exactly. Use `to_pydantic()` to get the validated model.
    """

    # Distance penalty coefficient
    distance_coef: float = 1.0

    # Effort penalty coefficient
    effort_coef: float = 0.01

    # Smoothness penalty coefficient
    smoothness_coef: float = 0.0

    # Success bonus
    success_bonus: float = 0.0

    # Success radius (m)
    success_radius: float = 0.05

    # Time penalty per step
    time_penalty: float = 0.0

    @classmethod
    def to_pydantic(cls, cfg: RewardConfig | dict) -> RewardParams:
        """Convert structured config to validated RewardParams model."""
        if isinstance(cfg, RewardConfig):
            data = cfg.__dict__
        else:
            data = cfg
        return RewardParams(**data)


__all__ = ["RewardConfig"]
