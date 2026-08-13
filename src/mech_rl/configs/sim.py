"""Hydra structured config for SimParams.

This module provides a Hydra-compatible dataclass that converts to the
validated SimParams Pydantic model at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from mech_rl.domain.parameters import SimParams


@dataclass
class SimConfig:
    """Structured config for simulator parameters.

    Mirrors SimParams exactly. Use `to_pydantic()` to get the validated model.
    """

    # Integration timestep (s)
    dt: float = 0.01

    # Episode length cap
    max_episode_steps: int = 1000

    # Integrator name: "semi_implicit_euler" or "rk4"
    integrator: str = "semi_implicit_euler"

    @classmethod
    def to_pydantic(cls, cfg: SimConfig | dict) -> SimParams:
        """Convert structured config to validated SimParams model."""
        if isinstance(cfg, SimConfig):
            data = cfg.__dict__
        else:
            data = cfg
        return SimParams(**data)


__all__ = ["SimConfig"]
