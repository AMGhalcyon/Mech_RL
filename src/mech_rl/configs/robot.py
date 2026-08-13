"""Hydra structured config for RobotParams.

This module provides a Hydra-compatible dataclass that converts to the
validated RobotParams Pydantic model at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from mech_rl.domain.parameters import RobotParams


@dataclass
class RobotConfig:
    """Structured config for robot physical parameters.

    Mirrors RobotParams exactly. Use `to_pydantic()` to get the validated model.
    """

    # Link lengths (m)
    l1: float = 0.3
    l2: float = 0.3

    # Link masses (kg)
    m1: float = 1.0
    m2: float = 1.0

    # Moment of inertia about the joint, parallel-axis included (kg*m^2)
    i1: float = 0.03
    i2: float = 0.03

    # Viscous friction coefficient (N*m*s/rad)
    friction: float = 0.05

    # Per-joint torque limit magnitude (N*m)
    max_torque: float = 5.0

    @classmethod
    def to_pydantic(cls, cfg: RobotConfig | dict) -> RobotParams:
        """Convert structured config to validated RobotParams model."""
        if isinstance(cfg, RobotConfig):
            data = cfg.__dict__
        else:
            data = cfg
        return RobotParams(**data)


__all__ = ["RobotConfig"]
