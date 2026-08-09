"""Robot state representation.

`RobotState` is the canonical state passed between physics, environment,
and reward layers. It is a frozen dataclass — once a state exists, it
cannot be mutated. New states are produced by the physics layer at each
integration step.

This is the entire shared vocabulary for "what the robot looks like
right now." Everything that needs a state, takes a `RobotState`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mech_rl.domain.types import Radians, RadiansPerSecond, as_array


@dataclass(frozen=True)
class EndEffectorPose:
    """Position and orientation of the tool tip in the base frame.

    For a 2-DOF arm in the plane, orientation is a single scalar (the
    end-effector angle). Position is (x, y) in meters.
    """

    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class RobotState:
    """Joint-space state of a 2-DOF arm.

    All angles in radians, all velocities in rad/s. Convention:
    - `q[0]` is the shoulder angle (base joint)
    - `q[1]` is the elbow angle

    Immutability is enforced by `frozen=True`; a new state must be
    constructed to "change" anything.
    """

    q: np.ndarray  # shape (2,)
    qdot: np.ndarray  # shape (2,)

    def __post_init__(self) -> None:
        # frozen=True prevents __setattr__, but __post_init__ runs once
        # at construction, so we can validate the shape here.
        q = as_array(self.q)
        qdot = as_array(self.qdot)
        if q.shape != (2,):
            raise ValueError(f"q must have shape (2,), got {q.shape}")
        if qdot.shape != (2,):
            raise ValueError(f"qdot must have shape (2,), got {qdot.shape}")
        # Replace with normalized dtype.
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "qdot", qdot)

    @property
    def shoulder_angle(self) -> Radians:
        return Radians(float(self.q[0]))

    @property
    def elbow_angle(self) -> Radians:
        return Radians(float(self.q[1]))

    @property
    def shoulder_velocity(self) -> RadiansPerSecond:
        return RadiansPerSecond(float(self.qdot[0]))

    @property
    def elbow_velocity(self) -> RadiansPerSecond:
        return RadiansPerSecond(float(self.qdot[1]))


__all__ = ["EndEffectorPose", "RobotState"]
