"""FK and Jacobian for the 2-DOF planar arm. q0=shoulder, q1=elbow, both from prev link.
Arm at q=[0,0] points along +x of base frame. Jacobian is geometric (position-level)."""

from __future__ import annotations

import numpy as np

from mech_rl.domain.parameters import RobotParams
from mech_rl.domain.state import EndEffectorPose
from mech_rl.domain.types import as_array


def forward_kinematics(q: np.ndarray, params: RobotParams) -> EndEffectorPose:
    """End-effector pose (x, y, theta) in base frame. Straightforward planar FK."""
    q = as_array(q)
    if q.shape != (2,):
        raise ValueError(f"q must have shape (2,), got {q.shape}")

    c0, s0 = np.cos(q[0]), np.sin(q[0])
    c01, s01 = np.cos(q[0] + q[1]), np.sin(q[0] + q[1])
    l1, l2 = params.l1, params.l2

    x = l1 * c0 + l2 * c01
    y = l1 * s0 + l2 * s01
    theta = q[0] + q[1]

    return EndEffectorPose(x=float(x), y=float(y), theta=float(theta))


def jacobian(q: np.ndarray, params: RobotParams) -> np.ndarray:
    """Geometric Jacobian J(q) — (2,2). ee_vel = J @ qdot.
    Singular at q1=pi (arm folded flat): both link velocities cancel along arm line."""
    q = as_array(q)
    if q.shape != (2,):
        raise ValueError(f"q must have shape (2,), got {q.shape}")

    s0 = np.sin(q[0])
    s01 = np.sin(q[0] + q[1])
    c0 = np.cos(q[0])
    c01 = np.cos(q[0] + q[1])
    l1, l2 = params.l1, params.l2

    return np.array(
        [
            [-l1 * s0 - l2 * s01, -l2 * s01],
            [l1 * c0 + l2 * c01, l2 * c01],
        ],
        dtype=np.float64,
    )


__all__ = ["forward_kinematics", "jacobian"]
