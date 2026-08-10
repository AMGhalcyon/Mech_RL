"""Forward kinematics and analytic Jacobian for the 2-DOF planar arm.

Convention (matches ``domain/state.py``):
- ``q[0]`` is the shoulder angle, ``q[1]`` is the elbow angle, both measured
  from the previous link.
- With both at zero, the arm points along the +x axis of the base frame.

The Jacobian here is the geometric (position-level) Jacobian ``J(q)`` such that
the end-effector linear velocity equals ``J @ qdot``. Orientation rate follows
from ``theta = q0 + q1``.
"""

from __future__ import annotations

import numpy as np

from mech_rl.domain.parameters import RobotParams
from mech_rl.domain.state import EndEffectorPose
from mech_rl.domain.types import as_array


def forward_kinematics(q: np.ndarray, params: RobotParams) -> EndEffectorPose:
    """Return end-effector pose in the base frame.

    For a planar 2-DOF arm:
        x      = l1*cos(q0) + l2*cos(q0 + q1)
        y      = l1*sin(q0) + l2*sin(q0 + q1)
        theta  = q0 + q1
    """
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
    """Return the (2, 2) geometric Jacobian ``J(q)``.

    ``ee_velocity = J @ qdot`` where ``ee_velocity`` is the (vx, vy) of the
    end-effector. Columns correspond to ``qdot[0]`` and ``qdot[1]``.

    Singular when the arm is folded flat: ``q = [a, pi]`` for any ``a``,
    because both link contributions to velocity cancel along the arm line.
    """
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
