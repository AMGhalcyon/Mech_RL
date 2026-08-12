"""2-DOF arm dynamics — M(q) qddot + C(q,qdot)qdot + G(q) + F(qdot) = tau.

Gravity acts along -y. G_ACCEL = 9.81 hardcoded for now (TODO: promote to RobotParams).
Links are uniform rods, CoM at midpoint. i1/i2 are joint inertias (parallel-axis included).
"""

from __future__ import annotations

import numpy as np

from mech_rl.domain.parameters import RobotParams
from mech_rl.domain.types import as_array

G_ACCEL = 9.81  # m/s^2, direction -y


def mass_matrix(q: np.ndarray, params: RobotParams) -> np.ndarray:
    """Inertia matrix M(q). Symmetric, PD. Only M[0,0] and M[0,1] vary with q1 (cos(q1)).
    M[1,1] = i2 is constant. i1/i2 are joint inertias (parallel-axis already folded in)."""
    q = as_array(q)
    if q.shape != (2,):
        raise ValueError(f"q must have shape (2,), got {q.shape}")

    l1, l2 = params.l1, params.l2
    m2 = params.m2
    i1, i2 = params.i1, params.i2

    beta = m2 * l1 * l2
    c1 = np.cos(q[1])

    m00 = i1 + i2 + m2 * l1 * l1 + beta * c1
    m01 = i2 + 0.5 * beta * c1
    m11 = i2

    return np.array(
        [
            [m00, m01],
            [m01, m11],
        ],
        dtype=np.float64,
    )


def coriolis(q: np.ndarray, qdot: np.ndarray, params: RobotParams) -> np.ndarray:
    """C(q, qdot) qdot as 2-vector. Energy-conserving Lagrangian form.
    Only q1-dependence of M contributes — scales with sin(q1). Zero at q1=0 or pi (singularity)."""
    q = as_array(q)
    qdot = as_array(qdot)
    if q.shape != (2,):
        raise ValueError(f"q must have shape (2,), got {q.shape}")
    if qdot.shape != (2,):
        raise ValueError(f"qdot must have shape (2,), got {qdot.shape}")

    l1, l2 = params.l1, params.l2
    m2 = params.m2
    beta = m2 * l1 * l2
    s1 = np.sin(q[1])
    h = beta * s1  # shorthand: scales every Coriolis component

    qd0 = qdot[0]
    qd1 = qdot[1]

    # Lagrangian form: C qdot = M_dot qdot - 0.5 grad_q(qdot^T M qdot)
    # M only depends on q1 -> gradient lives entirely in q1
    grad_qdotT_M_qdot_q1 = -h * (qd0 * qd0 + qd0 * qd1)
    m_dot_qdot_0 = -h * qd0 * qd1 - 0.5 * h * qd1 * qd1
    m_dot_qdot_1 = -0.5 * h * qd0 * qd1  # M[1,0] = M[0,1] contributes too

    return np.array(
        [
            m_dot_qdot_0,  # grad_q wrt q0 is zero
            m_dot_qdot_1 - 0.5 * grad_qdotT_M_qdot_q1,
        ],
        dtype=np.float64,
    )


def gravity(q: np.ndarray, params: RobotParams) -> np.ndarray:
    """Gravitational torque G(q). At q=0 arm points +x, gravity -y -> G[0] > 0 (CCW shoulder)."""
    q = as_array(q)
    if q.shape != (2,):
        raise ValueError(f"q must have shape (2,), got {q.shape}")

    l1, l2 = params.l1, params.l2
    m1, m2 = params.m1, params.m2
    g = G_ACCEL

    c0 = np.cos(q[0])
    c01 = np.cos(q[0] + q[1])

    g0 = g * (m1 * (l1 / 2.0) * c0 + m2 * (l1 * c0 + (l2 / 2.0) * c01))
    g1 = g * m2 * (l2 / 2.0) * c01

    return np.array([g0, g1], dtype=np.float64)


def friction(qdot: np.ndarray, params: RobotParams) -> np.ndarray:
    """Viscous friction = friction * qdot. Simple, linear."""
    qdot = as_array(qdot)
    if qdot.shape != (2,):
        raise ValueError(f"qdot must have shape (2,), got {qdot.shape}")
    return np.array([params.friction * qdot[0], params.friction * qdot[1]], dtype=np.float64)


def potential_energy(q: np.ndarray, params: RobotParams) -> float:
    """Gravitational PE in joules. U=0 at reference config (arm straight down, q=[-pi/2, 0]).
    Shift is arbitrary — only dU matters for energy-conservation tests."""
    q = as_array(q)
    if q.shape != (2,):
        raise ValueError(f"q must have shape (2,), got {q.shape}")

    l1, l2 = params.l1, params.l2
    m1, m2 = params.m1, params.m2
    g = G_ACCEL

    # y-CoM of each link, minus the reference config (arm straight down)
    y_c1 = (l1 / 2.0) * np.sin(q[0])
    y_c2 = l1 * np.sin(q[0]) + (l2 / 2.0) * np.sin(q[0] + q[1])
    y_c1_ref = -(l1 / 2.0)
    y_c2_ref = -(l1 + l2 / 2.0)

    u = g * (m1 * (y_c1 - y_c1_ref) + m2 * (y_c2 - y_c2_ref))
    return float(u)


def equation_of_motion(
    q: np.ndarray,
    qdot: np.ndarray,
    tau: np.ndarray,
    params: RobotParams,
) -> np.ndarray:
    """Solve M(q) qddot = tau - C - G - F for qddot.
    np.linalg.solve on 2x2 is faster than inv, and M is non-singular by construction."""
    q = as_array(q)
    qdot = as_array(qdot)
    tau = as_array(tau)
    for name, arr in (("q", q), ("qdot", qdot), ("tau", tau)):
        if arr.shape != (2,):
            raise ValueError(f"{name} must have shape (2,), got {arr.shape}")

    m = mass_matrix(q, params)
    c = coriolis(q, qdot, params)
    g = gravity(q, params)
    f = friction(qdot, params)

    rhs = tau - c - g - f
    qddot = np.linalg.solve(m, rhs)
    return qddot


__all__ = [
    "G_ACCEL",
    "coriolis",
    "equation_of_motion",
    "friction",
    "gravity",
    "mass_matrix",
    "potential_energy",
]
