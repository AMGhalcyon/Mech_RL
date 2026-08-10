"""Equation of motion for the 2-DOF planar arm.

The dynamics are written in the manipulator form

    M(q) qddot + C(q, qdot) qdot + G(q) + F(qdot) = tau

where:
- ``M(q)``   is the (2, 2) symmetric positive-definite inertia matrix
- ``C(q, qdot) qdot`` is the Coriolis / centripetal vector (we expose the
  velocity-product form, not the matrix ``C`` itself)
- ``G(q)``   is the gravitational torque
- ``F(qdot)``is the viscous friction torque (``friction * qdot``)
- ``tau``    is the applied joint torque

Gravity is taken along ``-y`` of the base frame. The module-level constant
``G_ACCEL`` defaults to 9.81 m/s^2; we do not thread it through
``RobotParams`` yet to keep Day2's surface area small.

Both links are modelled as uniform rods with the centre of mass at the
midpoint; rotational inertia about the joint is taken from the supplied
``i1`` and ``i2``. If a future variant needs non-uniform mass distributions,
swap the closed-form expressions for symbolic ones — the public API does
not change.
"""

from __future__ import annotations

import numpy as np

from mech_rl.domain.parameters import RobotParams
from mech_rl.domain.types import as_array

G_ACCEL: float = 9.81
"""Gravitational acceleration magnitude (m/s^2). Direction is ``-y``."""


def mass_matrix(q: np.ndarray, params: RobotParams) -> np.ndarray:
    """Return the (2, 2) symmetric positive-definite inertia matrix ``M(q)``.

    Both ``M[0, 0]`` and ``M[0, 1]`` carry the same ``cos(q1)`` dependence
    (only the magnitude differs). ``M[1, 1]`` is constant.

    The rotational inertias ``i1`` and ``i2`` are interpreted as the link's
    moment of inertia *about its own joint*, parallel-axis included — so the
    translational CoM contributions are already folded in.
    """
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
    """Return ``C(q, qdot) qdot`` as a 2-vector.

    Computed from the energy-conserving Lagrangian form
    ``C qdot = M_dot qdot - 0.5 grad_q(qdot^T M qdot)``. For this planar
    arm only the ``q1``-dependence of ``M`` contributes; the result scales
    with ``sin(q1)``.

    The returned vector is the unique quadratic-in-``qdot`` Coriolis term
    that, when plugged into ``M qddot + C qdot + G = tau``, yields a
    conservative mechanical system.
    """
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

    # From the Lagrangian form:
    #   C qdot = M_dot qdot - 0.5 grad_q(qdot^T M qdot)
    # M only depends on q1, so the gradient lives entirely in q1.
    grad_qdotT_M_qdot_q1 = -h * (qd0 * qd0 + qd0 * qd1)
    m_dot_qdot_0 = -h * qd0 * qd1 - 0.5 * h * qd1 * qd1
    m_dot_qdot_1 = -0.5 * h * qd0 * qd1  # M[1, 0] = M[0, 1] contributes too

    return np.array(
        [
            m_dot_qdot_0,  # grad_q wrt q0 is zero
            m_dot_qdot_1 - 0.5 * grad_qdotT_M_qdot_q1,
        ],
        dtype=np.float64,
    )


def gravity(q: np.ndarray, params: RobotParams) -> np.ndarray:
    """Return gravitational torque ``G(q)`` as a 2-vector.

    With both joints at zero, the arm points along +x and gravity (in -y)
    pulls it down, so ``G[0] > 0`` (counter-clockwise shoulder torque to
    hold the arm up).
    """
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
    """Return viscous friction torque ``friction * qdot`` as a 2-vector."""
    qdot = as_array(qdot)
    if qdot.shape != (2,):
        raise ValueError(f"qdot must have shape (2,), got {qdot.shape}")
    return np.array([params.friction * qdot[0], params.friction * qdot[1]], dtype=np.float64)


def potential_energy(q: np.ndarray, params: RobotParams) -> float:
    """Return gravitational potential energy ``U(q)`` in joules.

    Reference: ``U = 0`` when the arm hangs straight down at
    ``q = [-pi/2, 0]``. An additive constant is subtracted so that the
    gravitational torque ``G = dU/dq`` is unchanged — energy-conservation
    tests only need ``dU`` to be invariant under the shift.
    """
    q = as_array(q)
    if q.shape != (2,):
        raise ValueError(f"q must have shape (2,), got {q.shape}")

    l1, l2 = params.l1, params.l2
    m1, m2 = params.m1, params.m2
    g = G_ACCEL

    # y-CoM of each link in the base frame, and the same when the arm hangs
    # straight down (q = [-pi/2, 0]).
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
    """Solve ``M(q) qddot = tau - C - G - F`` for ``qddot``.

    ``np.linalg.solve`` on a (2, 2) dense system is faster than forming an
    inverse, and we already know ``M`` is non-singular by construction.
    """
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
