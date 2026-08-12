"""Time integrators for the 2-DOF arm. Two schemes, same signature for env dispatch.

- semi_implicit_euler: symplectic, first-order, cheap. Velocity first, then position with new velocity.
- rk4: classical RK4, 4 acceleration evals/step. Lower energy drift.

Adaptive step-size is out of scope — env drives fixed dt from SimParams.
"""

from __future__ import annotations

import numpy as np

from mech_rl.domain.parameters import RobotParams
from mech_rl.domain.state import RobotState
from mech_rl.domain.types import as_array
from mech_rl.physics.dynamics import equation_of_motion


def _stack_state(q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """Concatenate ``q`` and ``qdot`` into the (4,) state vector RK4 needs."""
    return np.concatenate([q, qdot])


def _unstack_state(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`_stack_state`."""
    return y[:2], y[2:]


def _ode(y: np.ndarray, tau: np.ndarray, params: RobotParams) -> np.ndarray:
    """Right-hand side of the ODE ``dy/dt = [qdot; qddot(y, tau)]``.

    ``y`` is the (4,) state ``[q0, q1, qdot0, qdot1]``.
    """
    q, qdot = _unstack_state(y)
    qddot = equation_of_motion(q, qdot, tau, params)
    return np.concatenate([qdot, qddot])


def semi_implicit_euler(
    state: RobotState,
    tau: np.ndarray,
    params: RobotParams,
    dt: float,
) -> RobotState:
    """Semi-implicit (symplectic) Euler step. Velocity update first, then position with new velocity.
    Order matters — that's what makes it symplectic and energy-stable."""
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    tau = as_array(tau)
    if tau.shape != (2,):
        raise ValueError(f"tau must have shape (2,), got {tau.shape}")

    qddot = equation_of_motion(state.q, state.qdot, tau, params)
    qdot_new = state.qdot + dt * qddot
    q_new = state.q + dt * qdot_new

    return RobotState(q=q_new, qdot=qdot_new)


def rk4(
    state: RobotState,
    tau: np.ndarray,
    params: RobotParams,
    dt: float,
) -> RobotState:
    """Classical RK4 step. y' = [qdot; qddot]. 4 stages, standard weights.
    Much lower energy drift than SI-Euler over long horizons."""
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    tau = as_array(tau)
    if tau.shape != (2,):
        raise ValueError(f"tau must have shape (2,), got {tau.shape}")

    y = _stack_state(state.q, state.qdot)
    k1 = _ode(y, tau, params)
    k2 = _ode(y + 0.5 * dt * k1, tau, params)
    k3 = _ode(y + 0.5 * dt * k2, tau, params)
    k4 = _ode(y + dt * k3, tau, params)

    y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    q_new, qdot_new = _unstack_state(y_new)

    return RobotState(q=q_new, qdot=qdot_new)


__all__ = ["rk4", "semi_implicit_euler"]
