"""Tests for the time integrators."""

from __future__ import annotations

import numpy as np
import pytest

from mech_rl.domain.state import RobotState
from mech_rl.physics.dynamics import mass_matrix, potential_energy
from mech_rl.physics.integrator import rk4, semi_implicit_euler


def _energy(state: RobotState, params) -> float:
    """Total mechanical energy ``KE + PE`` for the arm."""
    ke = 0.5 * state.qdot @ mass_matrix(state.q, params) @ state.qdot
    pe = potential_energy(state.q, params)
    return ke + pe


class TestSemiImplicitEuler:
    def test_step_shape(self, zero_state, default_robot_params):
        new_state = semi_implicit_euler(zero_state, np.zeros(2), default_robot_params, dt=0.01)
        assert new_state.q.shape == (2,)
        assert new_state.qdot.shape == (2,)

    def test_zero_state_zero_torque_stays_zero(self, frictionless_robot_params):
        # With no torque and friction=0, choose q aligned with gravity
        # (q=[pi/2, 0]) so qddot == 0 -> integrator produces the same state.
        s = RobotState(q=np.array([np.pi / 2.0, 0.0]), qdot=np.zeros(2))
        for _ in range(100):
            s = semi_implicit_euler(s, np.zeros(2), frictionless_robot_params, dt=0.001)
        np.testing.assert_allclose(s.q, np.array([np.pi / 2.0, 0.0]), atol=1e-9)
        np.testing.assert_allclose(s.qdot, np.zeros(2), atol=1e-9)

    def test_energy_conservation(self, frictionless_robot_params):
        # Free-motion energy drift on a frictionless arm. Semi-implicit
        # Euler should hold |dE/E| well below 1% over 1 s of *bounded*
        # motion. The arm starts at q=0 (horizontal) with a small kick;
        # the gravitational potential (~6 J) dominates KE so the arm
        # swings downward and oscillates rather than whirling through
        # multiple revolutions.
        q0 = np.zeros(2)
        qdot0 = np.array([0.5, 0.0])
        s = RobotState(q=q0, qdot=qdot0)
        e0 = _energy(s, frictionless_robot_params)
        assert abs(e0) > 1e-3  # non-trivial initial energy

        dt = 0.001
        n_steps = 1000
        for _ in range(n_steps):
            s = semi_implicit_euler(s, np.zeros(2), frictionless_robot_params, dt=dt)
        e1 = _energy(s, frictionless_robot_params)
        rel_drift = abs(e1 - e0) / abs(e0)
        assert rel_drift < 1e-2

    def test_no_nan_with_torque(self, default_robot_params):
        s = RobotState(q=np.array([0.1, 0.2]), qdot=np.array([0.1, 0.0]))
        for _ in range(10_000):
            s = semi_implicit_euler(s, np.array([0.5, -0.5]), default_robot_params, dt=0.001)
        assert np.all(np.isfinite(s.q))
        assert np.all(np.isfinite(s.qdot))

    def test_negative_dt_rejected(self, zero_state, default_robot_params):
        with pytest.raises(ValueError, match="dt must be positive"):
            semi_implicit_euler(zero_state, np.zeros(2), default_robot_params, dt=-0.01)

    def test_tau_shape_rejected(self, zero_state, default_robot_params):
        with pytest.raises(ValueError, match="tau must have shape"):
            semi_implicit_euler(zero_state, np.zeros(3), default_robot_params, dt=0.01)


class TestRK4:
    def test_step_shape(self, zero_state, default_robot_params):
        new_state = rk4(zero_state, np.zeros(2), default_robot_params, dt=0.01)
        assert new_state.q.shape == (2,)
        assert new_state.qdot.shape == (2,)

    def test_zero_state_zero_torque_stays_zero(self, frictionless_robot_params):
        s = RobotState(q=np.array([np.pi / 2.0, 0.0]), qdot=np.zeros(2))
        for _ in range(100):
            s = rk4(s, np.zeros(2), frictionless_robot_params, dt=0.001)
        np.testing.assert_allclose(s.q, np.array([np.pi / 2.0, 0.0]), atol=1e-9)
        np.testing.assert_allclose(s.qdot, np.zeros(2), atol=1e-9)

    def test_energy_conservation(self, frictionless_robot_params):
        # RK4 is 4th-order -> energy drift ~ dt^4 per step. Over 1 s with
        # dt=0.001, relative drift should be negligible.
        q0 = np.zeros(2)
        qdot0 = np.array([0.5, 0.0])
        s = RobotState(q=q0, qdot=qdot0)
        e0 = _energy(s, frictionless_robot_params)

        dt = 0.001
        n_steps = 1000
        for _ in range(n_steps):
            s = rk4(s, np.zeros(2), frictionless_robot_params, dt=dt)
        e1 = _energy(s, frictionless_robot_params)
        rel_drift = abs(e1 - e0) / abs(e0)
        assert rel_drift < 1e-5

    def test_more_accurate_than_euler(self, frictionless_robot_params):
        # Truth: tiny-step RK4. Compare both integrators at dt=0.01.
        q0 = np.zeros(2)
        qdot0 = np.array([0.5, 0.0])
        truth_steps = 1000
        truth_dt = 0.0001
        s_truth = RobotState(q=q0, qdot=qdot0)
        for _ in range(truth_steps):
            s_truth = rk4(s_truth, np.zeros(2), frictionless_robot_params, dt=truth_dt)
        truth = s_truth.q

        coarse_steps = truth_steps // 10
        s_euler = RobotState(q=q0, qdot=qdot0)
        for _ in range(coarse_steps):
            s_euler = semi_implicit_euler(
                s_euler, np.zeros(2), frictionless_robot_params, dt=truth_dt * 10
            )

        s_rk4 = RobotState(q=q0, qdot=qdot0)
        for _ in range(coarse_steps):
            s_rk4 = rk4(s_rk4, np.zeros(2), frictionless_robot_params, dt=truth_dt * 10)

        err_euler = float(np.linalg.norm(s_euler.q - truth))
        err_rk4 = float(np.linalg.norm(s_rk4.q - truth))
        assert err_rk4 < err_euler

    def test_no_nan_with_torque(self, default_robot_params):
        s = RobotState(q=np.array([0.1, 0.2]), qdot=np.array([0.1, 0.0]))
        for _ in range(10_000):
            s = rk4(s, np.array([0.5, -0.5]), default_robot_params, dt=0.001)
        assert np.all(np.isfinite(s.q))
        assert np.all(np.isfinite(s.qdot))

    def test_negative_dt_rejected(self, zero_state, default_robot_params):
        with pytest.raises(ValueError, match="dt must be positive"):
            rk4(zero_state, np.zeros(2), default_robot_params, dt=-0.01)

    def test_tau_shape_rejected(self, zero_state, default_robot_params):
        with pytest.raises(ValueError, match="tau must have shape"):
            rk4(zero_state, np.zeros(3), default_robot_params, dt=0.01)
