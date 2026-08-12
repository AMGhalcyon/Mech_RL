"""Tests for dynamics — M, C, G, F, PE, EOM assembly."""

from __future__ import annotations

import numpy as np
import pytest

from mech_rl.physics.dynamics import (
    coriolis,
    equation_of_motion,
    friction,
    gravity,
    mass_matrix,
    potential_energy,
)


class MassMatrixTests:
    def test_shape(self, default_robot_params):
        m = mass_matrix(np.zeros(2), default_robot_params)
        assert m.shape == (2, 2)

    def test_symmetric(self, default_robot_params):
        q = np.array([0.4, -0.3])
        m = mass_matrix(q, default_robot_params)
        np.testing.assert_allclose(m, m.T)

    def test_positive_definite(self, default_robot_params):
        m = mass_matrix(np.array([0.4, -0.3]), default_robot_params)
        eigvals = np.linalg.eigvalsh(m)
        assert np.all(eigvals > 0.0)

    def test_m11_constant(self, default_robot_params):
        m_a = mass_matrix(np.zeros(2), default_robot_params)
        m_b = mass_matrix(np.array([0.5, 1.2]), default_robot_params)
        np.testing.assert_allclose(m_a[1, 1], m_b[1, 1])

    def test_m00_varies_with_cos_q1(self, default_robot_params):
        m_straight = mass_matrix(np.array([0.0, 0.0]), default_robot_params)
        m_bent = mass_matrix(np.array([0.0, np.pi]), default_robot_params)
        assert m_straight[0, 0] > m_bent[0, 0]


class CoriolisTests:
    def test_zero_at_rest(self, default_robot_params):
        c = coriolis(np.array([0.4, 0.7]), np.zeros(2), default_robot_params)
        np.testing.assert_allclose(c, np.zeros(2), atol=1e-12)

    def test_zero_at_singularity(self, default_robot_params):
        qdot = np.array([1.0, 0.5])
        for q1 in (0.0, np.pi):
            c = coriolis(np.array([0.4, q1]), qdot, default_robot_params)
            np.testing.assert_allclose(c, np.zeros(2), atol=1e-12)


class GravityTests:
    def test_shoulder_torque_positive_at_zero(self, default_robot_params):
        g = gravity(np.zeros(2), default_robot_params)
        assert g[0] > 0.0

    def test_zero_when_vertical(self, default_robot_params):
        g = gravity(np.array([np.pi / 2.0, 0.0]), default_robot_params)
        np.testing.assert_allclose(g, np.zeros(2), atol=1e-9)


class FrictionTests:
    def test_proportional_to_velocity(self, default_robot_params):
        qdot = np.array([2.0, -3.0])
        f = friction(qdot, default_robot_params)
        expected = default_robot_params.friction * qdot
        np.testing.assert_allclose(f, expected)

    def test_zero_at_rest(self, default_robot_params):
        np.testing.assert_allclose(friction(np.zeros(2), default_robot_params), np.zeros(2))

    def test_zero_friction_coefficient(self, frictionless_robot_params):
        np.testing.assert_allclose(
            friction(np.array([1.0, 1.0]), frictionless_robot_params), np.zeros(2)
        )


class PotentialEnergyTests:
    def test_zero_at_reference_config(self, default_robot_params):
        u = potential_energy(np.array([-np.pi / 2.0, 0.0]), default_robot_params)
        np.testing.assert_allclose(u, 0.0, atol=1e-9)

    def test_max_at_straight_up(self, default_robot_params):
        u_up = potential_energy(np.array([np.pi / 2.0, 0.0]), default_robot_params)
        u_down = potential_energy(np.array([-np.pi / 2.0, 0.0]), default_robot_params)
        assert u_up > u_down

    def test_gradient_matches_gravity(self, default_robot_params):
        q = np.array([0.3, -0.4])
        eps = 1e-6
        u0 = potential_energy(q, default_robot_params)
        g_num = np.array(
            [
                (potential_energy(q + np.array([eps, 0.0]), default_robot_params) - u0) / eps,
                (potential_energy(q + np.array([0.0, eps]), default_robot_params) - u0) / eps,
            ]
        )
        g_analytic = gravity(q, default_robot_params)
        np.testing.assert_allclose(g_num, g_analytic, atol=1e-4)


class EquationOfMotionTests:
    def test_zero_torque_zero_state_stays_zero(self, frictionless_robot_params):
        qddot = equation_of_motion(
            np.array([np.pi / 2.0, 0.0]),
            np.zeros(2),
            np.zeros(2),
            frictionless_robot_params,
        )
        np.testing.assert_allclose(qddot, np.zeros(2), atol=1e-9)

    def test_inverse_consistency(self, default_robot_params):
        q = np.array([0.3, -0.4])
        qdot = np.array([0.5, -0.2])
        qddot = np.array([1.0, 2.0])
        tau = (
            mass_matrix(q, default_robot_params) @ qddot
            + coriolis(q, qdot, default_robot_params)
            + gravity(q, default_robot_params)
            + friction(qdot, default_robot_params)
        )
        qddot_recovered = equation_of_motion(q, qdot, tau, default_robot_params)
        np.testing.assert_allclose(qddot_recovered, qddot, atol=1e-9)

    def test_invalid_shape(self, default_robot_params):
        with pytest.raises(ValueError, match="q must have shape"):
            equation_of_motion(np.zeros(3), np.zeros(2), np.zeros(2), default_robot_params)
