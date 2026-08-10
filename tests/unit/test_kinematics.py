"""Tests for the kinematics layer."""

from __future__ import annotations

import numpy as np

from mech_rl.physics.kinematics import forward_kinematics, jacobian


def test_fk_at_zero(default_robot_params):
    # q=0 -> arm straight along +x, end-effector at (l1+l2, 0, 0).
    p = forward_kinematics(np.zeros(2), default_robot_params)
    expected_x = default_robot_params.l1 + default_robot_params.l2
    np.testing.assert_allclose(p.x, expected_x)
    np.testing.assert_allclose(p.y, 0.0)
    np.testing.assert_allclose(p.theta, 0.0)


def test_fk_at_bent(default_robot_params):
    # q = [0, pi/2] -> elbow bent up; ee at (l1, l2, pi/2).
    q = np.array([0.0, np.pi / 2.0])
    p = forward_kinematics(q, default_robot_params)
    np.testing.assert_allclose(p.x, default_robot_params.l1, atol=1e-12)
    np.testing.assert_allclose(p.y, default_robot_params.l2, atol=1e-12)
    np.testing.assert_allclose(p.theta, np.pi / 2.0)


def test_fk_at_straight_up(default_robot_params):
    # q = [pi/2, 0] -> arm straight up; ee at (0, l1+l2, pi/2).
    q = np.array([np.pi / 2.0, 0.0])
    p = forward_kinematics(q, default_robot_params)
    np.testing.assert_allclose(p.x, 0.0, atol=1e-12)
    np.testing.assert_allclose(p.y, default_robot_params.l1 + default_robot_params.l2)
    np.testing.assert_allclose(p.theta, np.pi / 2.0)


def test_jacobian_shape(default_robot_params):
    j = jacobian(np.zeros(2), default_robot_params)
    assert j.shape == (2, 2)


def test_jacobian_matches_finite_difference(default_robot_params):
    # J @ dq should match the FK position difference to O(eps).
    q = np.array([0.3, -0.7])
    eps = 1e-6
    j = jacobian(q, default_robot_params)

    base = forward_kinematics(q, default_robot_params)
    perturbed = forward_kinematics(q + np.array([eps, 0.0]), default_robot_params)
    finite_diff = np.array(
        [
            (perturbed.x - base.x) / eps,
            (perturbed.y - base.y) / eps,
        ]
    )
    analytic_col0 = j @ np.array([1.0, 0.0])
    np.testing.assert_allclose(analytic_col0, finite_diff, atol=1e-5)

    perturbed = forward_kinematics(q + np.array([0.0, eps]), default_robot_params)
    finite_diff = np.array(
        [
            (perturbed.x - base.x) / eps,
            (perturbed.y - base.y) / eps,
        ]
    )
    analytic_col1 = j @ np.array([0.0, 1.0])
    np.testing.assert_allclose(analytic_col1, finite_diff, atol=1e-5)


def test_jacobian_singular_at_folded_config(default_robot_params):
    # q = [0, pi] -> arm folded back on itself. Both Jacobian columns point
    # in the same direction (along the arm), so det(J) == 0.
    q = np.array([0.0, np.pi])
    j = jacobian(q, default_robot_params)
    assert abs(np.linalg.det(j)) < 1e-10


def test_jacobian_invalid_shape(default_robot_params):
    import pytest

    with pytest.raises(ValueError, match="q must have shape"):
        jacobian(np.zeros(3), default_robot_params)


def test_fk_invalid_shape(default_robot_params):
    import pytest

    with pytest.raises(ValueError, match="q must have shape"):
        forward_kinematics(np.zeros(3), default_robot_params)
