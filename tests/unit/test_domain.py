"""Tests for the domain layer (parameters + state)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from pydantic import ValidationError

from mech_rl.domain.parameters import RewardParams, RobotParams, SimParams
from mech_rl.domain.state import EndEffectorPose, RobotState


class TestRobotState:
    def test_construction_with_zeros(self):
        s = RobotState(q=np.zeros(2), qdot=np.zeros(2))
        np.testing.assert_array_equal(s.q, np.zeros(2))
        np.testing.assert_array_equal(s.qdot, np.zeros(2))

    def test_q_must_have_shape_two(self):
        with pytest.raises(ValueError, match="q must have shape"):
            RobotState(q=np.zeros(3), qdot=np.zeros(2))

    def test_qdot_must_have_shape_two(self):
        with pytest.raises(ValueError, match="qdot must have shape"):
            RobotState(q=np.zeros(2), qdot=np.zeros(3))

    def test_state_is_immutable(self):
        s = RobotState(q=np.zeros(2), qdot=np.zeros(2))
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            s.q = np.ones(2)  # type: ignore[misc]

    def test_angle_properties(self):
        s = RobotState(q=np.array([0.5, 1.0]), qdot=np.array([0.2, 0.3]))
        assert s.shoulder_angle == 0.5
        assert s.elbow_angle == 1.0
        assert s.shoulder_velocity == 0.2
        assert s.elbow_velocity == 0.3

    def test_dtypes_are_normalized(self):
        s = RobotState(q=[0.0, 0.0], qdot=[0.0, 0.0])  # lists, not arrays
        assert s.q.dtype == np.float64
        assert s.qdot.dtype == np.float64


class TestEndEffectorPose:
    def test_construction(self):
        p = EndEffectorPose(x=0.3, y=0.1, theta=0.5)
        assert p.x == 0.3
        assert p.y == 0.1
        assert p.theta == 0.5


class TestRobotParams:
    def test_default_construction(self, default_robot_params):
        assert default_robot_params.l1 > 0
        assert default_robot_params.friction >= 0

    def test_negative_length_rejected(self):
        with pytest.raises(ValidationError):
            RobotParams(
                l1=-0.1, l2=0.3, m1=1.0, m2=1.0,
                i1=0.01, i2=0.01, friction=0.05, max_torque=5.0,
            )

    def test_zero_mass_rejected(self):
        with pytest.raises(ValidationError):
            RobotParams(
                l1=0.3, l2=0.3, m1=0.0, m2=1.0,
                i1=0.01, i2=0.01, friction=0.05, max_torque=5.0,
            )

    def test_negative_torque_rejected(self):
        with pytest.raises(ValidationError):
            RobotParams(
                l1=0.3, l2=0.3, m1=1.0, m2=1.0,
                i1=0.01, i2=0.01, friction=0.05, max_torque=-1.0,
            )

    def test_frozen(self, default_robot_params):
        with pytest.raises(ValidationError):
            default_robot_params.l1 = 0.5  # type: ignore[misc]

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            RobotParams(
                l1=0.3, l2=0.3, m1=1.0, m2=1.0,
                i1=0.01, i2=0.01, friction=0.05, max_torque=5.0,
                unknown_field=42,
            )


class TestSimParams:
    def test_default(self, default_sim_params):
        assert default_sim_params.dt > 0

    def test_dt_must_be_positive(self):
        with pytest.raises(ValidationError):
            SimParams(dt=0.0, max_episode_steps=100)

    def test_dt_capped(self):
        with pytest.raises(ValidationError):
            SimParams(dt=1.0, max_episode_steps=100)  # too large

    def test_integrator_validated(self):
        with pytest.raises(ValidationError):
            SimParams(dt=0.01, max_episode_steps=100, integrator="verlet")


class TestRewardParams:
    def test_defaults(self, default_reward_params):
        assert default_reward_params.distance_coef >= 0
        assert default_reward_params.effort_coef >= 0

    def test_negative_coef_rejected(self):
        with pytest.raises(ValidationError):
            RewardParams(distance_coef=-0.1)

    def test_zero_success_radius_rejected(self):
        with pytest.raises(ValidationError):
            RewardParams(success_radius=0.0)
