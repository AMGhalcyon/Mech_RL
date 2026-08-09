"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from mech_rl.domain.parameters import RewardParams, RobotParams, SimParams
from mech_rl.domain.state import RobotState


@pytest.fixture
def default_robot_params() -> RobotParams:
    """Reasonable defaults for a 2-DOF arm."""
    return RobotParams(
        l1=0.3,
        l2=0.3,
        m1=1.0,
        m2=1.0,
        i1=0.01,
        i2=0.01,
        friction=0.05,
        max_torque=5.0,
    )


@pytest.fixture
def default_sim_params() -> SimParams:
    return SimParams(
        dt=0.01,
        max_episode_steps=1000,
        integrator="semi_implicit_euler",
    )


@pytest.fixture
def default_reward_params() -> RewardParams:
    return RewardParams()


@pytest.fixture
def zero_state() -> RobotState:
    """Arm hanging straight down (both joints at zero)."""
    return RobotState(q=np.zeros(2), qdot=np.zeros(2))


@pytest.fixture
def bent_state() -> RobotState:
    """Elbow at 90 degrees, shoulder at zero."""
    return RobotState(q=np.array([0.0, np.pi / 2]), qdot=np.zeros(2))
