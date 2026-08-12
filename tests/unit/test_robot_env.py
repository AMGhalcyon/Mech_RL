"""Tests for RobotEnv — mock-free, real physics at every step."""

from __future__ import annotations

import numpy as np
import pytest

from mech_rl.domain.parameters import RewardParams, RobotParams, SimParams
from mech_rl.domain.state import RobotState
from mech_rl.environment.robot_env import RobotEnv

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def robot_params() -> RobotParams:
    return RobotParams(
        l1=0.3, l2=0.3, m1=1.0, m2=1.0, i1=0.03, i2=0.03, friction=0.05, max_torque=5.0
    )


@pytest.fixture
def sim_params() -> SimParams:
    return SimParams(dt=0.01, max_episode_steps=100, integrator="semi_implicit_euler")


@pytest.fixture
def reward_params() -> RewardParams:
    return RewardParams(distance_coef=1.0, effort_coef=0.01, success_bonus=10.0, time_penalty=0.01)


@pytest.fixture
def env(robot_params: RobotParams, sim_params: SimParams, reward_params: RewardParams) -> RobotEnv:
    target = np.array([0.3, 0.0])
    return RobotEnv(robot_params, sim_params, reward_params, target=target)


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestConstruction:
    def test_observation_space_keys(self, env: RobotEnv):
        assert set(env.observation_space.keys()) == {"q", "qdot", "target"}

    def test_observation_space_shapes(self, env: RobotEnv):
        for key in ("q", "qdot", "target"):
            assert env.observation_space[key].shape == (2,)

    def test_action_space_shape(self, env: RobotEnv):
        assert env.action_space.shape == (2,)

    def test_action_space_bounds(self, env: RobotEnv, robot_params: RobotParams):
        np.testing.assert_allclose(env.action_space.low, -robot_params.max_torque)
        np.testing.assert_allclose(env.action_space.high, robot_params.max_torque)


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------


class TestReset:
    def test_returns_obs_and_info(self, env: RobotEnv):
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_obs_keys_match_space(self, env: RobotEnv):
        obs, _ = env.reset(seed=0)
        for key in env.observation_space:
            assert key in obs

    def test_obs_within_bounds(self, env: RobotEnv):
        for seed in range(20):
            obs, _ = env.reset(seed=seed)
            assert env.observation_space["q"].contains(obs["q"])
            assert env.observation_space["target"].contains(obs["target"])

    def test_deterministic_with_seed(self, env: RobotEnv):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1["q"], obs2["q"])
        np.testing.assert_array_equal(obs1["target"], obs2["target"])

    def test_custom_reset_position(self, env: RobotEnv):
        q0 = np.array([0.5, -0.5])
        obs, _ = env.reset(seed=0, options={"reset_position": q0})
        np.testing.assert_array_almost_equal(obs["q"], q0)

    def test_custom_target_via_options(self, env: RobotEnv):
        target = np.array([0.1, 0.2])
        obs, _ = env.reset(seed=0, options={"target": target})
        np.testing.assert_array_almost_equal(obs["target"], target)

    def test_zero_velocities_after_reset(self, env: RobotEnv):
        obs, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs["qdot"], np.zeros(2))


# ------------------------------------------------------------------
# Step
# ------------------------------------------------------------------


class TestStep:
    def test_returns_five_tuple(self, env: RobotEnv):
        env.reset(seed=0)
        result = env.step(np.zeros(2))
        assert len(result) == 5

    def test_obs_in_bounds_after_step(self, env: RobotEnv):
        env.reset(seed=0)
        for _ in range(10):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert env.observation_space["q"].contains(obs["q"])

    def test_truncated_at_max_episode_steps(self, robot_params: RobotParams):
        sim = SimParams(dt=0.01, max_episode_steps=5, integrator="semi_implicit_euler")
        rp = RewardParams()
        env = RobotEnv(robot_params, sim, rp, target=np.array([0.3, 0.0]))
        env.reset(seed=0)
        for _ in range(5):
            _, _, _, truncated, _ = env.step(np.zeros(2))
        assert truncated

    def test_info_contains_ee_pose(self, env: RobotEnv):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(2))
        assert "ee_pose" in info
        assert len(info["ee_pose"]) == 3

    def test_info_contains_distance(self, env: RobotEnv):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(2))
        assert "distance" in info
        assert isinstance(info["distance"], float)


# ------------------------------------------------------------------
# Reward
# ------------------------------------------------------------------


class TestReward:
    def test_distance_penalty_increases_with_distance(self, robot_params: RobotParams):
        sim = SimParams(dt=0.01, max_episode_steps=1000, integrator="semi_implicit_euler")
        rp = RewardParams(distance_coef=1.0, effort_coef=0.0, success_bonus=0.0, time_penalty=0.0)

        # Near target.
        env_near = RobotEnv(robot_params, sim, rp, target=np.array([0.3, 0.0]))
        env_near.reset(seed=0, options={"reset_position": np.array([0.29, 0.0])})
        _, r_near, _, _, _ = env_near.step(np.zeros(2))

        # Far from target.
        env_far = RobotEnv(robot_params, sim, rp, target=np.array([0.3, 0.0]))
        env_far.reset(seed=0, options={"reset_position": np.array([-0.3, 0.0])})
        _, r_far, _, _, _ = env_far.step(np.zeros(2))

        assert r_far < r_near

    def test_effort_penalty_for_nonzero_torque(self, env: RobotEnv):
        env.reset(seed=0)
        _, r_zero, _, _, _ = env.step(np.zeros(2))
        env.reset(seed=0)
        _, r_nonzero, _, _, _ = env.step(np.array([2.0, 2.0]))
        assert r_nonzero < r_zero

    def test_success_bonus_when_close(self, robot_params: RobotParams):
        sim = SimParams(dt=0.01, max_episode_steps=1000, integrator="semi_implicit_euler")
        rp = RewardParams(distance_coef=0.0, effort_coef=0.0, success_bonus=10.0, success_radius=0.05)

        # q=[0, 0] puts ee at (l1+l2, 0) = (0.6, 0).  Target at (0.3, 0)
        # is far from the ee — no bonus.
        env_fail = RobotEnv(robot_params, sim, rp, target=np.array([0.3, 0.0]))
        env_fail.reset(seed=0, options={"reset_position": np.array([0.0, 0.0])})
        _, r_fail, _, _, _ = env_fail.step(np.zeros(2))
        assert r_fail < 10.0

        # Target at ee position (0.6, 0) → distance ≈0, bonus applies.
        env_bonus = RobotEnv(robot_params, sim, rp, target=np.array([0.6, 0.0]))
        env_bonus.reset(seed=0, options={"reset_position": np.array([0.0, 0.0])})
        _, r_bonus, _, _, _ = env_bonus.step(np.zeros(2))
        assert r_bonus >= 10.0

    def test_time_penalty_each_step(self, robot_params: RobotParams):
        sim = SimParams(dt=0.01, max_episode_steps=1000, integrator="semi_implicit_euler")
        rp = RewardParams(distance_coef=0.0, effort_coef=0.0, time_penalty=0.1, success_bonus=0.0)
        target = np.array([0.3, 0.0])
        env = RobotEnv(robot_params, sim, rp, target=target)
        env.reset(seed=0, options={"reset_position": np.array([0.0, 0.0])})

        _, r1, _, _, _ = env.step(np.zeros(2))
        _, r2, _, _, _ = env.step(np.zeros(2))
        # Both should be negative due to time penalty; second step has same
        # distance so rewards should be equal (distance penalty is identical).
        assert r1 < 0
        assert r2 < 0


# ------------------------------------------------------------------
# Termination
# ------------------------------------------------------------------


class TestTermination:
    def test_terminated_when_close_to_target(self, robot_params: RobotParams):
        sim = SimParams(dt=0.01, max_episode_steps=1000, integrator="semi_implicit_euler")
        rp = RewardParams(success_radius=0.05)
        # q=[0, 0] puts ee at (l1+l2, 0) = (0.6, 0).  Target there → terminated.
        env = RobotEnv(robot_params, sim, rp, target=np.array([0.6, 0.0]))
        env.reset(seed=0, options={"reset_position": np.array([0.0, 0.0])})
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
        assert terminated is True
        assert truncated is False

    def test_not_terminated_when_far(self, env: RobotEnv):
        env.reset(seed=0, options={"reset_position": np.array([0.0, 0.0])})
        # target is (0.3, 0), ee at q=[0,0] is at (0.6, 0), distance=0.3 >> 0.05
        _, _, terminated, _, _ = env.step(np.zeros(2))
        assert terminated is False


# ------------------------------------------------------------------
# Integrator dispatch
# ------------------------------------------------------------------


class TestIntegratorDispatch:
    @pytest.mark.parametrize("integrator_name", ["semi_implicit_euler", "rk4"])
    def test_integrator_by_name(self, robot_params: RobotParams, integrator_name: str):
        sim = SimParams(dt=0.01, max_episode_steps=10, integrator=integrator_name)
        rp = RewardParams()
        env = RobotEnv(robot_params, sim, rp, target=np.array([0.3, 0.0]))
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(np.array([1.0, -1.0]))
        assert env.observation_space["q"].contains(obs["q"])

    def test_different_integrators_give_different_states(self, robot_params: RobotParams):
        target = np.array([0.3, 0.0])
        q0 = np.array([0.5, -0.3])

        env_e = RobotEnv(
            robot_params,
            SimParams(dt=0.01, max_episode_steps=10, integrator="semi_implicit_euler"),
            RewardParams(),
            target=target,
        )
        env_e.reset(seed=0, options={"reset_position": q0})
        _, _, _, _, info_e = env_e.step(np.array([2.0, -1.0]))

        env_r = RobotEnv(
            robot_params,
            SimParams(dt=0.01, max_episode_steps=10, integrator="rk4"),
            RewardParams(),
            target=target,
        )
        env_r.reset(seed=0, options={"reset_position": q0})
        _, _, _, _, info_r = env_r.step(np.array([2.0, -1.0]))

        # The ee poses should differ (RK4 vs SI-Euler).
        assert info_e["ee_pose"] != info_r["ee_pose"]


# ------------------------------------------------------------------
# Physics consistency
# ------------------------------------------------------------------


class TestPhysicsConsistency:
    def test_state_matches_manual_integration(self, robot_params: RobotParams):
        """Verify env state after a step matches a manual integrator call."""
        sim = SimParams(dt=0.01, max_episode_steps=100, integrator="semi_implicit_euler")
        rp = RewardParams()
        target = np.array([0.3, 0.0])
        env = RobotEnv(robot_params, sim, rp, target=target)
        env.reset(seed=0, options={"reset_position": np.array([0.5, 0.5])})

        from mech_rl.physics import semi_implicit_euler as si_euler

        q0 = np.array([0.5, 0.5])
        qdot0 = np.zeros(2)
        state0 = RobotState(q=q0, qdot=qdot0)
        tau = np.array([1.0, -0.5])
        expected = si_euler(state0, tau, robot_params, 0.01)

        obs, _, _, _, _ = env.step(tau)
        np.testing.assert_array_almost_equal(obs["q"], expected.q)
        np.testing.assert_array_almost_equal(obs["qdot"], expected.qdot)

    def test_multi_step_consistency(self, robot_params: RobotParams):
        """Run 20 random steps in the env and compare against manual integration."""
        sim = SimParams(dt=0.01, max_episode_steps=100, integrator="semi_implicit_euler")
        rp = RewardParams()
        target = np.array([0.3, 0.0])
        env = RobotEnv(robot_params, sim, rp, target=target)
        env.reset(seed=42, options={"reset_position": np.array([0.0, 0.0])})

        from mech_rl.physics import semi_implicit_euler as si_euler

        state = RobotState(q=np.zeros(2), qdot=np.zeros(2))
        rng = np.random.default_rng(99)

        for _ in range(20):
            tau = rng.uniform(-2.0, 2.0, size=2)
            state = si_euler(state, tau, robot_params, 0.01)
            obs, _, _, _, _ = env.step(tau)
            np.testing.assert_array_almost_equal(obs["q"], state.q)
            np.testing.assert_array_almost_equal(obs["qdot"], state.qdot)
