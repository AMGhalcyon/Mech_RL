"""Tests for the Hydra training entry point.

Verifies that the environment can be instantiated from a Hydra config
and that the training module exports the expected functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mech_rl.configs.train import TrainConfig
from mech_rl.training.train import instantiate_env, make_env


@dataclass
class FakeRobotCfg:
    l1: float = 0.3
    l2: float = 0.3
    m1: float = 1.0
    m2: float = 1.0
    i1: float = 0.03
    i2: float = 0.03
    friction: float = 0.05
    max_torque: float = 5.0


@dataclass
class FakeSimCfg:
    dt: float = 0.01
    max_episode_steps: int = 100
    integrator: str = "semi_implicit_euler"


@dataclass
class FakeRewardCfg:
    distance_coef: float = 1.0
    effort_coef: float = 0.01
    smoothness_coef: float = 0.0
    success_bonus: float = 0.0
    success_radius: float = 0.05
    time_penalty: float = 0.0


@dataclass
class FakeTrainCfg:
    algorithm: str = "PPO"
    total_timesteps: int = 100
    learning_rate: float = 0.0003
    batch_size: int = 64
    n_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: dict | None = None
    seed: int = 42
    device: str = "auto"


@dataclass
class FakeCfg:
    """A minimal config object that mimics Hydra's DictConfig interface."""
    robot: FakeRobotCfg = field(default_factory=FakeRobotCfg)
    sim: FakeSimCfg = field(default_factory=FakeSimCfg)
    reward: FakeRewardCfg = field(default_factory=FakeRewardCfg)
    train: FakeTrainCfg = field(default_factory=FakeTrainCfg)


class TestInstantiateEnv:
    """Tests for instantiate_env()."""

    def test_builds_env_from_config(self):
        """instantiate_env returns a valid RobotEnv."""
        cfg = FakeCfg()
        env = instantiate_env(cfg)

        assert env.robot_params.l1 == 0.3
        assert env.robot_params.max_torque == 5.0
        assert env.sim_params.integrator == "semi_implicit_euler"
        assert env.sim_params.max_episode_steps == 100

    def test_env_reset_works(self):
        """Environment built from config can reset."""
        cfg = FakeCfg()
        env = instantiate_env(cfg)
        obs, info = env.reset(seed=42)

        assert "q" in obs
        assert "qdot" in obs
        assert "target" in obs
        assert isinstance(info, dict)

    def test_env_step_works(self):
        """Environment built from config can take a step."""
        cfg = FakeCfg()
        env = instantiate_env(cfg)
        env.reset(seed=0)

        obs, reward, terminated, truncated, info = env.step([1.0, 0.5])

        assert "q" in obs
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestMakeEnv:
    """Tests for make_env()."""

    def test_make_env_returns_env(self):
        """make_env returns a RobotEnv instance."""
        cfg = FakeCfg()
        env = make_env(cfg, seed=42)

        assert env is not None

    def test_make_env_with_seed(self):
        """make_env with seed produces deterministic reset."""
        cfg = FakeCfg()
        env1 = make_env(cfg, seed=123)
        env2 = make_env(cfg, seed=123)

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)

        import numpy as np
        np.testing.assert_array_equal(obs1["q"], obs2["q"])


class TestTrainConfigValidation:
    """Tests that training config validates properly."""

    def test_train_config_defaults(self):
        """TrainConfig has sensible defaults."""
        cfg = TrainConfig()
        assert cfg.algorithm == "PPO"
        assert cfg.total_timesteps > 0
        assert cfg.learning_rate > 0
        assert cfg.batch_size > 0

    def test_policy_kwargs_default(self):
        """Policy kwargs default to a reasonable architecture."""
        cfg = TrainConfig()
        assert cfg.policy_kwargs is not None
        assert "net_arch" in cfg.policy_kwargs
