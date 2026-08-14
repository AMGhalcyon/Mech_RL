"""Tests for Hydra config loading and validation.

Verifies that YAML configs load correctly into Pydantic models
via the Hydra structured config system.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mech_rl.configs.reward import RewardConfig
from mech_rl.configs.robot import RobotConfig
from mech_rl.configs.sim import SimConfig
from mech_rl.configs.train import TrainConfig
from mech_rl.domain.parameters import RewardParams, RobotParams, SimParams


class TestRobotConfig:
    """Tests for RobotConfig structured config."""

    def test_default_values(self):
        """Default config values are sensible defaults."""
        cfg = RobotConfig()
        assert cfg.l1 == 0.3
        assert cfg.l2 == 0.3
        assert cfg.m1 == 1.0
        assert cfg.m2 == 1.0
        assert cfg.i1 == 0.03
        assert cfg.i2 == 0.03
        assert cfg.friction == 0.05
        assert cfg.max_torque == 5.0

    def test_to_pydantic(self):
        """to_pydantic() returns a validated RobotParams model."""
        cfg = RobotConfig()
        params = RobotConfig.to_pydantic(cfg)

        assert isinstance(params, RobotParams)
        assert params.l1 == cfg.l1
        assert params.l2 == cfg.l2

    def test_to_pydantic_from_dict(self):
        """to_pydantic() works with a plain dict."""
        data = {
            "l1": 0.4,
            "l2": 0.35,
            "m1": 1.2,
            "m2": 1.0,
            "i1": 0.04,
            "i2": 0.035,
            "friction": 0.02,
            "max_torque": 6.0,
        }
        params = RobotConfig.to_pydantic(data)

        assert isinstance(params, RobotParams)
        assert params.l1 == 0.4
        assert params.max_torque == 6.0


class TestSimConfig:
    """Tests for SimConfig structured config."""

    def test_default_values(self):
        """Default config values are sensible defaults."""
        cfg = SimConfig()
        assert cfg.dt == 0.01
        assert cfg.max_episode_steps == 1000
        assert cfg.integrator == "semi_implicit_euler"

    def test_to_pydantic(self):
        """to_pydantic() returns a validated SimParams model."""
        cfg = SimConfig()
        params = SimConfig.to_pydantic(cfg)

        assert isinstance(params, SimParams)
        assert params.dt == cfg.dt
        assert params.integrator == cfg.integrator

    def test_validates_integrator(self):
        """Invalid integrator raises validation error."""
        with pytest.raises(ValidationError):
            SimConfig.to_pydantic({"dt": 0.01, "max_episode_steps": 1000, "integrator": "invalid"})


class TestRewardConfig:
    """Tests for RewardConfig structured config."""

    def test_default_values(self):
        """Default config values match RewardParams defaults."""
        cfg = RewardConfig()
        assert cfg.distance_coef == 1.0
        assert cfg.effort_coef == 0.01
        assert cfg.smoothness_coef == 0.0
        assert cfg.success_bonus == 0.0
        assert cfg.success_radius == 0.05
        assert cfg.time_penalty == 0.0

    def test_to_pydantic(self):
        """to_pydantic() returns a validated RewardParams model."""
        cfg = RewardConfig()
        params = RewardConfig.to_pydantic(cfg)

        assert isinstance(params, RewardParams)
        assert params.distance_coef == cfg.distance_coef


class TestTrainConfig:
    """Tests for TrainConfig structured config."""

    def test_default_values(self):
        """Default config values are sensible defaults."""
        cfg = TrainConfig()
        assert cfg.algorithm == "PPO"
        assert cfg.total_timesteps == 100_000
        assert cfg.learning_rate == 0.0003
        assert cfg.batch_size == 64
        assert cfg.device == "auto"

    def test_policy_kwargs_default(self):
        """policy_kwargs defaults to a reasonable architecture."""
        cfg = TrainConfig()
        assert cfg.policy_kwargs == {"net_arch": [64, 64]}

    def test_policy_kwargs_can_be_set(self):
        """policy_kwargs can be overridden."""
        cfg = TrainConfig(policy_kwargs={"net_arch": [128, 128]})
        assert cfg.policy_kwargs["net_arch"] == [128, 128]


class TestYAMLFileLoad:
    """Tests for loading actual YAML files from the configs/ directory."""

    def test_robot_yaml_loads(self):
        """Robot YAML file can be loaded."""
        from pathlib import Path

        import yaml

        config_path = Path("configs/robot.yaml")
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            params = RobotParams(**data)
            assert params.l1 == 0.3
            assert params.max_torque == 5.0

    def test_sim_yaml_loads(self):
        """Sim YAML file can be loaded."""
        from pathlib import Path

        import yaml

        config_path = Path("configs/sim.yaml")
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            params = SimParams(**data)
            assert params.dt == 0.01
            assert params.integrator == "semi_implicit_euler"

    def test_reward_yaml_loads(self):
        """Reward YAML file can be loaded."""
        from pathlib import Path

        import yaml

        config_path = Path("configs/reward.yaml")
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            params = RewardParams(**data)
            assert params.distance_coef == 1.0
