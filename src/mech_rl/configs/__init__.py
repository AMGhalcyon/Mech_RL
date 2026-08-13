"""Hydra structured configs for Mech_RL.

These dataclasses provide type-safe config definitions that Hydra uses with
its ConfigStore. Each module exports a dataclass that mirrors the corresponding
Pydantic model in mech_rl.domain.parameters.

Usage:
    from hydra import initialize, compose
    from mech_rl.configs import RobotConfig, SimConfig, RewardConfig

    with initialize():
        cfg = compose(config_name="config")
        robot_params = RobotConfig.to_pydantic(cfg.robot)
        sim_params = SimConfig.to_pydantic(cfg.sim)
        reward_params = RewardConfig.to_pydantic(cfg.reward)
"""

from mech_rl.configs.reward import RewardConfig
from mech_rl.configs.robot import RobotConfig
from mech_rl.configs.sim import SimConfig

__all__ = ["RobotConfig", "SimConfig", "RewardConfig"]
