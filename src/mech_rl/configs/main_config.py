"""Top-level Hydra structured config for Mech_RL.

This dataclass combines the individual config structs (robot, sim, reward, train)
into a single config that Hydra can load and validate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mech_rl.configs.reward import RewardConfig
from mech_rl.configs.robot import RobotConfig
from mech_rl.configs.sim import SimConfig
from mech_rl.configs.train import TrainConfig


@dataclass
class MechRLConfig:
    """Top-level config for Mech_RL experiments.

    Attributes:
        robot: Robot physical parameters.
        sim: Simulator parameters.
        reward: Reward function coefficients.
        train: Training hyperparameters.
    """

    robot: RobotConfig = field(default_factory=RobotConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


__all__ = ["MechRLConfig"]
