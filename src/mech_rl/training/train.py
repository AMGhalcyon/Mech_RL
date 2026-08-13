"""Hydra-based training entry point for Mech_RL.

This module provides the main entry point for training PPO policies
on the torque-controlled 2-DOF robotic arm. It uses Hydra for config
management and Stable-Baselines3 for PPO training.

Usage:
    # Run with defaults
    .venv/Scripts/python.exe -m mech_rl.training.train

    # Override config values
    .venv/Scripts/python.exe -m mech_rl.training.train robot.l1=0.4 sim.integrator=rk4

    # Use different config
    .venv/Scripts/python.exe -m mech_rl.training.train --config-name train
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from mech_rl.configs.main_config import MechRLConfig
from mech_rl.configs.reward import RewardConfig
from mech_rl.configs.robot import RobotConfig
from mech_rl.configs.sim import SimConfig
from mech_rl.configs.train import TrainConfig
from mech_rl.domain.parameters import RewardParams, RobotParams, SimParams
from mech_rl.environment import RobotEnv
from mech_rl.utils.reproducibility import set_seed

if TYPE_CHECKING:
    from stable_baselines3 import PPO

# Register the config with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=MechRLConfig)
cs.store(name="train", node=TrainConfig)
cs.store(group="robot", name="default", node=RobotConfig)
cs.store(group="sim", name="default", node=SimConfig)
cs.store(group="reward", name="default", node=RewardConfig)


def instantiate_env(cfg: DictConfig) -> RobotEnv:
    """Build RobotEnv from Hydra config.

    Args:
        cfg: Hydra config with robot, sim, reward sections.

    Returns:
        Instantiated RobotEnv with validated parameters.
    """
    robot_params = RobotParams(
        l1=cfg.robot.l1,
        l2=cfg.robot.l2,
        m1=cfg.robot.m1,
        m2=cfg.robot.m2,
        i1=cfg.robot.i1,
        i2=cfg.robot.i2,
        friction=cfg.robot.friction,
        max_torque=cfg.robot.max_torque,
    )

    sim_params = SimParams(
        dt=cfg.sim.dt,
        max_episode_steps=cfg.sim.max_episode_steps,
        integrator=cfg.sim.integrator,
    )

    reward_params = RewardParams(
        distance_coef=cfg.reward.distance_coef,
        effort_coef=cfg.reward.effort_coef,
        smoothness_coef=cfg.reward.smoothness_coef,
        success_bonus=cfg.reward.success_bonus,
        success_radius=cfg.reward.success_radius,
        time_penalty=cfg.reward.time_penalty,
    )

    return RobotEnv(robot_params, sim_params, reward_params)


def make_env(cfg: DictConfig, seed: int | None = None) -> RobotEnv:
    """Create a seeded environment instance.

    Args:
        cfg: Hydra config.
        seed: Optional seed for the environment.

    Returns:
        RobotEnv instance.
    """
    env = instantiate_env(cfg)
    if seed is not None:
        env.reset(seed=seed)
    return env


def train(cfg: DictConfig) -> PPO:
    """Train a PPO policy on the robotic arm environment.

    This is the skeleton for Day 5+. Currently a placeholder that
    instantiates the environment and returns a policy placeholder.

    Args:
        cfg: Hydra config with all training parameters.

    Returns:
        Trained PPO policy.
    """
    from stable_baselines3 import PPO

    # Set seed for reproducibility
    set_seed(cfg.train.seed)

    # Create vectorized environment
    env = make_env(cfg)

    # Get policy kwargs
    policy_kwargs = cfg.train.policy_kwargs or dict(net_arch=[64, 64])

    # Instantiate PPO
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.train.learning_rate,
        n_steps=cfg.train.n_steps,
        batch_size=cfg.train.batch_size,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
        clip_range=cfg.train.clip_range,
        ent_coef=cfg.train.ent_coef,
        vf_coef=cfg.train.vf_coef,
        max_grad_norm=cfg.train.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=cfg.train.device,
    )

    # Train
    model.learn(total_timesteps=cfg.train.total_timesteps)

    return model


# Hydra entry point decorator
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    Parses config, instantiates environment, trains policy.
    """
    # Instantiate and verify the environment
    env = instantiate_env(cfg)
    print(f"Environment created: {type(env).__name__}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Demonstrate reset
    obs, _ = env.reset(seed=cfg.train.seed)
    print(f"Initial observation keys: {list(obs.keys())}")

    # Train
    print("\nStarting training...")
    model = train(cfg)
    print("Training complete!")

    # Save model
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    model_path = output_dir / "model.zip"
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")


# Make this module callable as a module: `python -m mech_rl.training.train`
if __name__ == "__main__":
    main()
