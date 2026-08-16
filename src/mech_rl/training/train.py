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
from mech_rl.tracking.mlflow_tracker import (
    finish_run,
    log_metrics,
    log_model,
    log_params,
    start_run,
)
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

    # Setup TensorBoard logging
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    tensorboard_log = str(output_dir / "tensorboard")

    # Instantiate PPO with TensorBoard logging
    model = PPO(
        policy="MultiInputPolicy",
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
        tensorboard_log=tensorboard_log,
    )

    # Train with checkpoint callback
    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1000, cfg.train.total_timesteps // 10),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="rl_model",
        verbose=1,
    )
    model.learn(
        total_timesteps=cfg.train.total_timesteps,
        callback=checkpoint_cb,
    )

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

    # Start MLflow run
    start_run(experiment_name="mech_rl")
    try:
        # Log hyperparameters
        log_params({
            **cfg.train,
            "robot/l1": cfg.robot.l1,
            "robot/l2": cfg.robot.l2,
            "sim/dt": cfg.sim.dt,
            "reward/distance_coef": cfg.reward.distance_coef,
        })

        # Log sweep information if running under Hydra multirun
        try:
            from hydra.core.hydra_config import HydraConfig
            hydra_cfg = HydraConfig.get()
            # Job number/index within the sweep
            if hydra_cfg.job.num is not None:
                log_params({"sweep/job_num": hydra_cfg.job.num})
            # Hydra stores the sweep config under cfg.hydra. Sweep params appear as
            # cfg.hydra.runtime.choices or can be inferred from the overrides.
            # Simpler approach: log any config key that differs from the defaults
            # as a swept parameter (requires default config accessible).
            # For now, log the explicit overrides passed via command line.
            if hydra_cfg.overrides.task:
                # Store the raw overrides as a param for reproducibility
                log_params({"sweep/overrides": " ".join(hydra_cfg.overrides.task)})
        except Exception:
            # Hydra config not available or other issue - continue without sweep logging
            pass

        # Train
        print("\nStarting training...")
        model = train(cfg)
        print("Training complete!")

        # Evaluate on a few episodes (optional)
        from mech_rl.evaluation.eval_loop import evaluate
        eval_env = instantiate_env(cfg)
        mean_reward = evaluate(model, eval_env, num_episodes=5)
        print(f"Mean evaluation reward: {mean_reward:.2f}")
        log_metrics({"eval_mean_reward": mean_reward})

        # Save evaluation results for sweep analysis
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        eval_results = {
            "eval_mean_reward": float(mean_reward),
            "total_timesteps": cfg.train.total_timesteps,
            "learning_rate": cfg.train.learning_rate,
            # Add any other params that might be swept? We'll leave it to the analysis function to read the config.
        }
        import json
        (output_dir / "eval_results.json").write_text(json.dumps(eval_results, indent=2))

        # Save model
        model_path = output_dir / "model.zip"
        model.save(str(model_path))
        print(f"Model saved to: {model_path}")
        log_model(model_path, artifact_path="model")

    finally:
        # End MLflow run
        finish_run()


# Make this module callable as a module: `python -m mech_rl.training.train`
if __name__ == "__main__":
    main()
