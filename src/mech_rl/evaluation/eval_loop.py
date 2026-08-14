"""Evaluation loop for Mech_RL policies."""

from __future__ import annotations

from mech_rl.environment import RobotEnv


def evaluate(model, env: RobotEnv, num_episodes: int = 10) -> float:
    """Evaluate a trained SB3 model on held-out episodes.

    Args:
        model: Trained Stable-Baselines3 model.
        env: RobotEnv instance.
        num_episodes: Number of evaluation episodes.

    Returns:
        Mean cumulative reward across episodes.
    """
    total_reward = 0.0
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_reward += episode_reward
    return float(total_reward / num_episodes)
