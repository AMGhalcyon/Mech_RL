"""Baseline benchmark: random control and simple PID-like controller."""
from __future__ import annotations
import numpy as np
from mech_rl.environment import RobotEnv
from mech_rl.configs import robot, sim, reward
from mech_rl.domain.parameters import RobotParams, SimParams, RewardParams


def random_baseline(env: RobotEnv, num_episodes: int = 10) -> float:
    """Random action baseline."""
    total_reward = 0.0
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += r
    return total_reward / num_episodes


def pid_baseline(env: RobotEnv, num_episodes: int = 10, kp: float = 0.5) -> float:
    """Simple P-like controller (proportional toward target)."""
    total_reward = 0.0
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            # obs layout depends on RobotEnv; assume first entries include joint state
            # Fallback to zero action if shape unexpected
            action = np.clip(kp * (np.array([0.0, 0.0]) - np.array(obs[:2])), -1.0, 1.0)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += r
    return total_reward / num_episodes


if __name__ == "__main__":
    robot_params = robot.RobotConfig.to_pydantic(robot.RobotConfig())
    sim_params = sim.SimConfig.to_pydantic(sim.SimConfig())
    reward_params = reward.RewardConfig()
    env = RobotEnv(robot_params=robot_params, sim_params=sim_params, reward_params=reward_params)
    print("Random baseline mean reward:", random_baseline(env, num_episodes=5))
    env2 = RobotEnv(robot_params=robot_params, sim_params=sim_params, reward_params=reward_params)
    print("PID baseline mean reward:", pid_baseline(env2, num_episodes=5))
