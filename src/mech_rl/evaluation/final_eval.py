"""Final evaluation suite for Mech_RL — release-ready."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
from mech_rl.domain.parameters import RobotParams, RewardParams, SimParams
from mech_rl.environment import RobotEnv
from mech_rl.evaluation.eval_loop import evaluate
from mech_rl.visualization import visualize_learned_policy, record_gif
from mech_rl.configs import robot, sim, reward


def _compute_success_and_length(model, env: RobotEnv, num_episodes: int) -> tuple[float, float]:
    """Compute success rate and mean episode length for a model.

    Args:
        model: Trained Stable-Baselines3 model.
        env: RobotEnv instance.
        num_episodes: Number of evaluation episodes.

    Returns:
        Tuple of (success_rate, mean_episode_length)
    """
    successes = 0
    total_length = 0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_length += 1

            # Check if episode ended due to success (terminated=True and not truncated)
            if terminated and not truncated:
                successes += 1

        total_length += episode_length

    success_rate = successes / num_episodes if num_episodes > 0 else 0.0
    mean_episode_length = total_length / num_episodes if num_episodes > 0 else 0.0

    return success_rate, mean_episode_length


def run_final_evaluation(
    model,
    output_dir: str | Path = "outputs/final_eval",
    num_episodes: int = 10,
    robustness_targets: int = 5,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use the exact configs from the best training run (Day 10/11 1M steps)
    robot_params = robot.RobotConfig.to_pydantic(robot.RobotConfig())
    sim_params = sim.SimConfig.to_pydantic(sim.SimConfig())
    # Reward coefficients as used in training (from hydra config of the 1M run)
    reward_params = reward.RewardConfig(
        distance_coef=0.5,
        effort_coef=0.01,
        smoothness_coef=0.0,
        success_bonus=0.0,
        success_radius=0.05,
        time_penalty=0.01,
    )
    # Build environment with rgb_array rendering for GIF generation
    env = RobotEnv(
        robot_params=robot_params,
        sim_params=sim_params,
        reward_params=reward_params,
        render_mode="rgb_array",
    )
    # Core evaluation (mean reward)
    mean_reward = evaluate(model, env, num_episodes=num_episodes)
    # Compute success rate and mean episode length ourselves (since env does not set info['is_success'])
    success_rate, mean_length = _compute_success_and_length(model, env, num_episodes)
    # Detailed episode data (for plotting and GIF)
    episode_data = visualize_learned_policy(
        model, env, num_episodes=num_episodes,
        output_dir=output_dir / "episodes",
        render_mode="rgb_array",
    )
    # Generate GIF (reuses visualization logic)
    gif_result = record_gif(
        model, env, num_episodes=3,
        output_dir=output_dir / "videos",
        fps=15,
    )
    gif_path = gif_result.get("gif_path")
    # Robustness tests: target variations and link-length perturbations
    robustness = {}
    # Vary target positions (sample from env's target sampler or use fixed grid)
    target_variations = []
    for i in range(robustness_targets):
        # Use a deterministic seed for reproducibility
        seed = 1000 + i
        # Sample a target using the same method as the env (if available) else pick a point in workspace
        try:
            # If env has a method to sample target (not exposed), we fallback to uniform in [-0.5,0.5]
            target = np.random.RandomState(seed).uniform(-0.5, 0.5, size=2)
        except Exception:
            target = np.array([0.3, 0.3])
        env_reset = RobotEnv(
            robot_params=robot_params,
            sim_params=sim_params,
            reward_params=reward_params,
            target=target,
            render_mode=None,
        )
        r = evaluate(model, env_reset, num_episodes=3)
        target_variations.append({
            "target": [float(target[0]), float(target[1])],
            "mean_reward": float(r),
        })
    robustness["target_variation"] = target_variations
    # Link-length perturbations (±5%)
    perturbed_rewards = []
    base_params = robot_params
    for factor in [0.95, 1.0, 1.05]:
        pert_params = RobotParams(
            l1=base_params.l1 * factor,
            l2=base_params.l2 * factor,
            m1=base_params.m1,
            m2=base_params.m2,
            i1=base_params.i1,
            i2=base_params.i2,
            friction=base_params.friction,
            max_torque=base_params.max_torque,
        )
        pert_env = RobotEnv(
            robot_params=pert_params,
            sim_params=sim_params,
            reward_params=reward_params,
            render_mode=None,
        )
        r = evaluate(model, pert_env, num_episodes=3)
        perturbed_rewards.append({
            "l_scale": factor,
            "mean_reward": float(r),
        })
    robustness["link_perturbation"] = perturbed_rewards
    # Assemble summary dictionary
    summary = {
        "success_rate": float(success_rate),
        "mean_reward": float(mean_reward),
        "mean_episode_length": float(mean_length),
        "num_episodes": num_episodes,
        "gif_path": str(gif_path) if gif_path else None,
        "robustness": robustness,
    }
    # Write JSON summary
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    # Write simple HTML report
    html_content = f"""<!DOCTYPE html>
<html>
<head><title>Mech_RL Final Evaluation Report</title></head>
<body>
<h1>Mech_RL Final Evaluation Report</h1>
<p>Mean Reward: {mean_reward:.2f}</p>
<p>Success Rate: {success_rate:.2%}</p>
<p>Mean Episode Length: {mean_length:.1f} steps</p>
<p>Summary JSON: <a href="eval_summary.json">eval_summary.json</a></p>
<p>GIF: <a href="{gif_path or 'N/A'}">{gif_path or 'N/A'}</a></p>
</body>
</html>"""
    (output_dir / "report.html").write_text(html_content, encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to trained SB3 model (.zip)")
    parser.add_argument("--output-dir", default="outputs/final_eval", help="Directory to save results")
    args = parser.parse_args()
    print("Final evaluation suite ready. Use run_final_evaluation() with a loaded model.")