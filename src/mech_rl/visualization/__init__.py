"""Visualization utilities for Mech_RL.

Provides plotting scripts for TensorBoard logs, training curves,
and learned policy visualization (arm animation, trajectory plots).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mech_rl.domain.parameters import RobotParams
from mech_rl.environment import RobotEnv
from mech_rl.physics.kinematics import forward_kinematics


def tensorboard_scalars_to_csv(logdir: str, output_dir: str | None = None) -> dict[str, Path]:
    """Extract scalar data from TensorBoard event files and save as CSV.

    Args:
        logdir: Path to TensorBoard log directory (contains events.out.tfevents.* files)
        output_dir: Directory to save CSV files. If None, creates 'tensorboard_csv' in logdir

    Returns:
        Dictionary mapping tag names to CSV file paths
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("tensorboard not installed. Install with: pip install tensorboard") from None

    logdir_path = Path(logdir)
    if output_dir is None:
        output_dir = logdir_path / "tensorboard_csv"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find event files
    event_files = list(logdir_path.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {logdir}")

    # Load event accumulator
    ea = EventAccumulator(str(logdir))
    ea.Reload()

    # Get scalar tags
    tags = ea.Tags()['scalars']

    csv_paths = {}
    for tag in tags:
        # Extract scalar events
        scalar_events = ea.Scalars(tag)
        if not scalar_events:
            continue

        # Convert to arrays
        steps = np.array([e.step for e in scalar_events])
        values = np.array([e.value for e in scalar_events])
        wall_time = np.array([e.wall_time for e in scalar_events])

        # Save as CSV
        safe_tag = re.sub(r'[^\w\-_]', '_', tag)
        csv_path = output_dir / f"{safe_tag}.csv"
        header = "step,value,wall_time"
        data = np.column_stack([steps, values, wall_time])
        np.savetxt(csv_path, data, delimiter=',', header=header, comments='')
        csv_paths[tag] = csv_path

    return csv_paths


def plot_training_curves(logdir: str, output_dir: str | None = None,
                        smooth: int = 10) -> dict[str, Path]:
    """Plot TensorBoard scalar data as training curves.

    Args:
        logdir: Path to TensorBoard log directory
        output_dir: Directory to save plots. If None, uses logdir/plots
        smooth: Window size for moving average smoothing (1 = no smoothing)

    Returns:
        Dictionary mapping tag names to plot file paths
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("tensorboard not installed. Install with: pip install tensorboard") from None

    logdir_path = Path(logdir)
    if output_dir is None:
        output_dir = logdir_path / "plots"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load event accumulator
    ea = EventAccumulator(str(logdir))
    ea.Reload()

    # Get scalar tags
    tags = ea.Tags()['scalars']

    plot_paths = {}
    for tag in tags:
        scalar_events = ea.Scalars(tag)
        if not scalar_events:
            continue

        steps = np.array([e.step for e in scalar_events])
        values = np.array([e.value for e in scalar_events])

        # Apply smoothing if requested
        if smooth > 1 and len(values) >= smooth:
            from numpy import convolve, ones
            weights = ones(smooth) / smooth
            values_smooth = convolve(values, weights, mode='valid')
            steps_smooth = steps[smooth-1:]
        else:
            values_smooth = values
            steps_smooth = steps

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps_smooth, values_smooth, linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel(tag.replace('/', ' :: ').title())
        plt.title(f'Training Curve: {tag}')
        plt.grid(True, alpha=0.3)

        # Save plot
        safe_tag = re.sub(r'[^\w\-_]', '_', tag)
        plot_path = output_dir / f"{safe_tag}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        plot_paths[tag] = plot_path

    return plot_paths


def visualize_learned_policy(model, env: RobotEnv, num_episodes: int = 3,
                           output_dir: str | None = None,
                           render_mode: str = 'rgb_array') -> dict[str, Any]:
    """Visualize a learned policy by running episodes and recording trajectories.

    Args:
        model: Trained Stable-Baselines3 model
        env: RobotEnv instance
        num_episodes: Number of episodes to visualize
        output_dir: Directory to save visualization outputs
        render_mode: Rendering mode for environment ('rgb_array' or None)

    Returns:
        Dictionary containing episode data (rewards, states, actions, etc.)
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get environment parameters for kinematics
    robot_params = env.robot_params

    episode_data = {
        'episodes': [],
        'success_rate': 0.0,
        'mean_reward': 0.0,
        'mean_episode_length': 0.0
    }

    total_reward = 0.0
    total_length = 0
    successes = 0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)  # Deterministic seeds for reproducibility
        done = False

        # Episode tracking
        states = [obs['q'].copy()]  # Joint positions
        actions = []
        rewards = []
        ee_positions = []  # End-effector (x, y) positions

        # Initial end-effector pose
        ee_pose = forward_kinematics(obs['q'], robot_params)
        ee_positions.append([ee_pose.x, ee_pose.y])

        episode_reward = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record step
            states.append(obs['q'].copy())
            actions.append(action.copy())
            rewards.append(reward)

            ee_pose = forward_kinematics(obs['q'], robot_params)
            ee_positions.append([ee_pose.x, ee_pose.y])

            episode_reward += reward
            episode_length += 1

            # Render if needed (for video generation)
            if render_mode == 'rgb_array':
                env.render()
                # Could save frames for GIF/video creation here

        # Episode complete
        episode_info = {
            'episode': ep,
            'reward': episode_reward,
            'length': episode_length,
            'success': info.get('is_success', False),
            'states': np.array(states),           # shape (T+1, 2)
            'actions': np.array(actions) if actions else np.empty((0, 2)),
            'rewards': np.array(rewards),
            'ee_positions': np.array(ee_positions)  # shape (T+1, 2)
        }

        episode_data['episodes'].append(episode_info)
        total_reward += episode_reward
        total_length += episode_length
        if info.get('is_success', False):
            successes += 1

        # Create per-episode plots
        if output_dir is not None:
            _plot_episode_trajectory(episode_info, robot_params, output_dir, ep)

    # Summary statistics
    episode_data['success_rate'] = successes / num_episodes if num_episodes > 0 else 0.0
    episode_data['mean_reward'] = total_reward / num_episodes if num_episodes > 0 else 0.0
    episode_data['mean_episode_length'] = total_length / num_episodes if num_episodes > 0 else 0.0

    # Create summary plots
    if output_dir is not None:
        _plot_training_summary(episode_data, output_dir)

    return episode_data


def record_gif(model, env: RobotEnv, num_episodes: int = 3,
               output_dir: str | None = None, fps: int = 15) -> dict[str, Any]:
    """Record a learned policy as a GIF (and PNG frames) by rolling out episodes.

    Parameters
    ----------
    model : Trained Stable-Baselines3 model
    env : RobotEnv instance (must have been constructed with render_mode='rgb_array')
    num_episodes : Number of episodes to record
    output_dir : Directory to save outputs. If None, uses a timestamped folder under ./outputs/
    fps : Frames per second for the output GIF

    Returns
    -------
    dict with keys:
        - episodes: list of per-episode dicts (same as visualize_learned_policy)
        - frame_paths: list[Path] to saved PNG frames (one per timestep)
        - gif_path : Path | None to encoded GIF if imageio is available, else None
    """
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("outputs") / timestamp
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the visualization logic to collect episode data AND render frames.
    episode_data = visualize_learned_policy(
        model, env, num_episodes=num_episodes,
        output_dir=output_dir / "episodes",
        render_mode="rgb_array"
    )

    # Collect frames by re-running the policy and rendering each step.
    frames: list[np.ndarray] = []
    frame_paths: list[Path] = []

    for ep_idx, ep_info in enumerate(episode_data['episodes']):
        # Reset env to the exact same initial condition used during visualization.
        obs, _ = env.reset(seed=42 + ep_idx)
        # Render initial frame.
        frame = env.render()
        assert frame is not None, "Environment must support rgb_array rendering"
        frames.append(frame)
        frame_paths.append(output_dir / f"frame_{len(frame_paths):06d}.png")
        # Step through the recorded actions.
        for action in ep_info['actions']:
            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()
            assert frame is not None
            frames.append(frame)
            frame_paths.append(output_dir / f"frame_{len(frame_paths):06d}.png")
            if terminated or truncated:
                break

    # Save all frames as PNG (lossless, universal).
    for i, (frame, path) in enumerate(zip(frames, frame_paths)):
        # matplotlib can save uint8 RGB arrays via imwrite.
        plt.imsave(path, frame)

    # Try to encode a GIF if imageio is available.
    gif_path: Path | None = None
    try:
        import imageio
        gif_path = output_dir / "policy.gif"
        with imageio.get_writer(gif_path, mode='I', fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
    except ImportError:
        # imageio not installed; skip GIF encoding. User can install via:
        #   pip install mech_rl[video]
        pass

    return {
        'episodes': episode_data['episodes'],
        'frame_paths': frame_paths,
        'gif_path': gif_path,
        'success_rate': episode_data['success_rate'],
        'mean_reward': episode_data['mean_reward'],
        'mean_episode_length': episode_data['mean_episode_length'],
    }


def analyze_sweeps(sweep_param: str = "learning_rate",
                   output_root: str | Path = "outputs",
                   output_dir: str | Path | None = None) -> dict[str, Any]:
    """Analyze Hydra multirun sweeps by collecting eval results from output directories.

    Scans the output_root directory recursively for eval_results.json files
    (generated by train.py after each run), extracts the sweep parameter value
    and the eval_mean_reward, and creates a comparison plot.

    Parameters
    ----------
    sweep_param : Name of the parameter that was swept (e.g., "learning_rate", "distance_coef")
                  This should match a key in the eval_results.json files.
    output_root : Root directory containing Hydra output subdirectories (default: "outputs")
    output_dir  : Directory to save analysis outputs. If None, uses a timestamped folder
                  under output_root/analysis/

    Returns
    -------
    dict with keys:
        - sweep_values: list of swept parameter values
        - rewards: list of corresponding eval mean rewards
        - csv_path: Path to saved CSV with columns [sweep_param, eval_mean_reward]
        - plot_path: Path to saved comparison plot (sweep_param vs eval_mean_reward)
    """
    output_root = Path(output_root)
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root directory not found: {output_root}")

    # Find all eval_results.json files recursively under output_root
    eval_files = list(output_root.rglob("eval_results.json"))
    if not eval_files:
        raise FileNotFoundError(f"No eval_results.json files found in {output_root}")

    # Collect data from each run
    sweep_vals = []
    rewards = []
    run_paths = []

    for eval_file in eval_files:
        # The run directory is the parent of eval_results.json
        run_dir = eval_file.parent
        try:
            import json
            with open(eval_file, 'r') as f:
                results = json.load(f)

            if sweep_param in results and "eval_mean_reward" in results:
                sweep_vals.append(results[sweep_param])
                rewards.append(results["eval_mean_reward"])
                run_paths.append(run_dir)
            else:
                # Skip runs missing the data we need
                continue
        except (json.JSONDecodeError, KeyError, ValueError):
            # Skip malformed files
            continue

    if not sweep_vals:
        raise ValueError(f"No valid runs found with parameter '{sweep_param}'")

    # Sort by sweep parameter for cleaner plotting
    sorted_indices = sorted(range(len(sweep_vals)), key=lambda i: sweep_vals[i])
    sweep_vals = [sweep_vals[i] for i in sorted_indices]
    rewards = [rewards[i] for i in sorted_indices]
    run_paths = [run_paths[i] for i in sorted_indices]

    # Prepare output directory
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = output_root / "analysis" / timestamp
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    import csv
    csv_path = output_dir / f"sweep_analysis_{sweep_param}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([sweep_param, "eval_mean_reward"])
        for val, reward in zip(sweep_vals, rewards):
            writer.writerow([val, reward])

    # Create comparison plot
    plt.figure(figsize=(8, 6))
    plt.plot(sweep_vals, rewards, 'o-', linewidth=2, markersize=8)
    plt.xlabel(sweep_param.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Mean Evaluation Reward', fontsize=12)
    plt.title(f'Sweep Analysis: {sweep_param.replace("_", " ").title()} vs Performance', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Annotate points with run directory names (just the last part for brevity)
    for i, (x, y, run_dir) in enumerate(zip(sweep_vals, rewards, run_paths)):
        plt.annotate(run_dir.name, (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, alpha=0.7)

    plt.tight_layout()
    plot_path = output_dir / f"sweep_plot_{sweep_param}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'sweep_values': sweep_vals,
        'rewards': rewards,
        'csv_path': csv_path,
        'plot_path': plot_path,
        'num_runs': len(sweep_vals),
    }


def _plot_episode_trajectory(episode_info: dict[str, Any],
                           robot_params: RobotParams,
                           output_dir: Path,
                           episode_num: int) -> None:
    """Create trajectory plots for a single episode."""
    states = episode_info['states']  # (T+1, 2) joint angles
    ee_positions = episode_info['ee_positions']  # (T+1, 2) end-effector (x,y)
    rewards = episode_info['rewards']  # (T,) rewards

    # Compute forward kinematics for each joint position to get link positions
    link1_positions = []
    link2_positions = []

    for q in states:
        # Link 1: from base to first joint
        link1_end = np.array([
            robot_params.l1 * np.cos(q[0]),
            robot_params.l1 * np.sin(q[0])
        ])
        # Link 2: from first joint to end-effector
        link2_end = link1_end + np.array([
            robot_params.l2 * np.cos(q[0] + q[1]),
            robot_params.l2 * np.sin(q[0] + q[1])
        ])
        link1_positions.append(link1_end)
        link2_positions.append(link2_end)

    link1_positions = np.array(link1_positions)
    link2_positions = np.array(link2_positions)

    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Arm configuration over time (show start, middle, end)
    ax = axes[0, 0]
    n_steps = len(states)
    indices = [0, n_steps//2, n_steps-1] if n_steps >= 3 else list(range(n_steps))
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    for idx, color in zip(indices, colors, strict=False):
        q = states[idx]
        # Base at (0,0)
        link1_end = np.array([
            robot_params.l1 * np.cos(q[0]),
            robot_params.l1 * np.sin(q[0])
        ])
        ee_pos = link1_end + np.array([
            robot_params.l2 * np.cos(q[0] + q[1]),
            robot_params.l2 * np.sin(q[0] + q[1])
        ])

        # Draw arm
        ax.plot([0, link1_end[0], ee_pos[0]],
                [0, link1_end[1], ee_pos[1]],
                'o-', color=color, linewidth=3, markersize=8,
                label=f'Step {idx}')

        # Draw target if available
        if 'target' in episode_info:
            target = episode_info['target']
            ax.plot(target[0], target[1], 'rx', markersize=12, markeredgewidth=3, label='Target')

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Episode {episode_num} - Arm Configurations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 2: End-effector trajectory
    ax = axes[0, 1]
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2, label='EE Trajectory')
    ax.plot(ee_positions[0, 0], ee_positions[0, 1], 'go', markersize=10, label='Start')
    ax.plot(ee_positions[-1, 0], ee_positions[-1, 1], 'ro', markersize=10, label='End')

    if 'target' in episode_info:
        target = episode_info['target']
        ax.plot(target[0], target[1], 'kx', markersize=15, markeredgewidth=4, label='Target')
        # Draw target tolerance circle
        tolerance = getattr(episode_info.get('success_radius', 0.05), 'success_radius', 0.05)
        circle = plt.Circle((target[0], target[1]), tolerance, fill=False,
                          linestyle='--', color='gray', alpha=0.7)
        ax.add_patch(circle)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Episode {episode_num} - End-Effector Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 3: Joint angles over time
    ax = axes[1, 0]
    time_steps = np.arange(len(states))
    ax.plot(time_steps, states[:, 0], 'r-', linewidth=2, label='Shoulder (q0)')
    ax.plot(time_steps, states[:, 1], 'b-', linewidth=2, label='Elbow (q1)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Joint Angle (rad)')
    ax.set_title(f'Episode {episode_num} - Joint Angles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Rewards over time
    ax = axes[1, 1]
    if len(rewards) > 0:
        ax.plot(np.arange(len(rewards)), rewards, 'g-', linewidth=2, label='Step Reward')
        ax.plot(np.arange(len(rewards)), np.cumsum(rewards), 'orange', linewidth=2,
                label='Cumulative Reward')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Reward')
    ax.set_title(f'Episode {episode_num} - Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"episode_{episode_num:02d}_trajectory.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_training_summary(episode_data: dict[str, Any], output_dir: Path) -> None:
    """Create summary plots across all episodes."""
    episodes = episode_data['episodes']
    if not episodes:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Episode rewards
    ax = axes[0, 0]
    episode_nums = [ep['episode'] for ep in episodes]
    rewards = [ep['reward'] for ep in episodes]
    ax.bar(episode_nums, rewards, alpha=0.7, color='skyblue', edgecolor='navy')
    ax.axhline(y=episode_data['mean_reward'], color='red', linestyle='--',
               label=f'Mean: {episode_data["mean_reward"]:.2f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Episode lengths
    ax = axes[0, 1]
    lengths = [ep['length'] for ep in episodes]
    ax.bar(episode_nums, lengths, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    ax.axhline(y=episode_data['mean_episode_length'], color='red', linestyle='--',
               label=f'Mean: {episode_data["mean_episode_length"]:.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Success rate over episodes (cumulative)
    ax = axes[1, 0]
    successes = np.array([1 if ep['success'] else 0 for ep in episodes])
    cumulative_success_rate = np.cumsum(successes) / (np.arange(len(successes)) + 1)
    ax.plot(episode_nums, cumulative_success_rate, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Success Rate')
    ax.set_title('Learning Progress')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Plot 4: End-effector error over time (for first episode)
    ax = axes[1, 1]
    if episodes and 'ee_positions' in episodes[0] and 'target' in episodes[0]:
        ee_pos = episodes[0]['ee_positions']
        target = np.array(episodes[0]['target'])
        errors = np.linalg.norm(ee_pos - target, axis=1)
        ax.plot(np.arange(len(errors)), errors, 'purple', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('End-Effector Error (m)')
        ax.set_title('EE Error vs Target (Episode 0)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "training_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_training_report(logdir: str, output_dir: str | None = None) -> Path:
    """Create a comprehensive HTML report from TensorBoard logs and policy visualization.

    Args:
        logdir: Path to TensorBoard log directory
        output_dir: Directory to save report. If None, uses logdir/report

    Returns:
        Path to the generated HTML report
    """
    if output_dir is None:
        output_dir = Path(logdir) / "report"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    csv_paths = tensorboard_scalars_to_csv(logdir, output_dir / "csv")
    plot_paths = plot_training_curves(logdir, output_dir / "plots")

    # Create simple HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mech_RL Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2c3e50; }}
            .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            .plot img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>Mech_RL Training Report</h1>
        <p>Generated from TensorBoard logs in: <code>{logdir}</code></p>

        <div class="section">
            <h2>Training Curves</h2>
            <p>The following plots show key metrics tracked during training:</p>
    """

    # Add plots
    for tag, plot_path in plot_paths.items():
        relative_path = plot_path.relative_to(output_dir)
        safe_tag = tag.replace('/', ' :: ')
        html_content += f"""
            <div class="plot">
                <h3>{safe_tag}</h3>
                <img src="{relative_path}" alt="{safe_tag} plot">
            </div>
        """

    html_content += """
        </div>

        <div class="section">
            <h2>Extracted Data</h2>
            <p>Scalar data has been extracted to CSV files for further analysis:</p>
            <table>
                <tr><th>Tag</th><th>CSV File</th></tr>
    """

    for tag, csv_path in csv_paths.items():
        relative_path = csv_path.relative_to(output_dir)
        html_content += f"""
                <tr><td>{tag}</td><td><code>{relative_path}</code></td></tr>
        """

    html_content += """
            </table>
        </div>

        <div class="section">
            <h2>Usage Notes</h2>
            <p>
                To view TensorBoard interactively during or after training:<br>
                <code>tensorboard --logdir {logdir}</code><br><br>
                The extracted CSV files can be used with pandas, Excel, or other analysis tools.<br>
                Individual episode visualizations are available in the plots/ directory.
            </p>
        </div>
    </body>
    </html>
    """

    report_path = output_dir / "report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return report_path


__all__ = [
    "tensorboard_scalars_to_csv",
    "plot_training_curves",
    "visualize_learned_policy",
    "create_training_report"
]
