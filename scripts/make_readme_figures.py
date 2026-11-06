"""Generate the figures and headline numbers embedded in README.md.

Run from the repository root:

    python scripts/make_readme_figures.py

Writes to docs/assets/:
    arm_schematic.png  arm geometry, joint angles, target and success radius
    workspace.png      reachable workspace, sampled targets, two elbow solutions
    sweep_summary.png  hyperparameter sweep results (numbers copied from Notes/)
    policy.gif         the trained policy reaching three random targets
    results.json       success rate / reward of the policy vs a random policy

The policy comes from the newest MLflow run in ./mlruns that used the final
hyperparameters (see FINAL_PARAMS). Nothing is retrained.

Why the GIF is not made with visualization.record_gif: that function writes one
PNG per frame to disk, which is thousands of files for 1000-step episodes. This
script reuses the same env.render() frames and encodes the GIF directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Circle  # noqa: E402
from scipy import stats  # noqa: E402

from mech_rl.domain.parameters import RewardParams, RobotParams, SimParams  # noqa: E402
from mech_rl.environment import RobotEnv  # noqa: E402
from mech_rl.physics import forward_kinematics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"

# Colours match the env renderer (robot_env._ArmRenderer).
LINK, EE, TARGET, MUTED = "#2c3e50", "#e74c3c", "#27ae60", "#7f8c8d"

# Final configuration (Notes/Day_11.md, Notes/Day_13.md).
ROBOT = RobotParams(l1=0.3, l2=0.3, m1=1.0, m2=1.0, i1=0.03, i2=0.03, friction=0.05, max_torque=5.0)
SIM = SimParams(dt=0.01, max_episode_steps=1000, integrator="semi_implicit_euler")
REWARD = RewardParams(distance_coef=0.5, effort_coef=0.01, time_penalty=0.01)
FINAL_PARAMS = {
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "gae_lambda": 0.9,
    "learning_rate": 0.0001,
    "total_timesteps": 1_000_000,
}
EVAL_SEEDS = range(20)  # same seed-per-episode convention as eval_loop.evaluate
CHOSEN = {"1e-4", "0.1", "0.01", "0.9"}  # sweep values used in FINAL_PARAMS

# name, setting, (label, mean, std, n) x2 -- copied from Notes/Day_11.md and Notes/Day_12.md.
# The notes report population std (ddof=0).
SWEEPS = [
    ("Learning rate", "100k steps, 10 seeds each", ("1e-4", -164.67, 50.16, 10), ("3e-4", -189.83, 34.77, 10)),
    ("Clip range", "8 runs each", ("0.1", -338.96, 28.71, 8), ("0.2", -345.83, 26.18, 8)),
    ("Entropy coefficient", "8 runs each", ("0.0", -345.90, 26.15, 8), ("0.01", -338.88, 28.72, 8)),
    ("GAE lambda", "8 runs each", ("0.9", -332.07, 29.99, 8), ("0.95", -352.72, 19.84, 8)),
]


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _arm_points(q: np.ndarray) -> np.ndarray:
    """Base, elbow and end-effector positions; the end effector comes from forward_kinematics."""
    ee = forward_kinematics(q, ROBOT)
    elbow = ROBOT.l1 * np.array([np.cos(q[0]), np.sin(q[0])])
    return np.array([[0.0, 0.0], elbow, [ee.x, ee.y]])


def make_arm_schematic() -> None:
    q = np.array([0.7, 1.1])
    base, elbow, tip = _arm_points(q)
    target = tip + np.array([0.07, 0.06])

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    ax.plot(*zip(base, elbow, tip), "-", color=LINK, lw=6, solid_capstyle="round", zorder=3)
    ax.plot(*zip(base, elbow), "o", color="white", mec=LINK, mew=2.5, ms=11, zorder=4)
    ax.plot(*tip, "o", color=EE, ms=9, zorder=5)
    ax.plot(*target, "x", color=TARGET, ms=12, mew=3, zorder=5)
    ax.add_patch(Circle(target, 0.05, fill=False, ls="--", color=TARGET, zorder=2))
    ax.plot(*zip(tip, target), ":", color=MUTED, lw=1.5, zorder=1)

    ax.plot([0, 0.30], [0, 0], "-", color=MUTED, lw=1, zorder=1)  # q1 reference axis
    link1_dir = np.array([np.cos(q[0]), np.sin(q[0])])
    ax.plot(*zip(elbow, elbow + 0.17 * link1_dir), "--", color=MUTED, lw=1, zorder=1)
    ax.add_patch(Arc((0, 0), 0.2, 0.2, theta1=0, theta2=np.degrees(q[0]), color=EE, lw=2))
    ax.add_patch(Arc(elbow, 0.2, 0.2, theta1=np.degrees(q[0]),
                     theta2=np.degrees(q[0] + q[1]), color=EE, lw=2))

    def at_angle(origin: np.ndarray, angle: float, radius: float) -> np.ndarray:
        return origin + radius * np.array([np.cos(angle), np.sin(angle)])

    def beside(a: np.ndarray, b: np.ndarray, offset: float) -> np.ndarray:
        """Midpoint of a-b pushed `offset` metres to the left of the a->b direction."""
        d = (b - a) / np.linalg.norm(b - a)
        return (a + b) / 2 + offset * np.array([-d[1], d[0]])

    for text, (x, y), color in (
        ("$q_1$", at_angle(base, q[0] / 2, 0.145), EE),
        ("$q_2$", at_angle(elbow, q[0] + q[1] / 2, 0.145), EE),
        ("$l_1$", beside(base, elbow, 0.045), LINK),
        ("$l_2$", beside(elbow, tip, 0.045), LINK),
    ):
        ax.text(x, y, text, color=color, fontsize=13, ha="center", va="center")
    ax.annotate("target", target, xytext=(target[0] + 0.03, target[1] + 0.085), color=TARGET,
                fontsize=11, ha="center")
    ax.annotate("success radius\n0.05 m", target + 0.05 * np.array([0.7, -0.7]),
                xytext=(target[0] + 0.15, target[1] - 0.02), color=TARGET, fontsize=9, va="center",
                arrowprops={"arrowstyle": "-", "color": TARGET, "lw": 0.8})
    ax.text(*(base + np.array([-0.02, -0.055])), "shoulder", fontsize=9, color=MUTED, ha="center")
    ax.text(*(tip + np.array([-0.03, 0.04])), "end effector", fontsize=9, color=EE, ha="right")

    ax.set_xlim(-0.08, 0.62)
    ax.set_ylim(-0.1, 0.62)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT / "arm_schematic.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _two_elbow_solutions(point: np.ndarray) -> list[np.ndarray]:
    """Closed-form IK (law of cosines) for the elbow-down and elbow-up configurations."""
    l1, l2 = ROBOT.l1, ROBOT.l2
    cos_q2 = (point @ point - l1**2 - l2**2) / (2 * l1 * l2)
    solutions = []
    for sign in (1.0, -1.0):
        q2 = sign * np.arccos(np.clip(cos_q2, -1.0, 1.0))
        q1 = np.arctan2(point[1], point[0]) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        q = np.array([q1, q2])
        ee = forward_kinematics(q, ROBOT)
        assert np.allclose([ee.x, ee.y], point, atol=1e-9), "IK solution misses the target"
        solutions.append(q)
    return solutions


def make_workspace() -> None:
    reach = ROBOT.l1 + ROBOT.l2
    env = RobotEnv(ROBOT, SIM, REWARD)
    targets = np.array([env.reset(seed=s)[0]["target"] for s in range(400)])
    example = np.array([0.38, 0.30])

    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
    ax.add_patch(Circle((0, 0), reach, color="#eaf2f8", zorder=0))
    ax.add_patch(Circle((0, 0), reach, fill=False, color=MUTED, lw=1.2, ls="--", zorder=1))
    ax.scatter(*targets.T, s=7, color=TARGET, alpha=0.45, zorder=2, label="sampled targets")
    for q, color, name in zip(_two_elbow_solutions(example), (LINK, EE), ("elbow down", "elbow up")):
        ax.plot(*zip(*_arm_points(q)), "-o", color=color, lw=3, ms=6, alpha=0.85, zorder=3, label=name)
    ax.plot(*example, "x", color="black", ms=11, mew=2.5, zorder=4)
    ax.plot(0, 0, "s", color="black", ms=7, zorder=4)
    ax.text(-0.70, 0.62, "max reach\n$l_1 + l_2$ = 0.6 m", fontsize=9, color=MUTED,
            ha="left", va="bottom")

    ax.set_xlim(-0.72, 0.72)
    ax.set_ylim(-0.72, 0.72)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    _style(ax)
    fig.savefig(OUT / "workspace.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def welch_p(a: tuple, b: tuple) -> float:
    """Welch t-test p-value from (label, mean, std, n); converts population std to sample std."""
    def sample_std(std: float, n: int) -> float:
        return std * np.sqrt(n / (n - 1))
    return float(stats.ttest_ind_from_stats(a[1], sample_std(a[2], a[3]), a[3],
                                            b[1], sample_std(b[2], b[3]), b[3],
                                            equal_var=False).pvalue)


def make_sweep_summary() -> dict[str, float]:
    p_values: dict[str, float] = {}
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), dpi=150)
    for ax, (name, setting, a, b) in zip(axes.flat, SWEEPS):
        p = welch_p(a, b)
        p_values[name] = p
        for x, (label, mean, std, _n) in enumerate((a, b)):
            ax.errorbar(x, mean, yerr=std, fmt="o", ms=9, capsize=5, lw=2,
                        color=TARGET if label in CHOSEN else MUTED)
            ax.text(x, mean + std + 4, f"{mean:.0f}", ha="center", fontsize=9)
        ax.set_xticks([0, 1], [a[0], b[0]])
        ax.set_xlim(-0.6, 1.6)
        pad = 25 + max(a[2], b[2])
        ax.set_ylim(min(a[1], b[1]) - pad - 10, max(a[1], b[1]) + pad + 12)
        ax.set_title(f"{name}\n{setting}", fontsize=10)
        ax.text(0.5, 0.04, f"Welch p = {p:.2f}  (not significant)" if p >= 0.05
                else f"Welch p = {p:.3f}", transform=ax.transAxes, ha="center", fontsize=8.5,
                color=MUTED)
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean eval reward (higher is better)")
    fig.suptitle("Every pair differs by less than the run-to-run spread (mean ± std)\n"
                 "green = value used in the final configuration", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_summary.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p_values


def find_final_model() -> tuple[Path, str]:
    """Newest MLflow run whose logged params match FINAL_PARAMS and that saved a model."""
    candidates = []
    for run in (ROOT / "mlruns").glob("*/*"):
        params, model = run / "params", run / "artifacts" / "model" / "model.zip"
        if not model.is_file() or not params.is_dir():
            continue
        try:
            logged = {k: float((params / k).read_text()) for k in FINAL_PARAMS}
            reward_coef = float((params / "reward" / "distance_coef").read_text())
        except (OSError, ValueError):
            continue
        if logged == FINAL_PARAMS and reward_coef == REWARD.distance_coef:
            candidates.append((run.stat().st_mtime, model, run.name))
    if not candidates:
        raise SystemExit("No MLflow run with the final hyperparameters and a saved model found.")
    _, model, run_id = max(candidates)
    return model, run_id


def rollout(env: RobotEnv, policy, seed: int, render_stride: int = 0) -> tuple[float, bool, int, list]:
    """One episode. Returns (return, reached target, steps, frames every `render_stride` steps)."""
    obs, _ = env.reset(seed=seed)
    frames = [env.render()] if render_stride else []
    total, steps, terminated, truncated = 0.0, 0, False, False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total, steps = total + reward, steps + 1
        if render_stride and (steps % render_stride == 0 or terminated or truncated):
            frames.append(env.render())
    return total, terminated, steps, frames


def summarize(env: RobotEnv, policy) -> dict[str, float]:
    runs = [rollout(env, policy, seed) for seed in EVAL_SEEDS]
    return {
        "success_rate": float(np.mean([r[1] for r in runs])),
        "mean_reward": float(np.mean([r[0] for r in runs])),
        "mean_episode_length": float(np.mean([r[2] for r in runs])),
    }


def make_policy_gif_and_results(p_values: dict[str, float]) -> None:
    from stable_baselines3 import PPO

    model_path, run_id = find_final_model()
    model = PPO.load(str(model_path), device="cpu")
    env = RobotEnv(ROBOT, SIM, REWARD, render_mode="rgb_array")
    trained = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
    random_rng = np.random.default_rng(0)
    random_policy = lambda obs: random_rng.uniform(-ROBOT.max_torque, ROBOT.max_torque, 2)  # noqa: E731

    results = {
        "model_run": run_id,
        "episodes": len(EVAL_SEEDS),
        "seeds": [EVAL_SEEDS.start, EVAL_SEEDS.stop - 1],
        "trained_policy": summarize(env, trained),
        "random_policy": summarize(env, random_policy),
        "sweep_welch_p": {k: round(v, 3) for k, v in p_values.items()},
    }
    # The GIF shows the first three *successful* evaluation episodes (a disclosed choice:
    # the README states the true success rate next to it).
    frames: list[np.ndarray] = []
    gif_seeds: list[int] = []
    for seed in EVAL_SEEDS:
        _, reached, _, ep_frames = rollout(env, trained, seed, render_stride=4)
        if reached:
            gif_seeds.append(seed)
            frames += ep_frames + [ep_frames[-1]] * 8  # hold the last frame briefly
        if len(gif_seeds) == 3:
            break
    imageio.mimsave(OUT / "policy.gif", frames, duration=1000 / 25, loop=0)

    results["gif_seeds"] = gif_seeds
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_arm_schematic()
    make_workspace()
    p_values = make_sweep_summary()
    make_policy_gif_and_results(p_values)
    print("Wrote:", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
