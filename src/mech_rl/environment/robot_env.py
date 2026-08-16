"""Gymnasium env for the 2-DOF arm. Torque control, Dict obs (q, qdot, target)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from mech_rl.domain.parameters import RewardParams, RobotParams, SimParams
from mech_rl.domain.state import RobotState
from mech_rl.domain.types import as_array
from mech_rl.physics import forward_kinematics, rk4, semi_implicit_euler

# Integrator dispatch — resolved once at init, no if/else in hot step().
_INTEGRATORS = {
    "semi_implicit_euler": semi_implicit_euler,
    "rk4": rk4,
}

# Observation dict keys.
_Q_KEY = "q"
_QDOT_KEY = "qdot"
_TARGET_KEY = "target"


class RobotEnv(gym.Env):
    """Torque-controlled 2-DOF arm env. Dict obs (q, qdot, target). Use FlattenObservation for SB3.

    Args:
        robot_params: Arm physical params (lengths, masses, inertias, friction, max_torque).
        sim_params: dt, max_episode_steps, integrator name.
        reward_params: Coefficients for reward terms.
        target: Fixed (x,y) target. None = random target each reset.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        robot_params: RobotParams,
        sim_params: SimParams,
        reward_params: RewardParams,
        *,
        target: np.ndarray | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unknown render_mode {render_mode!r}; "
                f"expected one of {self.metadata['render_modes']}"
            )

        self.robot_params = robot_params
        self.sim_params = sim_params
        self.reward_params = reward_params

        # Integrator function for the hot path.
        self._integrator = _INTEGRATORS[sim_params.integrator]

        # Observation space: Dict with explicit keys.
        max_reach = robot_params.l1 + robot_params.l2
        self.observation_space = gym.spaces.Dict(
            {
                _Q_KEY: gym.spaces.Box(
                    low=-np.pi, high=np.pi, shape=(2,), dtype=np.float64
                ),
                _QDOT_KEY: gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                ),
                _TARGET_KEY: gym.spaces.Box(
                    low=-max_reach, high=max_reach, shape=(2,), dtype=np.float64
                ),
            }
        )

        # Action space: joint torques bounded by max_torque.
        self.action_space = gym.spaces.Box(
            low=-robot_params.max_torque,
            high=robot_params.max_torque,
            shape=(2,),
            dtype=np.float64,
        )

        # Internal state (set properly on reset).
        self._state = RobotState(q=np.zeros(2), qdot=np.zeros(2))
        self._target: np.ndarray = np.zeros(2)
        self._step_count: int = 0
        self._prev_action: np.ndarray | None = None

        # Fixed target for reproducible testing / eval.
        self._fixed_target: np.ndarray | None = (
            as_array(target) if target is not None else None
        )

        # Random number generator (gymnasium-managed via seed()).
        self._rng = np.random.default_rng()

        # Rendering state. render_mode set only when explicitly requested so the
        # default training path pays no matplotlib import / figure cost.
        self.render_mode: str | None = render_mode
        self._renderer: _ArmRenderer | None = None

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options or {})
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}

        # Joint positions: custom or random.
        if "reset_position" in options:
            q = as_array(options["reset_position"])
        else:
            q = self._rng.uniform(-np.pi, np.pi, size=2)

        self._state = RobotState(q=q, qdot=np.zeros(2))
        self._step_count = 0
        self._prev_action = None

        # Target: options override fixed target; if neither is set, resample.
        if "target" in options:
            self._target = as_array(options["target"])
        elif self._fixed_target is not None:
            self._target = self._fixed_target.copy()
        else:
            self._target = self._sample_target()

        return self._obs_dict(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = as_array(action)
        action = np.clip(action, -self.robot_params.max_torque, self.robot_params.max_torque)

        # Advance physics.
        new_state = self._integrator(self._state, action, self.robot_params, self.sim_params.dt)

        # Count this transition before termination checks so that
        # the Nth step is truncated when max_episode_steps == N.
        self._step_count += 1

        # Reward.
        reward = self._compute_reward(self._state, action, new_state)

        # Termination checks.
        terminated, truncated = self._check_done(new_state)

        # Bookkeeping.
        self._prev_action = action.copy()
        self._state = new_state

        info: dict[str, Any] = {}
        ee = forward_kinematics(new_state.q, self.robot_params)
        info["ee_pose"] = (ee.x, ee.y, ee.theta)
        info["distance"] = float(np.linalg.norm(np.array([ee.x, ee.y]) - self._target))

        return self._obs_dict(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering (rgb_array only — matplotlib Agg backend, lazy-imported)
    # ------------------------------------------------------------------

    def render(self):
        """Render the current arm state as an rgb_array frame.

        Returns an (H, W, 3) uint8 numpy array, or None if render_mode is not
        set. matplotlib is imported lazily so the training hot path pays no
        figure/Agg cost when rendering is not requested.
        """
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = _ArmRenderer(self.robot_params)
        return self._renderer.draw(self._state.q, self._target, self._step_count)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_dict(self) -> dict[str, Any]:
        return {
            _Q_KEY: self._state.q.copy(),
            _QDOT_KEY: self._state.qdot.copy(),
            _TARGET_KEY: self._target.copy(),
        }

    def _compute_reward(
        self, state: RobotState, action: np.ndarray, new_state: RobotState
    ) -> float:
        rp = self.reward_params

        ee = forward_kinematics(new_state.q, self.robot_params)
        ee_xy = np.array([ee.x, ee.y])
        distance = float(np.linalg.norm(ee_xy - self._target))

        # Distance penalty
        reward = -rp.distance_coef * distance

        # Effort: ||tau||^2
        reward -= rp.effort_coef * float(np.dot(action, action))

        # Smoothness: ||tau_t - tau_{t-1}||^2 (skipped on first step)
        if self._prev_action is not None and rp.smoothness_coef > 0.0:
            tau_diff = action - self._prev_action
            reward -= rp.smoothness_coef * float(np.dot(tau_diff, tau_diff))

        # Success bonus
        if distance < rp.success_radius:
            reward += rp.success_bonus

        # Time penalty per step
        reward -= rp.time_penalty

        return float(reward)

    def _check_done(self, state: RobotState) -> tuple[bool, bool]:
        terminated = False
        truncated = False

        # Success: end-effector within success_radius of target.
        ee = forward_kinematics(state.q, self.robot_params)
        ee_xy = np.array([ee.x, ee.y])
        if np.linalg.norm(ee_xy - self._target) < self.reward_params.success_radius:
            terminated = True

        # Time limit.
        if self._step_count >= self.sim_params.max_episode_steps:
            truncated = True

        return terminated, truncated

    def _sample_target(self) -> np.ndarray:
        """Sample a random target inside the reachable workspace."""
        max_reach = self.robot_params.l1 + self.robot_params.l2
        while True:
            xy = self._rng.uniform(-max_reach, max_reach, size=2)
            if np.linalg.norm(xy) <= max_reach:
                return xy


class _ArmRenderer:
    """Lazy matplotlib Agg renderer for the 2-DOF arm.

    Holds one figure/axes pair across draws to avoid per-step re-creation.
    A small fixed square view bounds the workspace regardless of arm pose.
    """

    def __init__(self, robot_params: RobotParams, figsize: float = 3.0) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        self._plt = plt
        self._Circle = Circle
        self._params = robot_params

        reach = robot_params.l1 + robot_params.l2
        # Pad the view so the arm never sits flush against the edge.
        self._lim = float(reach * 1.15)

        self._fig = plt.figure(figsize=(figsize, figsize), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_aspect("equal")
        self._ax.set_xlim(-self._lim, self._lim)
        self._ax.set_ylim(-self._lim, self._lim)
        self._ax.set_facecolor("#f7f7f7")

        # Reusable artists so each draw mutates in place instead of redrawing
        # the whole canvas from scratch.
        self._link_line, = self._ax.plot([], [], "-o", color="#2c3e50",
                                         linewidth=4, markersize=6, zorder=3)
        self._ee_dot, = self._ax.plot([], [], "o", color="#e74c3c",
                                      markersize=8, zorder=4)
        self._target_marker, = self._ax.plot([], [], "x", color="#27ae60",
                                             markersize=12, markeredgewidth=3,
                                             zorder=4)
        # Tolerance circle — fixed visible radius (not tied to reward params here)
        self._target_circle = Circle((0.0, 0.0), 0.05, fill=False,
                                     linestyle="--", color="#27ae60", zorder=2)
        self._ax.add_patch(self._target_circle)
        self._step_text = self._ax.text(
            0.02, 0.98, "", transform=self._ax.transAxes, va="top", fontsize=9
        )
        self._fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)

    def draw(self, q: np.ndarray, target: np.ndarray, step: int) -> np.ndarray:
        """Update artist data and return an (H, W, 3) uint8 frame."""
        p0 = np.array([0.0, 0.0])
        p1 = np.array([
            self._params.l1 * np.cos(q[0]), self._params.l1 * np.sin(q[0])
        ])
        p2 = p1 + np.array([
            self._params.l2 * np.cos(q[0] + q[1]),
            self._params.l2 * np.sin(q[0] + q[1]),
        ])

        self._link_line.set_data([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]])
        self._ee_dot.set_data([p2[0]], [p2[1]])
        self._target_marker.set_data([target[0]], [target[1]])
        self._target_circle.set_center((float(target[0]), float(target[1])))
        self._step_text.set_text(f"step {step}")

        self._fig.canvas.draw()
        # buffer_rgba() — (H, W, 4) uint8 on the Agg backend.
        arr = np.asarray(self._fig.canvas.buffer_rgba())
        return arr[:, :, :3].copy()


__all__ = ["RobotEnv"]
