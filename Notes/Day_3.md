# Day 3 — Gymnasium env wrapping the physics layer

## Goal
Wrap the framework-independent physics in a standard `gym.Env` so the arm can be trained with Stable-Baselines3's PPO.

## What I did

**mech_rl/environment/robot_env.py**
- `RobotEnv(gym.Env)` — torque-control environment for the 2-DOF arm.
- **Observation space** — `Dict` with explicit keys:
  - `q`: joint angles `(2,)`, bounded `[-π, π]`
  - `qdot`: joint velocities `(2,)`, unbounded
  - `target`: end-effector target `(x, y)` `(2,)`
- **Action space** — `Box(-max_torque, +max_torque, shape=(2,))`, continuous joint torques.
- **`reset(seed, options)`** — supports:
  - `options["reset_position"]`: fixed initial joint angles
  - `options["target"]`: override target for this episode
  - Random sampling otherwise (joint angles uniform in `[-π, π]`, target sampled inside the reachable workspace circle).
- **`step(action)`** — clips torques, integrates via the dispatched integrator, computes reward, checks termination.
- **Integrator dispatch** — a `dict` mapping `SimParams.integrator` string to the physics-layer function, resolved once at construction.
- **Reward assembly** — linear combination of `RewardParams` coefficients:
  - `-distance_coef * ||ee - target||`
  - `-effort_coef * ||tau||²`
  - `-smoothness_coef * ||tau_t - tau_{t-1}||²`
  - `+success_bonus` when `||ee - target|| < success_radius`
  - `-time_penalty` per step
- **Termination** — `terminated` when end-effector reaches `success_radius` of target; `truncated` at `max_episode_steps`.

**mech_rl/environment/__init__.py**
Re-exports `RobotEnv` for `from mech_rl.environment import RobotEnv`.

**Tests**
- `tests/unit/test_robot_env.py` — 27 tests across 7 groups:
  - Construction (4): space shapes, bounds, keys
  - Reset (7): obs shape, determinism, custom options, zero velocities
  - Step (5): tuple format, obs bounds, truncation, info contents
  - Reward (4): distance penalty, effort penalty, success bonus, time penalty
  - Termination (2): success and non-success cases
  - Integrator dispatch (3): both integrators work, produce different states
  - Physics consistency (2): single-step and multi-step manual integration check

## Why I did it this way
1. **`Dict` observation space over flat array.** Keeps field names explicit in logs and replay buffers. SB3's `FlattenObservation` wrapper handles flattening when feeding into a policy network.
2. **Torque clipping inside `step()`.** The action is clipped to `[-max_torque, max_torque]` before integration, so the physics never sees illegal torques. No separate torque-limit termination — the clip enforces it by construction.
3. **Target is persistent but mutable.** A fixed target passed to the constructor is reused across episodes (for reproducible eval). The `reset()` options dict can override it per-episode. If neither is set, a random target is sampled inside the reachable workspace.
4. **Integrator dispatch as a `dict` at construction time.** Avoids an `if/else` in the hot `step()` path. The lookup happens once in `__init__`.
5. **`_prev_action` initialized to `None`.** The smoothness penalty is skipped on the first step of an episode (no previous action to compare against). This avoids an arbitrary initial penalty.

## What's broken / annoying
- **No `render()` implementation.** Visualization is Day 10+ scope.
- **`_sample_target` uses rejection sampling.** Fine for 2D but if the arm ever gets more DOF, consider analytic workspace sampling.
- **No joint-angle wrapping.** Angles can drift outside `[-π, π]` over very long episodes. Not a problem for the current `max_episode_steps` of 1000 but worth noting.

## How to check it works
```bash
.venv/Scripts/python.exe -m pytest tests/unit -v       # 98 passed
.venv/Scripts/python.exe -m ruff check src/mech_rl tests/  # clean
```

Smoke test:
```bash
.venv/Scripts/python.exe -c "
from mech_rl.environment import RobotEnv
from mech_rl.domain.parameters import RobotParams, SimParams, RewardParams
env = RobotEnv(
    RobotParams(l1=0.3, l2=0.3, m1=1.0, m2=1.0, i1=0.03, i2=0.03, friction=0.05, max_torque=5.0),
    SimParams(dt=0.01, max_episode_steps=1000, integrator='semi_implicit_euler'),
    RewardParams(), target=[0.3, 0.0])
obs, _ = env.reset()
for _ in range(100):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc: break
print('Smoke test passed')
"
```

## Next up
**Day 4 — Hydra config integration.** Load all params from YAML files, wire up the experiment loop skeleton.