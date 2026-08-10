# Day2 — Physics core

## Goal

Framework-independent physics for the 2-DOF arm, fully tested. No torch,
gymnasium, or matplotlib imports inside `physics/`.

## What Shipped

### `mech_rl/physics/kinematics.py`

- `forward_kinematics(q, params) -> EndEffectorPose` — analytic planar FK.
  Returns `(x, y, theta)` of the end-effector in the base frame.
- `jacobian(q, params) -> np.ndarray` — analytic (2, 2) geometric
  Jacobian. Singular at `q = [a, pi]` (arm folded flat).

### `mech_rl/physics/dynamics.py`

- `mass_matrix(q, params)` — symmetric positive-definite inertia matrix.
  `M[0, 0]` and `M[0, 1]` both carry the same `cos(q1)` dependence;
  `M[1, 1] = i2` is constant.
- `coriolis(q, qdot, params)` — Coriolis / centripetal vector
  `C(q, qdot) qdot`, derived from the energy-conserving Lagrangian form
  `C qdot = M_dot qdot - 0.5 grad_q(qdot^T M qdot)`.
- `gravity(q, params)` — gravitational torque (gravity along `-y`).
- `friction(qdot, params)` — viscous friction `friction * qdot`.
- `potential_energy(q, params)` — gravitational PE, referenced to
  `U = 0` when the arm hangs straight down.
- `equation_of_motion(q, qdot, tau, params)` — solves
  `M qddot = tau - C - G - F` via `np.linalg.solve`.
- Module-level `G_ACCEL = 9.81`.

### `mech_rl/physics/integrator.py`

- `semi_implicit_euler(state, tau, params, dt)` — symplectic, first
  order. Updates velocity first, then position with the new velocity.
- `rk4(state, tau, params, dt)` — classical 4th-order Runge-Kutta on the
  (q, qdot) state. Each step evaluates the EOM four times.

### `mech_rl/physics/__init__.py`

Re-exports the public physics API so callers can write
`from mech_rl.physics import forward_kinematics` without reaching into
the submodules.

### Tests

- `tests/unit/test_kinematics.py` — 8 tests
- `tests/unit/test_dynamics.py` — 18 tests
- `tests/unit/test_integrator.py` — 13 tests
- `tests/conftest.py` — added `frictionless_robot_params` fixture

**Result:** 71 passed (32 from Day1 +39 new). Ruff clean.

## Decisions Made

1. **`i1`, `i2` mean moment of inertia about the joint**, parallel-axis
   included. Updated the docstring on `RobotParams` and bumped the test
   fixtures from `0.01` to `0.03` (`m * l^2 / 3` for a uniform rod about
   its end). The previous value made the inertia matrix non-positive-
   definite with the corrected Coriolis formulation, so the new default
   matches the rigid-rod model the physics layer assumes.
2. **Coriolis from the Lagrangian form, not the Christoffel symbols.**
   The Lagrangian form `C qdot = M_dot qdot - 0.5 grad_q(qdot^T M qdot)`
   is the unique quadratic-in-`qdot` choice that guarantees energy
   conservation. Christoffel-symbol formulations require careful index
   bookkeeping that is easy to get wrong on a planar arm.
3. **Gravity as a module-level constant** (`G_ACCEL = 9.81`). Threading
   it through `RobotParams` would add a field that no caller needs yet
   — Day3+ can promote it when an env config wants to vary it.
4. **`np.linalg.solve` over `inv`** for the 2x2 system. Faster and
   avoids allocating an inverse that the caller doesn't need.
5. **PE referenced to `U = 0` at `q = [-pi/2, 0]`** (arm straight down)
   with an explicit additive shift in the formula. The shift does not
   change `G = dU/dq` but makes the energy-conservation tests
   independent of the absolute reference.
6. **Adaptive step-size integrators deferred** to a future day. Day3's
   env layer drives integration at a fixed `dt` from `SimParams`; adding
   adaptive stepping would complicate that boundary.

## Known Issues / Tech Debt

- **No `gravity` field on `RobotParams`.** Hardcoded `G_ACCEL = 9.81` is
  fine for Day2 but limits future scenarios (zero-g, lunar-g). Promote
  when a config demands it.
- **Closed-form expressions assume uniform rods with CoM at the
  midpoint.** If we ever need non-uniform links, swap the symbolic M
  and Coriolis for general formulas — the public API does not change.
- **Inverse-consistency test (`test_inverse_consistency`) is the only
  end-to-end check on the EOM assembly.** A more thorough test would
  compare the symbolic `qddot` against a finite-difference
  acceleration-from-trajectory, but that's a Day3+ concern.
- **The Coriolis derivation assumed the Lagrangian form**. If anyone
  later swaps in a different formulation (Christoffel, Kane, Lie-group),
  re-run the energy-conservation tests with `dt = 1e-5` to confirm the
  new form is also conservative — `pytest tests/unit/test_integrator.py
  -v` will surface any regression.

## Verification

```bash
.venv/Scripts/python.exe -m pytest tests/unit -v   # 71 passed
.venv/Scripts/python.exe -m ruff check src/mech_rl tests/   # clean
```

Spot-checked energy conservation outside the test suite:

```
RK4 (free motion, 1000 steps @ dt=0.001s):   rel_drift = 1.3e-10
SI-Euler (free motion, 1000 steps @ dt=0.001s): rel_drift = 8.4e-03
```

## Next: Day3

**Goal:** gymnasium environment wrapping the physics layer. The env takes
`RobotParams`, `SimParams`, `RewardParams`, builds an initial state, and
exposes the standard `reset` / `step` / `observation` / `action_space`
API.

**Tasks:**
1. Implement `mech_rl/environment/robot_env.py` — `RobotEnv(gym.Env)`
   with torque-control action space, observation = `[q, qdot, target_xy]`.
2. Wire `SimParams.integrator` into a dispatch (semi-implicit Euler vs
   RK4) at env construction.
3. Reward function assembly from `RewardParams` coefficients.
4. Episode termination: success within `success_radius`, time limit,
   torque-limit violation.
5. Unit tests for the env (mock-free; uses real physics).
6. `Notes/Day_3.md` and `AGENTS.md` update.
