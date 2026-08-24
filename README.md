# Mech_RL

Energy-aware reinforcement learning for a torque-controlled 2-DOF planar robotic arm. The physics core is analytical: forward kinematics, Lagrangian dynamics (Coriolis, centrifugal, gravity, viscous friction), symplectic Euler or RK4 integration. The Gymnasium environment wraps this and exposes continuous joint-torque actions with structured observations (joint angles and velocities, end-effector target). The reward is a linear combination of distance to target, effort (torque squared), smoothness (torque rate), a sparse success bonus, and a time penalty. All parameters are validated Pydantic models that serialize to YAML; `extra="forbid"` catches config typos at load time. The framework is modular by design: physics, environment, and control are separate, so replacing the RL algorithm does not require rewriting dynamics.

## Why PPO

PPO was chosen because it handles continuous control with clipped surrogate updates, which stabilize training when the reward has multiple competing terms (distance, effort, time). The framework uses `MultiInputPolicy` (structured observations) with a `[128, 128]` network. Hyperparameter optimization (clip_range=0.1, ent_coef=0.01, gae_lambda=0.9) was performed across multi-seed sweeps over 1M timesteps. A simpler algorithm (e.g., DQN) is not appropriate because actions are continuous and multi-dimensional.

## Project root

- `README.md` — this file — overview and quick start
- `pyproject.toml` — package metadata (version 1.0.0), dependencies, ruff and pytest configuration; no package integration required for standalone scripts
- `configs/` — Hydra YAML configurations for robot, simulation, reward, and training parameters
- `src/mech_rl/` — source package: physics (`dynamics.py`), environment (`robot_env.py`), domain (`parameters.py` with validated Pydantic models), evaluation (`eval_loop.py`, `final_eval.py`, `baseline_benchmark.py` — standalone comparison tool), visualization, and training
- `tests/` — 131 unit tests (domain, environment, evaluation, visualization); all pass
- `Notes/` — daily progress logs from project scaffolding through final evaluation and release preparation
- `outputs/` — trained models, evaluation results, GIFs, JSON summaries
- `multirun/` — hyperparameter sweep outputs from optimization
- `graphify-out/` — persistent knowledge graph of the codebase (not source)
- `AGENTS.md` — documentation of project agents

## Installation

Requires Python >= 3.11. The environment uses `torch`, `stable-baselines3`, `gymnasium`, `hydra-core`, `pydantic`, `numpy`, and `matplotlib`.

```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows
.venv/Scripts/activate
# On Linux/macOS
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

The editable install links `src/` to the installed package so changes are reflected immediately without reinstall.

## Quick start (detailed)

### Training a model

The entry point is a Hydra-based module:

```bash
.venv/Scripts/python.exe -m mech_rl.training.train
```

This loads the default config from `configs/config.yaml`, builds the environment from validated parameters, seeds the random generators for reproducibility, creates a PPO model with `MultiInputPolicy`, sets up TensorBoard logging, and trains for the configured number of timesteps. During training, a checkpoint callback saves the model every 10% of the run, and an evaluation callback logs mean reward every `eval_interval` steps.

To override parameters at the command line:

```bash
.venv/Scripts/python.exe -m mech_rl.training.train robot.l1=0.4 sim.integrator=rk4 train.learning_rate=0.0001
```

To run a sweep (e.g., over clip_range):

```bash
.venv/Scripts/python.exe -m mech_rl.training.train --multirun train.clip_range=0.1,0.2,0.3
```

The best hyperparameters found through sweeps are `clip_range=0.1`, `ent_coef=0.01`, `gae_lambda=0.9`, with `net_arch=[128,128]` and `learning_rate=0.0001`.

### Evaluating a trained model

Point the evaluation suite at the saved `.zip`:

```python
import sys
sys.path.insert(0, 'src')
from stable_baselines3 import PPO
from mech_rl.evaluation.final_eval import run_final_evaluation

model = PPO.load(r'outputs/2026-08-24/18-57-59/model.zip')
summary = run_final_evaluation(model, output_dir='outputs/final_eval_final', num_episodes=20)
print('Success Rate:', summary['success_rate'])
print('Mean Reward:', summary['mean_reward'])
print('GIF:', summary['gif_path'])
```

This computes success rate, mean reward, episode length, generates trajectory GIFs and per-episode plots, runs robustness tests (target position variations, link-length perturbations at ±5%), and writes `eval_summary.json` and `report.html`.

### Running baselines

Compare the RL agent against random control and a naive P-controller:

```bash
python src/mech_rl/evaluation/baseline_benchmark.py
```

This script is standalone: it requires no tests, no package integration, and no external dependencies beyond the environment itself. It exists solely to provide reference performance numbers for the benchmark step of previous progress.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/unit -v
```

Results: 131 passed, 3005 warnings (mostly imageio deprecation), completed in under 5 minutes. The suite covers physics consistency (state matches manual integration, multi-step consistency, integrator dispatch), parameter validation (`RobotParams`, `SimParams`, `RewardParams`), environment reset and step behavior, evaluation output generation, and visualization module imports.

## Limitations

The framework is purpose-built for a single 2-DOF planar arm. It does not support multi-arm coordination, 3D kinematics, or real-time hardware interfaces. The physics model assumes uniform rods with fixed inertia; non-uniform or flexible links are out of scope. The evaluation robustness tests vary targets and link lengths deterministically but do not inject sensor noise, actuator delay, or external disturbances — these could be added by extending `RobotEnv` but are not implemented. The baseline benchmark is standalone and does not integrate into the evaluation pipeline; it must be run manually. The README language is concise; deeper theory (Lagrangian derivation, symplectic integration proof) is not included and would require supplementary notes.

## Future work

- Add noise and disturbance injection to the environment for more realistic robustness testing.
- Extend the physics to 3-DOF (e.g., spherical joint) or multi-link serial chains.
- Implement PID or model-predictive control baselines for direct comparison against RL in the evaluation suite.
- Add real-time hardware interface for physical arm control.
- Expand documentation with derivation notes and interactive tutorials.
- Integrate the baseline benchmark into `run_final_evaluation()` so comparisons are automatic rather than manual.
