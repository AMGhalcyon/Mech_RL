# Day 4 — Hydra Config Integration

## Goal
Load all simulation parameters from YAML files using Hydra's structured configs, and scaffold the experiment loop entry point. This enables reproducible experiments and easy parameter sweeps.

## What I did

**configs/** directory with YAML files:
- `configs/robot.yaml` — `RobotParams` fields: link lengths, masses, inertias, friction, max_torque
- `configs/sim.yaml` — `SimParams` fields: dt, max_episode_steps, integrator
- `configs/reward.yaml` — `RewardParams` fields: distance_coef, effort_coef, smoothness_coef, success_bonus, success_radius, time_penalty
- `configs/train.yaml` — training hyperparameters: algorithm, total_timesteps, learning_rate, batch_size, n_steps, gamma, gae_lambda, clip_range, ent_coef, vf_coef, max_grad_norm, policy_kwargs, seed, device

**src/mech_rl/configs/** — Hydra structured configs:
- `src/mech_rl/configs/__init__.py` — exports RobotConfig, SimConfig, RewardConfig
- `src/mech_rl/configs/robot.py` — `RobotConfig` dataclass with `to_pydantic()` classmethod
- `src/mech_rl/configs/sim.py` — `SimConfig` dataclass with `to_pydantic()` classmethod
- `src/mech_rl/configs/reward.py` — `RewardConfig` dataclass with `to_pydantic()` classmethod
- `src/mech_rl/configs/train.py` — `TrainConfig` dataclass for PPO hyperparameters
- `src/mech_rl/configs/main_config.py` — `MechRLConfig` combining all configs

**src/mech_rl/training/** — Experiment loop:
- `src/mech_rl/training/__init__.py` — exports `make_env`, `train`
- `src/mech_rl/training/train.py` — Hydra entry point with:
  - ConfigStore registration for Hydra
  - `instantiate_env()` — builds `RobotEnv` from config
  - `make_env()` — creates seeded environment instances
  - `train()` — PPO training with SB3 (scaffold for Day 5)
  - `main()` — `@hydra.main` decorated entry point

**tests/unit/** — New tests:
- `tests/unit/test_hydra_configs.py` — tests for all structured configs, YAML loading
- `tests/unit/test_training_entry.py` — tests for `instantiate_env`, `make_env`

## Why I did it this way

1. **YAML files in project root** (`configs/`), not in `src/`. This follows Hydra's convention and matches where `utils/paths.py:CONFIG_DIR` points. Users can edit these without touching code.

2. **Dataclass structured configs with `to_pydantic()`**. Using Python dataclasses for Hydra's `ConfigStore` gives us type-safe config without coupling to Pydantic's validation. The `to_pydantic()` method converts to the validated Pydantic models we already have. This keeps validation logic in one place (`RobotParams`, etc.).

3. **ConfigStore per-group** (`robot`, `sim`, `reward` in addition to top-level `config`). This enables Hydra's compose API for flexible config composition and overrides. Users can run `train.py robot.l1=0.4` without a custom flag parser.

4. **Training entry point as a module**. Using `@hydra.main` with `config_path="../configs"` makes the configs findable. The entry point creates the environment, runs training, and saves the model to Hydra's output dir (timestamped runs/logs).

5. **SB3 integration in the same module**. `train()` uses Stable-Baselines3's PPO directly, with policy_kwargs support for architecture tuning. The placeholder is ready for Day 5 experiments.

## What's broken / annoying

- **No config overrides for target**. The initial target is fixed in `RobotEnv.__init__`. Consider adding `target` to the config for reproducibility across experiments.

- **No logging setup**. Hydra provides automatic hydra logging, but we should add experiment-level logging (MLflow integration) in Day 5.

- **No checkpoint callback**. Models are saved only at the end. SB3's `CheckpointCallback` should be added when training runs get long.

## How to check it works

```bash
# Run tests
.venv/Scripts/python.exe -m pytest tests/unit/test_hydra_configs.py tests/unit/test_training_entry.py -v

# Run training entry point (dry run with small timesteps)
.venv/Scripts/python.exe -m mech_rl.training.train train.total_timesteps=100

# Override config values
.venv/Scripts/python.exe -m mech_rl.training.train robot.max_torque=10.0 train.total_timesteps=200

# See Hydra help
.venv/Scripts/python.exe -m mech_rl.training.train --help
```

## Next up

**Day 5 — Training and evaluation loop.** Integrate MLflow tracking, run PPO training, evaluate on held-out episodes, log metrics to MLflow, save models. Add a config for evaluation runs.