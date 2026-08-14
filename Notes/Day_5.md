# Day 5 — Training and evaluation loop

## Goal
Integrate MLflow tracking, run PPO training, evaluate on held-out episodes, log metrics to MLflow, save models. Add a config for evaluation runs.

## What I did

**MLflow Tracking**
- Created `src/mech_rl/tracking/mlflow_tracker.py` with functions to start/end runs, log params, metrics, and models.
- Updated `src/mech_rl/training/train.py` to:
  - Import MLflow tracker.
  - Start an MLflow run at the beginning of `main()`.
  - Log hyperparameters (from `cfg.train` and key config values).
  - Log evaluation metrics after training.
  - Log the trained model as an artifact.
  - Ensure the run is ended even if an error occurs (using try/finally).

**Checkpoint Callback**
- Added `stable_baselines3.common.callbacks.CheckpointCallback` in `train()` function.
- Checkpoints saved every `max(1000, total_timesteps // 10)` steps to the Hydra output directory under `checkpoints/`.

**Evaluation Loop**
- Created `src/mech_rl/evaluation/eval_loop.py` with an `evaluate()` function that:
  - Takes a trained model, environment, and number of episodes.
  - Runs deterministic evaluation (using `model.predict(..., deterministic=True)`).
  - Returns the mean cumulative reward.
- Integrated evaluation into `train.py`'s `main()` function after training, logging the mean reward to MLflow.

**Evaluation Config**
- Added `configs/evaluation.yaml` with:
  - `eval_episodes: 10`
  - `eval_interval: 1000` (placeholder for future use)
  - `seed_base: 42`

**Tests**
- Added `tests/unit/test_evaluation_loop.py` to test the evaluation loop with mocks.
- Fixed an existing test in `tests/unit/test_hydra_configs.py` by importing `ValidationError` from pydantic.

## How to check it works
```bash
# Run training with MLflow tracking (creates an MLflow run)
.venv/Scripts/python.exe -m mech_rl.training.train

# Override for quick test
.venv/Scripts/python.exe -m mech_rl.training.train train.total_timesteps=100

# After training, check:
# 1. Model saved to <hydra_output_dir>/model.zip
# 2. Checkpoints saved in <hydra_output_dir>/checkpoints/
# 3. MLflow run visible via `mlflow ui` (if MLflow tracking URI is set) or in ./mlruns/

# Run tests
.venv/Scripts/python.exe -m pytest tests/unit/test_evaluation_loop.py tests/unit/test_hydra_configs.py -v
```

## What's broken / annoying
- **MLflow requires a tracking URI**. By default, MLflow logs to `./mlruns`. For remote tracking, set `MLFLOW_TRACKING_URI` env var or adjust in code.
- **Evaluation during training** is not yet implemented (only post-training). Could be added via a callback in the future.
- **Checkpoint frequency** is heuristic; consider making it configurable.

## Next up
**Day 6 — Visualization and analysis.** Add TensorBoard/plotting scripts, analyze training curves, and visualize learned policies.
