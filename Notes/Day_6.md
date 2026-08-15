# Day 6 — Visualization and Analysis

## Goal
Add TensorBoard logging integration, build plotting scripts to analyze training curves, and create learned policy visualization tools (arm animation, trajectory plots). This enables monitoring training progress and understanding learned behaviors.

## What I did

**TensorBoard Integration**
- Updated `src/mech_rl/training/train.py` to enable TensorBoard logging in PPO training
- Added `tensorboard_log` parameter pointing to Hydra output directory
- Checkpoints and TensorBoard logs are now saved together in the experiment output directory

**Visualization Module**
- Created comprehensive `src/mech_rl/visualization/__init__.py` with:
  - `tensorboard_scalars_to_csv()`: Extract scalar data from TensorBoard event files to CSV
  - `plot_training_curves()`: Generate training curve plots with optional smoothing
  - `visualize_learned_policy()`: Run episodes and create detailed visualizations:
    - Arm configuration plots (showing start/middle/end poses)
    - End-effector trajectory vs target
    - Joint angles over time
    - Rewards over time (step and cumulative)
  - `_plot_episode_trajectory()` and `_plot_training_summary()`: Helper functions for episode-level and summary plots
  - `create_training_report()`: Generate HTML report combining all visualizations

**Tests**
- Added `tests/unit/test_visualization.py` with:
  - Import error handling tests (graceful degradation when tensorboard unavailable)
  - Basic functionality tests using mocks
  - Module import verification

## How to check it works

```bash
# Run training with TensorBoard logging (creates tensorboard logs in output directory)
.venv/Scripts/python.exe -m mech_rl.training.train

# View TensorBoard logs interactively
tensorboard --logdir <path-to-latest-output-dir>

# Or use the visualization utilities programmatically:
# After training, extract data and create plots:
from mech_rl.visualization import tensorboard_scalars_to_csv, plot_training_curves
csv_paths = tensorboard_scalars_to_csv("./outputs/2026-08-15/14-30-00")
plot_paths = plot_training_curves("./outputs/2026-08-15/14-30-00")

# Create full report:
from mech_rl.visualization import create_training_report
report_path = create_training_report("./outputs/2026-08-15/14-30-00")
```

## Files Modified
- `src/mech_rl/training/train.py` - Added TensorBoard logging to PPO
- `src/mech_rl/visualization/__init__.py` - New visualization utilities module
- `tests/unit/test_visualization.py` - New test file for visualization module

## Next Steps (Day 7)
- Experiment with different reward coefficients and analyze impact on learning
- Try different PPO hyperparameters (learning rate, network architecture)
- Record and save videos of learned policies
- Begin hyperparameter sweeping experiments using Hydra's multirun capability