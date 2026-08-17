# Day 8 — Hyperparameter Sweep Execution and Analysis

## Goal
Execute a Hydra multirun sweep over key hyperparameters (reward.distance_coef and train.learning_rate) and use the `analyze_sweeps()` utility from Day 7 to visualize and compare the results.

## What I did

### 1. Hydra Multirun Sweep
- Ran the training script with Hydra's multirun flag (`-m`) over a grid of two parameters:
  - `reward.distance_coef`: [0.5, 1.0, 2.0]
  - `train.learning_rate`: [0.0001, 0.0003, 0.001]
- This launched 9 training jobs (3 × 3) in the `multirun/2026-08-17/18-58-50/` directory.
- Each job trained a PPO policy for 100,000 timesteps and saved an `eval_results.json` file containing the final evaluation mean reward.

### 2. Sweep Analysis with `analyze_sweeps()`
- Used the `analyze_sweeps()` function from `src/mech_rl/visualization/__init__.py` to:
  - Recursively scan the multirun output directory for `eval_results.json` files.
  - Extract the swept parameter value (`learning_rate`) and the corresponding `eval_mean_reward`.
  - Generate a CSV file and a scatter plot comparing the swept parameter to the evaluation reward.
- The analysis revealed the relationship between learning rate and performance (higher learning rates did not necessarily yield better rewards in this sweep).

### 3. Results and Observations
- The sweep produced 9 eval results (some duplicate learning rates due to the grid).
- The analysis CSV and plot are saved under:
  - `multirun/2026-08-17/18-58-50/analysis/2026-08-17_19-11-18/sweep_analysis_learning_rate.csv`
  - `multirun/2026-08-17/18-58-50/analysis/2026-08-17_19-11-18/sweep_plot_learning_rate.png`
- The sweep demonstrated that the visualization and analysis tools from Day 7 are working correctly and can be used to interpret hyperparameter sensitivity.

## How to Check It Works
```bash
# Re-run the sweep (optional)
.venv/Scripts/python.exe -m mech_rl.training.train -m reward.distance_coef=0.5,1.0,2.0 train.learning_rate=0.0001,0.0003,0.001

# Or run analysis on existing multirun output
.venv/Scripts/python.exe -c "
from src.mech_rl.visualization import analyze_sweeps
result = analyze_sweeps(sweep_param='learning_rate', output_root='multirun/2026-08-17/18-58-50')
print('CSV:', result['csv_path'])
print('Plot:', result['plot_path'])
```

## Files Created or Modified
- `multirun/2026-08-17/18-58-50/` — contains 9 Hydra job subdirectories, each with `eval_results.json`
- `multirun/2026-08-17/18-58-50/analysis/` — contains CSV and PNG outputs from `analyze_sweeps()`
- No source code changes; this day focused on execution and validation of existing sweep infrastructure.

## Next Steps
With the sweep and analysis pipeline validated, Day 9 could focus on:
- Refining the reward function based on sweep insights
- Experimenting with different network architectures or training hyperparameters
- Beginning preparation for final evaluation and release (Day 14).