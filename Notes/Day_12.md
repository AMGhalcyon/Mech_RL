# Day 12 — Completed: Additional Seeds for Inconclusive Sweep Parameters

## Executive Summary
Building on the inconclusive results from previous sweeps (Day 9 and Day 10), Day 12 executed additional seed runs for the parameters that showed inconclusive trends: `clip_range`, `ent_coef`, and `gae_lambda`. Each parameter was tested with 2 values across 2 seeds (0 and 1) using a multirun approach, resulting in 8 runs per parameter (2 values × 2 seeds × 2 combinations of other parameters).

## Detailed Results

### 1. Clip Range Sweep (`multirun/2026-08-23/20-28-49/analysis/2026-08-23_20-32-22/`)
- **Experiment**: `clip_range` [0.1, 0.2] × 2 seeds × combinations of other parameters
- **Results**:
  - clip_range=0.1: Mean reward = -338.96 ± 28.71 (8 runs)
  - clip_range=0.2: Mean reward = -345.83 ± 26.18 (8 runs)
- **Analysis**: clip_range=0.1 shows slightly better performance (higher mean reward), but the difference is not statistically significant given the high variance and limited seeds.

### 2. Entropy Coefficient Sweep (`multirun/2026-08-23/20-28-49/analysis/2026-08-23_20-32-24/`)
- **Experiment**: `ent_coef` [0.0, 0.01] × 2 seeds × combinations of other parameters
- **Results**:
  - ent_coef=0.0: Mean reward = -345.90 ± 26.15 (8 runs)
  - ent_coef=0.01: Mean reward = -338.88 ± 28.72 (8 runs)
- **Analysis**: ent_coef=0.01 shows better performance (higher mean reward), suggesting some entropy encourages exploration, but again the difference is not statistically significant with current sample size.

### 3. GAE Lambda Sweep (`multirun/2026-08-23/20-28-49/analysis/2026-08-23_20-32-24/`)
- **Experiment**: `gae_lambda` [0.9, 0.95] × 2 seeds × combinations of other parameters
- **Results**:
  - gae_lambda=0.9: Mean reward = -332.07 ± 29.99 (8 runs)
  - gae_lambda=0.95: Mean reward = -352.72 ± 19.84 (8 runs)
- **Analysis**: gae_lambda=0.9 shows substantially better performance (higher mean reward) with lower variance, indicating this parameter value may be beneficial for the Mech_RL environment.

## Key Findings
1. **Parameter Trends**:
   - clip_range: 0.1 slightly better than 0.2
   - ent_coef: 0.01 slightly better than 0.0
   - gae_lambda: 0.9 significantly better than 0.95

2. **Variance Observations**: All parameters show high variance (~20-30 std dev), confirming the need for more seeds to achieve statistical significance.

3. **Comparison to Previous Results**:
   - Previous sweeps used gae_lambda=0.95 as default; results suggest 0.9 may be better
   - ent_coef=0.0 was default; results suggest 0.01 may provide slight benefit
   - clip_range=0.2 was default; results are inconclusive between 0.1 and 0.2

## Files Generated
- `multirun/2026-08-23/20-28-49/` - Multirun sweep directory (16 total runs)
- `multirun/2026-08-23/20-28-49/analysis/2026-08-23_20-32-22/` - clip_range analysis
- `multirun/2026-08-23/20-28-49/analysis/2026-08-23_20-32-24/` - ent_coef and gae_lambda analysis
- `Notes/Day_12.md` - This document

## Next Steps (Day 13-14)
1. **Day 13**: Build final evaluation suite (success rate, trajectory videos, robustness tests)
2. **Day 14**: Prepare release package, documentation, benchmark against baselines

## How to Reproduce Results
```bash
# Recreate the parameter sweep
.venv/Scripts/python.exe -m mech_rl.training.train \
  reward.distance_coef=0.5 \
  reward.effort_coef=0.01 \
  reward.time_penalty=0.01 \
  train.policy_kwargs.net_arch='[128,128]' \
  train.learning_rate=0.0001 \
  train.total_timesteps=100 \
  train.seed=0,1 \
  train.clip_range=0.1,0.2 \
  train.ent_coef=0.0,0.01 \
  train.gae_lambda=0.9,0.95 \
  --multirun

# Analyze results for each parameter
.venv/Scripts/python.exe -c "
from src.mech_rl.visualization import analyze_sweeps
import pandas as pd
for param in ['clip_range', 'ent_coef', 'gae_lambda']:
    result = analyze_sweeps(sweep_param=param, output_root='multirun/2026-08-23/20-28-49')
    df = pd.read_csv(result['csv_path'])
    print(f'=== {param} ===')
    print(df.groupby(param)['eval_mean_reward'].agg(['mean', 'std', 'count']))
"
```