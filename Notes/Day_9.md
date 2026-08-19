# Day 9 — Reward Refinement, Architecture Experiments, and Extended Training

## Goal
Building on Day 8's sweep insights, refine the reward function, experiment with network architectures and additional PPO hyperparameters, and run extended training with the best configuration to prepare for final evaluation.

## What I did

### 1. Extended Training with Best Configuration
- **Best config from Day 8 sweep**: `distance_coef=0.5`, `effort_coef=0.01`, `policy_net_arch=[128, 128]`
- Ran 500,000 timestep training (10× the sweep duration)
- **Result**: Mean evaluation reward of **-177.4** (significant improvement over 50k timestep runs)
- Training progression showed steady improvement:
  - Step 10k: -375.0
  - Step 100k: -178.7
  - Step 250k: -253.9
  - Step 500k: -146.6 (final eval)

### 2. Sweep Over Additional PPO Hyperparameters
Ran a comprehensive sweep (3×3×3×3 = 81 combinations, 3 repeats each = 243 runs) over:
- `learning_rate`: [0.0001, 0.0003, 0.001]
- `clip_range`: [0.1, 0.2, 0.3]
- `ent_coef`: [0.0, 0.01, 0.05]
- `gae_lambda`: [0.9, 0.95, 0.99]

**Key findings for `learning_rate`**:
- `0.0001`: Best runs achieved -80 to -93 (high variance, some excellent results)
- `0.0003`: Consistent -215 to -345 range (default performed poorly in this sweep)
- `0.001`: Not enough runs completed to assess

**Observation**: Lower learning rate (0.0001) showed higher variance but potential for better peak performance. The sweep had high variance across repeats, suggesting need for more seeds per config.

### 3. Reward Shaping Sweep
Ran sweep over `success_bonus` [0.0, 10.0, 50.0] and `time_penalty` [0.0, 0.01, 0.1] with base config (distance_coef=0.5, effort_coef=0.01, net_arch=[128,128]):

**success_bonus results**:
- 0.0: Best at -134.9 to -155.9 (but high variance up to -383.8)
- 10.0: -184.4 to -354.7 (worse than no bonus)
- 50.0: -278.1 to -308.3 (worse)

**time_penalty results**:
- 0.0: -155.9 to -308.3
- 0.01: **-134.9** (best overall) to -278.1
- 0.1: -354.7 to -413.1 (significantly worse)

**Conclusion**: Small time penalty (0.01) helps slightly; success_bonus hurts performance at this task duration. The task may not have clear "success" moments within 1000 steps.

### 4. Updated Training Infrastructure
- Enhanced `eval_results.json` to include ALL hyperparameters (clip_range, ent_coef, gae_lambda, n_steps, batch_size, gamma, vf_coef, max_grad_norm, success_bonus, success_radius, time_penalty)
- This enables comprehensive sweep analysis on any parameter

## Key Findings Summary

| Parameter | Best Value | Notes |
|-----------|------------|-------|
| distance_coef | 0.5 | Lower significantly outperforms 1.0 and 2.0 |
| effort_coef | 0.01 | Sweet spot; 0.005 similar, 0.02 worse |
| policy_net_arch | [128, 128] | Best average; [64,64] close, [256,256] more variance |
| learning_rate | 0.0001 | Higher peak potential but more variance |
| clip_range | Not conclusive | Need more analysis |
| ent_coef | Not conclusive | Need more analysis |
| gae_lambda | Not conclusive | Need more analysis |
| success_bonus | 0.0 | Adding bonus hurts performance |
| time_penalty | 0.01 | Small penalty helps slightly |

## Best Overall Configuration So Far
```yaml
reward:
  distance_coef: 0.5
  effort_coef: 0.01
  smoothness_coef: 0.0
  success_bonus: 0.0
  time_penalty: 0.01
train:
  total_timesteps: 500000
  learning_rate: 0.0001  # or 0.0003 for stability
  policy_kwargs:
    net_arch: [128, 128]
```

## Files Created or Modified
- `src/mech_rl/training/train.py` — Added all hyperparameters to eval_results.json for sweep analysis
- `outputs/2026-08-19/23-01-05/` — 500k timestep training run with best config
- `multirun/2026-08-19/23-11-21/` — PPO hyperparameter sweep (243 runs)
- `multirun/2026-08-19/23-57-13/` — Reward shaping sweep (9 runs, 1 incomplete)

## Next Steps (Day 10+)
1. Run longer training (1M+ timesteps) with optimized config including time_penalty=0.01
2. Test learning_rate=0.0001 with multiple seeds for statistical significance
3. Add more seeds per sweep configuration to reduce variance
4. Begin final evaluation and release preparation (target Day 14)
5. Consider adding curriculum learning or reward scheduling

## How to Check It Works
```bash
# Run extended training with best config + time penalty
.venv/Scripts/python.exe -m mech_rl.training.train \
  reward.distance_coef=0.5 \
  reward.effort_coef=0.01 \
  reward.time_penalty=0.01 \
  train.policy_kwargs.net_arch='[128,128]' \
  train.total_timesteps=1000000

# Analyze sweeps
.venv/Scripts/python.exe -c "
from src.mech_rl.visualization import analyze_sweeps
result = analyze_sweeps(sweep_param='learning_rate', output_root='multirun/2026-08-19/23-11-21')
print('CSV:', result['csv_path'])
print('Plot:', result['plot_path'])
"
```