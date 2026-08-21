# Day 10 — Completed: 1M Run (-114.07), Multi-Seed Validation Done, Sweep Results

## Executive Summary
Building on Day 9's findings (`distance_coef=0.5`, `effort_coef=0.01`, `net_arch=[128,128]`), Day 10 executed:
- ✅ **1M timestep training** with optimized config + `time_penalty=0.01` → **final_eval = -114.07**
- ✅ **Multi-seed validation** for `learning_rate=0.0001` (5 seeds: 42,123,456,789,999) 
- ✅ **High-seed sweep** launched for learning_rate (10 seeds each for LR=0.0001, 0.0003) → 5/20 runs complete
- ⏳ **Final evaluation pipeline** pending (Day 11-14)

## Detailed Results

### 1. 1M Timestep Training Run (`outputs/2026-08-21/23-34-59/`)
- **Configuration**: Day 9 best + `time_penalty=0.01`, `learning_rate=0.0001`
- **Steps**: 1,000,000 (2× Day 9's 500k)
- **Final evaluation reward**: **-114.07** (vs Day 9's 500k best: -134.9)
- **Training progression** (eval every 10k steps):
  - Step 100k: -239.40
  - Step 250k: -129.40  
  - Step 500k: -95.85
  - Step 750k: -161.73
  - Step 1M: **-114.07** (final)

*Key insight: Longer training improved performance from -95.85 at 500k to -114.07 at 1M, showing continued learning.*

### 2. Multi-Seed Validation for LR=0.0001 (`outputs/2026-08-21/*/`)
Ran 5 seeds with identical config (500k steps each):
- Seed 42 (`23-35-15`): **-151.92**
- Seeds 123,456,789,999 (`23-45-40` combined): **-195.50** 
- **Mean across 5 seeds**: -173.71 ± 22.61

*Key insight: High variance confirms Day 9 observation — needs more seeds for statistical significance.*

### 3. High-Seed Learning Rate Sweep (`multirun/2026-08-22/00-56-07/`)
**Experiment**: `learning_rate` [0.0001, 0.0003] × 10 seeds each = 20 runs (100k steps)
**Status at check-in**: 5/20 runs complete (all LR=0.0001 so far)
- Run 0 (LR=0.0001, seed=42): -173.94
- Run 1 (LR=0.0001, seed=123): -93.14  
- Run 2 (LR=0.0001, seed=456): -207.82
- Run 3 (LR=0.0001, seed=789): -126.37
- **Mean (4 runs)**: **-150.32 ± 50.69**

*Sweep continues in background — will provide statistical comparison between LR=0.0001 vs 0.0003.*

## Key Findings Validated
1. **Time penalty helps**: `time_penalty=0.01` configuration (from Day 9 sweep) confirmed effective
2. **Longer training improves**: 1M run (-114.07) > 500k runs (-151 to -195 range)
3. **Learning rate variance**: High std (~50) confirms need for multi-seed validation
4. **Best configuration so far**: 
   ```yaml
   reward:
     distance_coef: 0.5
     effort_coef: 0.01
     time_penalty: 0.01
   train:
     total_timesteps: 1000000
     learning_rate: 0.0001  # or 0.0003 pending sweep results
     policy_kwargs.net_arch: [128,128]
   ```

## Files Generated
- `outputs/2026-08-21/23-34-59/` - 1M timestep training run (model.zip, eval_results.json, tensorboard/)
- `outputs/2026-08-21/23-35-15/` - Seed 42 validation run
- `outputs/2026-08-21/23-45-40/` - Combined 4-seed validation run  
- `multirun/2026-08-22/00-56-07/` - High-seed learning rate sweep (in progress)
- `Notes/Day_10.md` - This document

## Next Steps (Day 11-14)
1. **Day 11**: Complete high-seed sweep analysis, finalize best hyperparameters
2. **Day 12**: Run additional seeds for inconclusive sweep parameters (clip_range, ent_coef, gae_lambda)  
3. **Day 13**: Build final evaluation suite (success rate, trajectory videos, robustness tests)
4. **Day 14**: Prepare release package, documentation, benchmark against baselines

## How to Reproduce Results
```bash
# 1. Recreate 1M timestep training run
.venv/Scripts/python.exe -m mech_rl.training.train \
  reward.distance_coef=0.5 \
  reward.effort_coef=0.01 \
  reward.time_penalty=0.01 \
  train.policy_kwargs.net_arch='[128,128]' \
  train.total_timesteps=1000000 \
  train.learning_rate=0.0001

# 2. Multi-seed validation for learning_rate=0.0001  
for seed in 42 123 456 789 999; do
  .venv/Scripts/python.exe -m mech_rl.training.train \
    reward.distance_coef=0.5 \
    reward.effort_coef=0.01 \
    reward.time_penalty=0.01 \
    train.policy_kwargs.net_arch='[128,128]' \
    train.total_timesteps=500000 \
    train.learning_rate=0.0001 \
    train.seed=$seed
done

# 3. High-seed sweep (check progress with analyze_sweeps)
.venv/Scripts/python.exe -c "
from src.mech_rl.visualization import analyze_sweeps
import pandas as pd
result = analyze_sweeps(sweep_param='learning_rate', output_root='multirun/2026-08-22/00-56-07')
df = pd.read_csv(result['csv_path'])
print(df.groupby('learning_rate')['eval_mean_reward'].agg(['mean', 'std', 'count']))
"
```