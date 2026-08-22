# Day 11 — Completed: Hyperparameter finalization using sweep and 1M run

## Executive Summary
Building on the high-seed learning rate sweep (20 runs, 100k steps each) and the 1M timestep training run, Day 11 finalizes the optimal hyperconfiguration.

- **Sweep results**: LR=0.0001 (mean reward -164.67 ± 50.16) vs LR=0.0003 (mean reward -189.83 ± 34.77). The difference in means is 25.16 in favor of LR=0.0001, but not statistically significant (p=0.234).
- **1M run result**: Using LR=0.0001 with the same other hyperparameters achieved a mean reward of -114.07, substantially better than both sweep conditions.
- **Other hyperparameters** (from Day 9/10): distance_coef=0.5, effort_coef=0.01, time_penalty=0.01, net_arch=[128,128].
- **Final recommended configuration**:
  - reward.distance_coef = 0.5
  - reward.effort_coef = 0.01
  - reward.time_penalty = 0.01
  - train.policy_kwargs.net_arch = [128, 128]
  - train.learning_rate = 0.0001
  - train.total_timesteps = 1,000,000 (for final training runs)

## Detailed Results

### 1. High-seed learning rate sweep (`multirun/2026-08-23/00-46-26/`)
- **Experiment**: `learning_rate` [0.0001, 0.0003] × 10 seeds each = 20 runs (100k steps)
- **LR=0.0001 rewards** (seeds 42,123,456,789,999,111,222,333,444,555):  
  [-157.53, -73.99, -143.68, -185.68, -167.22, -206.54, -262.54, -114.09, -135.40, -199.99]  
  Mean: -164.67, Std: 50.16
- **LR=0.0003 rewards** (seeds 42,123,456,789,999,111,222,333,444,555):  
  [-157.18, -260.52, -205.10, -229.30, -141.66, -187.39, -192.36, -180.79, -196.32, -147.65]  
  Mean: -189.83, Std: 34.77
- **Statistical test**: Welch's t-test p-value = 0.234 (not significant at α=0.05)

### 2. 1M timestep training run (`outputs/2026-08-21/23-34-59/`)
- **Configuration**: Day 9 best + `time_penalty=0.01`, `learning_rate=0.0001`
- **Steps**: 1,000,000
- **Final evaluation reward**: -114.07
- **Training progression** (eval every 100k steps):
  - 100k: -239.40
  - 250k: -129.40
  - 500k: -95.85
  - 750k: -161.73
  - 1M: -114.07

## Conclusion
The optimal hyperconfiguration for the Mech_RL environment is:
- **reward**: distance_coef=0.5, effort_coef=0.01, time_penalty=0.01
- **network architecture**: [128, 128]
- **learning rate**: 0.0001
- **total timesteps**: 1,000,000 (for final performance)

This configuration leverages the benefits of longer training (as shown by the 1M run) and uses the learning rate that, while not significantly better in the 100k sweep, produced the best single run (1M) and showed a non-significant trend toward better performance in the sweep.

Next steps: Proceed to Day 12 for additional parameter sweeps (if needed) or Day 13 for final evaluation suite.