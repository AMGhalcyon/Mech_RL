"""Lightweight tests for final evaluation suite."""
from pathlib import Path
import numpy as np

from mech_rl.evaluation.final_eval import run_final_evaluation
from mech_rl.domain.parameters import RobotParams, SimParams, RewardParams
from mech_rl.environment import RobotEnv


def test_final_eval_runs_without_crash():
    # Use a dummy random policy instead of trained model for speed
    class DummyModel:
        def predict(self, obs, deterministic=True):
            return np.zeros(2), None

    result = run_final_evaluation(DummyModel(), output_dir="outputs/test_final_eval", num_episodes=2, robustness_targets=2)
    assert "success_rate" in result
    assert "mean_reward" in result
    assert "robustness" in result
    assert (Path("outputs/test_final_eval") / "eval_summary.json").exists()