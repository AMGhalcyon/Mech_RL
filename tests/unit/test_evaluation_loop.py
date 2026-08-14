"""Tests for the evaluation loop."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from mech_rl.evaluation.eval_loop import evaluate


def test_evaluation_loop():
    """Test evaluate function with a mock model."""
    # Mock model that always predicts [0.0, 0.0]
    mock_model = Mock()
    mock_model.predict.return_value = (np.array([0.0, 0.0]), None)

    # Mock environment
    mock_env = Mock()
    mock_env.reset.return_value = (
        {"q": np.array([0.0, 0.0]), "qdot": np.array([0.0, 0.0]), "target": np.array([0.3, 0.0])},
        {},
    )
    # Each episode ends after one step (done=True)
    mock_env.step.side_effect = [
        ({"q": np.array([0.0, 0.0]), "qdot": np.array([0.0, 0.0]), "target": np.array([0.3, 0.0])}, 0.1, True, False, {}),
        ({"q": np.array([0.0, 0.0]), "qdot": np.array([0.0, 0.0]), "target": np.array([0.3, 0.0])}, 0.2, True, False, {}),
    ]

    mean_reward = evaluate(mock_model, mock_env, num_episodes=2)

    # Should have called reset twice (once per episode)
    assert mock_env.reset.call_count == 2
    # Should have called step multiple times
    assert mock_env.step.call_count >= 2
    # Mean reward should be positive
    assert mean_reward > 0
