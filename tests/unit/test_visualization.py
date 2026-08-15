"""Tests for the visualization module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from mech_rl.domain.parameters import RobotParams
from mech_rl.environment import RobotEnv
from mech_rl.visualization import (
    create_training_report,
    plot_training_curves,
    tensorboard_scalars_to_csv,
    visualize_learned_policy,
)


def test_tensorboard_scalars_to_csv_import_error():
    """Test that ImportError is raised when tensorboard is not available."""
    with patch.dict('sys.modules', {'tensorboard': None}):
        with pytest.raises(ImportError, match="tensorboard not installed"):
            tensorboard_scalars_to_csv("/fake/logdir")


def test_plot_training_curves_import_error():
    """Test that ImportError is raised when tensorboard is not available."""
    with patch.dict('sys.modules', {'tensorboard': None}):
        with pytest.raises(ImportError, match="tensorboard not installed"):
            plot_training_curves("/fake/logdir")


def test_visualize_learned_policy_basic():
    """Test basic visualization function structure."""
    # Mock environment and model
    mock_env = Mock(spec=RobotEnv)
    mock_env.robot_params = RobotParams(
        l1=0.3, l2=0.2, m1=0.5, m2=0.4,
        i1=0.01, i2=0.01, friction=0.1, max_torque=5.0
    )
    mock_env.reset.return_value = ({"q": np.array([0.0, 0.0]), "qdot": np.array([0.0, 0.0])}, {})
    # Make the environment terminate after 2 steps to avoid infinite loop
    mock_env.step.side_effect = [
        ({"q": np.array([0.1, 0.1]), "qdot": np.array([0.0, 0.0])}, 0.1, False, False, {"is_success": False}),
        ({"q": np.array([0.2, 0.2]), "qdot": np.array([0.0, 0.0])}, 0.2, True, False, {"is_success": False}),  # terminated=True
    ]

    mock_model = Mock()
    mock_model.predict.return_value = (np.array([0.0, 0.0]), None)

    # Test function runs without error
    with tempfile.TemporaryDirectory() as tmpdir:
        result = visualize_learned_policy(
            mock_model, mock_env, num_episodes=1, output_dir=tmpdir
        )

        # Check return structure
        assert 'episodes' in result
        assert 'success_rate' in result
        assert 'mean_reward' in result
        assert 'mean_episode_length' in result
        assert len(result['episodes']) == 1


def test_create_training_report_basic():
    """Test report creation function structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logdir = Path(tmpdir) / "logs"
        logdir.mkdir()

        # Create a dummy events file to avoid FileNotFoundError
        (logdir / "events.out.tfevents.12345").touch()

        # Test function runs without error (will fail on tensorboard import but that's expected in test env)
        try:
            report_path = create_training_report(str(logdir))
            # If we get here, tensorboard is available in test env
            assert report_path.exists()
            assert report_path.suffix == ".html"
        except ImportError:
            # Expected in test environment without tensorboard
            pass


def test_visualization_module_imports():
    """Test that all expected functions can be imported."""
    from mech_rl.visualization import (
        create_training_report,
        plot_training_curves,
        tensorboard_scalars_to_csv,
        visualize_learned_policy,
    )

    # All functions should be callable
    assert callable(tensorboard_scalars_to_csv)
    assert callable(plot_training_curves)
    assert callable(visualize_learned_policy)
    assert callable(create_training_report)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
