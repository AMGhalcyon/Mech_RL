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


def test_record_gif_import_error():
    """Test that ImportError is raised when imageio is not available."""
    with patch.dict('sys.modules', {'imageio': None}):
        # Should not raise ImportError on import (optional dep), but should
        # return gif_path=None when imageio missing.
        from mech_rl.visualization import record_gif
        from unittest.mock import Mock
        import numpy as np
        from mech_rl.environment import RobotEnv
        from mech_rl.domain.parameters import RobotParams

        mock_env = Mock(spec=RobotEnv)
        mock_env.robot_params = RobotParams(
            l1=0.3, l2=0.3, m1=1.0, m2=1.0, i1=0.03, i2=0.03, friction=0.05, max_torque=5.0
        )
        mock_env.render.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_env.reset.return_value = ({"q": np.array([0.0, 0.0]), "qdot": np.array([0.0, 0.0])}, {})
        mock_env.step.return_value = ({"q": np.array([0.1, 0.1]), "qdot": np.array([0.0, 0.0])},
                                      0.1, True, False, {"is_success": False})

        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0.0, 0.0]), None)

        result = record_gif(mock_model, mock_env, num_episodes=1, output_dir="/tmp")
        assert result['gif_path'] is None
        # Frames and episode data should still be present
        assert isinstance(result['frame_paths'], list)
        assert len(result['frame_paths']) >= 1


def test_record_gif_basic():
    """Test basic structure of record_gif output."""
    from unittest.mock import Mock
    import numpy as np
    from mech_rl.environment import RobotEnv
    from mech_rl.visualization import record_gif
    from mech_rl.domain.parameters import RobotParams

    mock_env = Mock(spec=RobotEnv)
    mock_env.robot_params = RobotParams(
        l1=0.3, l2=0.3, m1=1.0, m2=1.0, i1=0.03, i2=0.03, friction=0.05, max_torque=5.0
    )
    mock_env.render.return_value = np.zeros((20, 20, 3), dtype=np.uint8)
    mock_env.reset.return_value = ({"q": np.array([0.0, 0.0]), "qdot": np.array([0.0, 0.0])}, {})
    mock_env.step.return_value = ({"q": np.array([0.1, 0.1]), "qdot": np.array([0.0, 0.0])},
                                  0.1, True, False, {"is_success": False})

    mock_model = Mock()
    mock_model.predict.return_value = (np.array([0.5, 0.5]), None)

    result = record_gif(mock_model, mock_env, num_episodes=1, output_dir=None, fps=10)

    assert 'episodes' in result
    assert 'frame_paths' in result
    assert 'gif_path' in result  # May be None if imageio not installed
    assert isinstance(result['episodes'], list)
    assert isinstance(result['frame_paths'], list)
    assert len(result['episodes']) == 1
    assert len(result['frame_paths']) >= 1  # at least one frame per step
    # Check episode dict keys (matches visualize_learned_policy format)
    ep = result['episodes'][0]
    assert 'reward' in ep
    assert 'length' in ep
    assert 'actions' in ep
    assert 'success' in ep


def test_analyze_sweeps_no_runs():
    """Test that analyze_sweeps raises FileNotFoundError when no run directories exist."""
    from mech_rl.visualization import analyze_sweeps
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty directory
        with pytest.raises(FileNotFoundError, match="No eval_results.json files found"):
            analyze_sweeps(output_root=tmpdir)


def test_analyze_sweeps_basic():
    """Test basic structure of analyze_sweeps output."""
    from mech_rl.visualization import analyze_sweeps
    import tempfile
    import json
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)

        # Create two fake run directories with eval_results.json
        for i, (lr, reward) in enumerate([(0.001, 0.5), (0.003, 0.8)]):
            run_dir = output_root / f"run_{i}"
            run_dir.mkdir()
            eval_results = {
                "learning_rate": lr,
                "eval_mean_reward": reward,
                "total_timesteps": 1000
            }
            (run_dir / "eval_results.json").write_text(json.dumps(eval_results))

        # Run the analysis
        result = analyze_sweeps(sweep_param="learning_rate", output_root=output_root)

        # Check return structure
        assert 'sweep_values' in result
        assert 'rewards' in result
        assert 'csv_path' in result
        assert 'plot_path' in result
        assert 'num_runs' in result

        assert result['num_runs'] == 2
        assert len(result['sweep_values']) == 2
        assert len(result['rewards']) == 2

        # Values should be sorted by sweep_param (learning_rate)
        assert result['sweep_values'] == [0.001, 0.003]
        assert result['rewards'] == [0.5, 0.8]

        # Check that files were created
        assert result['csv_path'].exists()
        assert result['plot_path'].exists()
        assert result['csv_path'].suffix == ".csv"
        assert result['plot_path'].suffix == ".png"


def test_create_training_report_import_error():
    """Test that ImportError is raised when tensorboard is not available."""
    with patch.dict('sys.modules', {'tensorboard': None}):
        with pytest.raises(ImportError, match="tensorboard not installed"):
            create_training_report("/fake/logdir", "/fake/output")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
