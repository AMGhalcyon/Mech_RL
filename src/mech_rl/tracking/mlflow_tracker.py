"""MLflow experiment tracking for Mech_RL."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def start_run(experiment_name: str = "mech_rl") -> None:
    """Start or set an MLflow tracking run."""
    import mlflow
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()


def log_params(params: dict) -> None:
    """Log a dictionary of hyperparameters."""
    import mlflow

    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log training/evaluation metrics."""
    import mlflow

    for k, v in metrics.items():
        mlflow.log_metric(k, v, step=step)


def log_model(model, artifact_path: str = "model") -> None:
    """Log the trained SB3 model as an artifact."""
    import mlflow

    mlflow.log_artifact(str(model), artifact_path)


def finish_run() -> None:
    """End the active MLflow run."""
    import mlflow

    mlflow.end_run()
