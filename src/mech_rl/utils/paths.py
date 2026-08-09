"""Project-root resolution and conventional paths.

The package needs a stable reference to the repo root regardless of
where Python is invoked from (scripts, tests, notebooks).
"""

from __future__ import annotations

from pathlib import Path

# Markers that identify the project root. Ordered most-specific first.
_ROOT_MARKERS = ("pyproject.toml", ".git")


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (or cwd) until a root marker is found.

    Falls back to the starting directory if no marker is found, so the
    function never raises in odd install layouts.
    """
    start = (start or Path.cwd()).resolve()
    current = start
    for parent in [current, *current.parents]:
        for marker in _ROOT_MARKERS:
            if (parent / marker).exists():
                return parent
    return start


# Resolved once at import time. Cheap; the project structure is fixed.
PROJECT_ROOT: Path = find_project_root()

# Conventional subdirectories. Created on first write, not on import.
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
NOTES_DIR: Path = PROJECT_ROOT / "Notes"
TESTS_DIR: Path = PROJECT_ROOT / "tests"


def ensure_directory(path: Path) -> Path:
    """Create `path` (and parents) if missing. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "EXPERIMENTS_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "METRICS_DIR",
    "NOTES_DIR",
    "TESTS_DIR",
    "ensure_directory",
    "find_project_root",
]
