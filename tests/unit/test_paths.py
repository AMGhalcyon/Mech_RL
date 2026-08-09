"""Tests for project-root resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from mech_rl.utils.paths import (
    PROJECT_ROOT,
    find_project_root,
)


def test_project_root_is_set():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_find_project_root_from_subdir(tmp_path: Path):
    # Build a fake project tree: tmp/pyproject.toml, tmp/sub/deeper/
    (tmp_path / "pyproject.toml").touch()
    deeper = tmp_path / "sub" / "deeper"
    deeper.mkdir(parents=True)
    resolved = find_project_root(deeper)
    assert resolved == tmp_path


def test_find_project_root_falls_back_when_no_marker(tmp_path: Path):
    # No markers anywhere; should return the start directory, not crash.
    start = tmp_path / "isolated"
    start.mkdir()
    result = find_project_root(start)
    assert result == start


def test_conventional_paths_are_under_root():
    from mech_rl.utils.paths import (
        CONFIG_DIR,
        EXPERIMENTS_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        TESTS_DIR,
    )

    for p in [CONFIG_DIR, EXPERIMENTS_DIR, MODELS_DIR, RESULTS_DIR, TESTS_DIR]:
        assert p.is_absolute()
        assert PROJECT_ROOT in p.parents or p == PROJECT_ROOT


def test_ensure_directory_creates_path(tmp_path: Path):
    from mech_rl.utils.paths import ensure_directory

    target = tmp_path / "a" / "b" / "c"
    result = ensure_directory(target)
    assert result.exists()
    assert result.is_dir()
    # Idempotent.
    ensure_directory(target)
    assert result.exists()


@pytest.mark.parametrize(
    "marker",
    ["pyproject.toml", ".git"],
)
def test_find_root_accepts_each_marker(tmp_path: Path, marker: str):
    (tmp_path / marker).touch()
    assert find_project_root(tmp_path) == tmp_path
