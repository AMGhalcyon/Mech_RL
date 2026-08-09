"""Tests for seeding utilities."""

from __future__ import annotations

import random

import numpy as np
import pytest

from mech_rl.utils.reproducibility import set_seed


def test_set_seed_rejects_non_int():
    with pytest.raises(TypeError):
        set_seed("42")  # type: ignore[arg-type]


def test_same_seed_produces_same_numpy_sequence():
    set_seed(123)
    a = np.random.rand(10)
    set_seed(123)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)


def test_same_seed_produces_same_python_sequence():
    set_seed(7)
    a = [random.random() for _ in range(5)]
    set_seed(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_different_seeds_produce_different_sequences():
    set_seed(1)
    a = np.random.rand(10)
    set_seed(2)
    b = np.random.rand(10)
    assert not np.array_equal(a, b)


def test_set_seed_seeds_torch():
    import torch

    set_seed(99)
    a = torch.rand(5)
    set_seed(99)
    b = torch.rand(5)
    assert torch.equal(a, b)
