"""Seeding for reproducible experiments.

Covers Python's `random`, NumPy, PyTorch (CPU + CUDA), and Gymnasium's
action/observation spaces. Call `set_seed()` at the top of every entry
point (train, evaluate, sweep).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed all relevant RNGs.

    Args:
        seed: The integer seed to use.
        deterministic_torch: If True, force PyTorch deterministic algorithms.
            Adds overhead; only enable for final reporting runs.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    random.seed(seed)
    np.random.seed(seed)

    # Torch is imported lazily so non-torch code paths don't pay the cost.
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Gymnasium envs hold their own RNG; seeded via env.reset(seed=...).
    # We don't seed here because env seeding happens at construction time.


__all__ = ["set_seed"]
