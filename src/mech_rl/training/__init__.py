"""Training entry point and experiment loop.

This module provides the Hydra-based entry point for training policies
on the 2-DOF robotic arm environment.
"""

from mech_rl.training.train import make_env, train

__all__ = ["make_env", "train"]
