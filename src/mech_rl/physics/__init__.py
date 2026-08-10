"""Physics layer: kinematics, dynamics, integrator. Framework-independent.

The only allowed imports in this subpackage are ``numpy`` and other
``mech_rl`` modules. No ``torch``, ``gymnasium``, ``matplotlib``, or
``stable_baselines3`` — this layer must stay runnable in a pure-numpy
test environment.
"""

from mech_rl.physics.dynamics import (
    G_ACCEL,
    coriolis,
    equation_of_motion,
    friction,
    gravity,
    mass_matrix,
    potential_energy,
)
from mech_rl.physics.integrator import rk4, semi_implicit_euler
from mech_rl.physics.kinematics import forward_kinematics, jacobian

__all__ = [
    "G_ACCEL",
    "coriolis",
    "equation_of_motion",
    "forward_kinematics",
    "friction",
    "gravity",
    "jacobian",
    "mass_matrix",
    "potential_energy",
    "rk4",
    "semi_implicit_euler",
]
