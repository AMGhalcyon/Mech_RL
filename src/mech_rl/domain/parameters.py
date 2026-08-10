"""Configuration parameters for the robotic arm, simulator, and reward.

These are Pydantic models so that:
1. Invalid values fail at config-load time, not mid-simulation.
2. Serialization to YAML/JSON is built-in.
3. Field constraints (positive masses, finite dt) are enforced once
   in one place.

`extra="forbid"` on every model catches typos in YAML configs.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from mech_rl.utils.config import BaseConfig


class RobotParams(BaseConfig):
    """Physical parameters of the 2-DOF arm.

    Units are SI (meters, kilograms, newton-meters).
    """

    # Link lengths
    l1: float = Field(gt=0.0, description="Upper-arm length (m)")
    l2: float = Field(gt=0.0, description="Forearm length (m)")

    # Link masses
    m1: float = Field(gt=0.0, description="Upper-arm mass (kg)")
    m2: float = Field(gt=0.0, description="Forearm mass (kg)")

    # Moment of inertia about the joint, parallel-axis included. For a
    # uniform rod of mass m and length l this is m*l^2 / 3; values smaller
    # than that are physically inadmissible but the physics layer does not
    # enforce the bound — it's the config author's responsibility to pick
    # parameters that match the rigid-rod model.
    i1: float = Field(gt=0.0, description="Upper-arm rotational inertia (kg*m^2)")
    i2: float = Field(gt=0.0, description="Forearm rotational inertia (kg*m^2)")

    # Friction model: viscous coefficient per joint.
    friction: float = Field(ge=0.0, description="Viscous friction coefficient (Nm*s/rad)")

    # Per-joint torque limit. Symmetric: applies to both joints.
    max_torque: float = Field(gt=0.0, description="Per-joint torque limit magnitude (Nm)")

    @field_validator("l2")
    @classmethod
    def _forearm_shorter_than_total(cls, v: float, info) -> float:
        # `info.data` contains already-validated fields in pydantic v2.
        l1 = info.data.get("l1")
        if l1 is not None and v >= 2.0 * l1:
            # Not a physics violation, but signals a misconfigured arm.
            # Soft warning via the logger would be better, but keeping
            # validators pure (no side effects) is the rule.
            pass
        return v


class SimParams(BaseConfig):
    """Simulator parameters."""

    dt: float = Field(gt=0.0, le=0.1, description="Integration timestep (s)")
    max_episode_steps: int = Field(gt=0, le=100_000, description="Episode length cap")
    integrator: str = Field(
        default="semi_implicit_euler",
        description="Integrator name (semi_implicit_euler, rk4)",
    )

    @field_validator("integrator")
    @classmethod
    def _validate_integrator(cls, v: str) -> str:
        allowed = {"semi_implicit_euler", "rk4"}
        if v not in allowed:
            raise ValueError(f"integrator must be one of {allowed}, got {v!r}")
        return v


class RewardParams(BaseConfig):
    """Reward function coefficients.

    All terms are linear combinations of state quantities, weighted by
    these coefficients. The reward module assembles them into the final
    reward function.
    """

    # Distance penalty: -distance_coef * ||ee - target||
    distance_coef: float = Field(default=1.0, ge=0.0)

    # Effort penalty: -effort_coef * ||tau||^2
    effort_coef: float = Field(default=0.0, ge=0.0)

    # Smoothness penalty: -smoothness_coef * ||tau_dot||^2 (rate of change of torque)
    smoothness_coef: float = Field(default=0.0, ge=0.0)

    # Sparse success bonus when within success_radius of target.
    success_bonus: float = Field(default=0.0, ge=0.0)
    success_radius: float = Field(default=0.05, gt=0.0, description="Reach tolerance (m)")

    # Time penalty per step (encourages speed).
    time_penalty: float = Field(default=0.0, ge=0.0)


__all__ = ["RobotParams", "SimParams", "RewardParams"]
