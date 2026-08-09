"""Shared type aliases for the domain layer.

These are documentation as much as they are types — using `Radians`
instead of `float` makes it clear at call sites that a function expects
an angle, not an arbitrary number.
"""

from __future__ import annotations

from typing import NewType

import numpy as np

# SI-consistent unit tags. Functions still take plain floats; the newtype
# only documents intent and helps static analyzers.

Radians = NewType("Radians", float)
"""Joint angle in radians."""

RadiansPerSecond = NewType("RadiansPerSecond", float)
"""Joint angular velocity."""

NewtonMeters = NewType("NewtonMeters", float)
"""Joint torque."""

Meters = NewType("Meters", float)
"""Length in meters."""

Kilograms = NewType("Kilograms", float)
"""Mass in kilograms."""

Seconds = NewType("Seconds", float)
"""Time duration."""


def as_array(x) -> np.ndarray:
    """Convert to a float64 ndarray, ensuring consistent dtype."""
    return np.asarray(x, dtype=np.float64)


__all__ = [
    "Radians",
    "RadiansPerSecond",
    "NewtonMeters",
    "Meters",
    "Kilograms",
    "Seconds",
    "as_array",
]
