"""Deterministic top-down driving laboratory.

The package deliberately keeps vehicle dynamics independent from Pygame so the
same environment powers manual play, reproducible value learning, population
evolution, and champion races. Rendering is an optional view over that state.
"""

from .src.environment import DrivingAction, DrivingEnv

__all__ = ["DrivingAction", "DrivingEnv"]
