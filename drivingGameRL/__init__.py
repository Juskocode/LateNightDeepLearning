"""Deterministic top-down driving laboratory.

The package deliberately keeps vehicle dynamics independent from Pygame so the
same environment can be used for manual play, reproducible experiments, and RL
agents.  Rendering is an optional view over that state.
"""

from .src.environment import DrivingAction, DrivingEnv

__all__ = ["DrivingAction", "DrivingEnv"]
