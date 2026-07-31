"""Public rendering API for the Pacman DQN observatory."""

from .observatory import ObservatoryLayout, ObservatoryTab, PacmanObservatory
from .theme import DEFAULT_THEME, ObservatoryTheme

__all__ = [
    "DEFAULT_THEME",
    "ObservatoryLayout",
    "ObservatoryTab",
    "ObservatoryTheme",
    "PacmanObservatory",
]
