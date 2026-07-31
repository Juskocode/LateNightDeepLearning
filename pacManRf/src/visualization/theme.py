"""Visual constants for the Pacman reinforcement-learning observatory."""

from __future__ import annotations

from dataclasses import dataclass


Color = tuple[int, int, int]


@dataclass(frozen=True)
class ObservatoryTheme:
    """A small, replaceable colour palette used by :class:`PacmanObservatory`."""

    background: Color = (5, 8, 18)
    header: Color = (9, 14, 30)
    panel: Color = (12, 19, 39)
    panel_alt: Color = (16, 25, 49)
    grid: Color = (31, 45, 75)
    grid_bright: Color = (48, 68, 108)
    text: Color = (235, 242, 255)
    muted: Color = (132, 149, 181)
    blue: Color = (62, 142, 255)
    cyan: Color = (45, 220, 226)
    green: Color = (74, 222, 128)
    yellow: Color = (255, 210, 77)
    orange: Color = (255, 145, 77)
    red: Color = (255, 84, 112)
    purple: Color = (167, 112, 255)
    magenta: Color = (244, 93, 193)


DEFAULT_THEME = ObservatoryTheme()
