"""Cached procedural sprites used by the Snake renderer.

The atlas keeps drawing work out of the training loop.  Body tiles are keyed by
their neighbour connections, so straight sections, corners, and tails join up
without leaking game rules into the sprite implementation.
"""

from __future__ import annotations

from collections.abc import Iterable

import pygame

from .constants import Direction
from snakeGameQDlearning.src.config.settings import (
    BLOCK_SIZE, BLUE1, BLUE2, BLUE3, GREEN, RED, WHITE, YELLOW,
)


_DIRECTION_BITS = {
    Direction.LEFT: 1,
    Direction.RIGHT: 2,
    Direction.UP: 4,
    Direction.DOWN: 8,
}


def _mix(first: tuple[int, int, int], second: tuple[int, int, int], amount: float):
    return tuple(round(a + (b - a) * amount) for a, b in zip(first, second))


class SnakeSpriteAtlas:
    """Small, fully cached sprite atlas with no external asset dependency."""

    def __init__(self, size: int = BLOCK_SIZE):
        self.size = size
        self.heads = {direction: self._head(direction) for direction in Direction}
        self.body_frames = {
            (mask, phase): self._body(mask, phase)
            for mask in range(1, 16)
            for phase in range(3)
        }
        self.food_frames = [self._food(scale, phase) for phase, scale in enumerate(
            (0.78, 0.88, 0.97, 1.0, 0.97, 0.88)
        )]
        self.crash_frame = self._crash()

    def _surface(self):
        return pygame.Surface((self.size, self.size), pygame.SRCALPHA)

    def _head(self, direction: Direction):
        surface = self._surface()
        dx, dy = direction.value
        perpendicular = (-dy, dx)
        center = self.size / 2

        # A short neck makes the head connect to the first body tile.
        if dx:
            neck = pygame.Rect(0 if dx > 0 else self.size // 2, 3,
                               self.size // 2, self.size - 6)
        else:
            neck = pygame.Rect(3, 0 if dy > 0 else self.size // 2,
                               self.size - 6, self.size // 2)
        pygame.draw.rect(surface, BLUE1, neck)
        pygame.draw.rect(surface, BLUE2, (1, 1, self.size - 2, self.size - 2), border_radius=8)

        # Snout and highlight make orientation readable even at training speed.
        snout_center = (round(center + dx * 6), round(center + dy * 6))
        pygame.draw.circle(surface, _mix(BLUE2, WHITE, 0.12), snout_center, 5)
        shine = (round(center - dx * 3 - perpendicular[0] * 3),
                 round(center - dy * 3 - perpendicular[1] * 3))
        pygame.draw.circle(surface, _mix(BLUE2, WHITE, 0.35), shine, 2)

        for side in (-1, 1):
            eye = (
                round(center + dx * 5 + perpendicular[0] * side * 4),
                round(center + dy * 5 + perpendicular[1] * side * 4),
            )
            pygame.draw.circle(surface, WHITE, eye, 2)
            pupil = (eye[0] + dx, eye[1] + dy)
            pygame.draw.circle(surface, BLUE3, pupil, 1)
        return surface

    def _body(self, mask: int, phase: int):
        surface = self._surface()
        outer = _mix(BLUE1, BLUE2, (phase + 1) * 0.06)
        inner = _mix(BLUE2, WHITE, phase * 0.035)
        center = self.size // 2
        half_width = max(5, self.size // 2 - 3)

        # Draw connectors first, then soften the junction with concentric discs.
        if mask & _DIRECTION_BITS[Direction.LEFT]:
            pygame.draw.rect(surface, outer, (0, center - half_width, center + 1, half_width * 2))
        if mask & _DIRECTION_BITS[Direction.RIGHT]:
            pygame.draw.rect(surface, outer, (center, center - half_width,
                                               self.size - center, half_width * 2))
        if mask & _DIRECTION_BITS[Direction.UP]:
            pygame.draw.rect(surface, outer, (center - half_width, 0, half_width * 2, center + 1))
        if mask & _DIRECTION_BITS[Direction.DOWN]:
            pygame.draw.rect(surface, outer, (center - half_width, center,
                                               half_width * 2, self.size - center))
        pygame.draw.circle(surface, outer, (center, center), half_width + 1)
        pygame.draw.circle(surface, inner, (center - 1, center - 1), max(3, half_width - 3))
        pygame.draw.circle(surface, (255, 255, 255, 55), (center - 3, center - 4), 2)
        return surface

    def _food(self, scale: float, phase: int):
        surface = self._surface()
        radius = max(3, round(self.size * 0.37 * scale))
        center = (self.size // 2, self.size // 2 + 1)
        glow = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*RED, 35 + phase * 5), center, min(self.size // 2, radius + 4))
        surface.blit(glow, (0, 0))
        pygame.draw.circle(surface, _mix(RED, YELLOW, 0.12), center, radius)
        pygame.draw.circle(surface, (255, 255, 255, 105),
                           (center[0] - max(1, radius // 3), center[1] - max(1, radius // 3)), 2)
        pygame.draw.line(surface, GREEN, (center[0], center[1] - radius),
                         (center[0] + 3, center[1] - radius - 4), 2)
        pygame.draw.ellipse(surface, GREEN, (center[0] + 1, center[1] - radius - 5, 7, 4))
        return surface

    def _crash(self):
        surface = self._surface()
        center = self.size // 2
        pygame.draw.circle(surface, (*RED, 75), (center, center), center)
        pygame.draw.line(surface, WHITE, (5, 5), (self.size - 5, self.size - 5), 3)
        pygame.draw.line(surface, WHITE, (self.size - 5, 5), (5, self.size - 5), 3)
        return surface

    def head(self, direction: Direction):
        return self.heads[direction]

    def body(self, connections: Iterable[Direction], ticks: int, index: int):
        mask = 0
        for direction in connections:
            mask |= _DIRECTION_BITS[direction]
        # A one-cell snake is not expected, but still has a valid fallback tile.
        mask = mask or _DIRECTION_BITS[Direction.RIGHT]
        phase = (ticks // 150 + index) % 3
        return self.body_frames[(mask, phase)]

    def food(self, ticks: int):
        return self.food_frames[(ticks // 110) % len(self.food_frames)]

    def crash(self):
        return self.crash_frame
