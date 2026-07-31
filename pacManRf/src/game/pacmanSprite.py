"""Pre-rendered, stateful sprites for the Pacman game.

The game rules only select a direction/state. Geometry is cached here so a
normal frame never allocates a new surface, including frightened and eaten-eye
animations.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import pygame

from .constants import Direction, YELLOW


class AnimatedSprite(pygame.sprite.Sprite):
    def __init__(self, frames: Iterable[pygame.Surface], frame_seconds: float = 0.09):
        super().__init__()
        self.frames = list(frames)
        self.frame_seconds = frame_seconds
        self.frame_index = 0
        self.elapsed = 0.0
        self.image = self.frames[0]
        self.rect = self.image.get_rect()

    def update_animation(self, dt: float) -> None:
        self.elapsed += max(0.0, dt)
        if len(self.frames) > 1 and self.frame_seconds > 0:
            self.frame_index = int(self.elapsed / self.frame_seconds) % len(self.frames)


class PacmanSprite(AnimatedSprite):
    """Direction-aware chomping animation plus a classic death sequence."""

    def __init__(self, size: int = 24):
        self.size = size
        self.direction = Direction.RIGHT
        self._direction_frames = {
            direction: [self._make_frame(angle, direction) for angle in (4, 23, 43, 23)]
            for direction in Direction
        }
        self._death_frames = {
            direction: [self._make_death_frame(index / 11, direction) for index in range(12)]
            for direction in Direction
        }
        super().__init__(self._direction_frames[self.direction], frame_seconds=0.085)

    @staticmethod
    def _heading(direction: Direction) -> int:
        return {
            Direction.RIGHT: 0,
            Direction.DOWN: 90,
            Direction.LEFT: 180,
            Direction.UP: 270,
        }[direction]

    def _make_frame(self, mouth_angle: int, direction: Direction) -> pygame.Surface:
        surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        center = self.size / 2
        radius = center - 1
        heading = self._heading(direction)
        points = [(center, center)]
        for degree in range(heading + mouth_angle, heading + 360 - mouth_angle + 1, 4):
            radians = math.radians(degree)
            points.append((center + math.cos(radians) * radius, center + math.sin(radians) * radius))
        pygame.draw.polygon(surface, YELLOW, points)

        # A tiny eye gives directional frames more personality without changing
        # the familiar silhouette.
        eye_offset = {
            Direction.RIGHT: (2, -5),
            Direction.LEFT: (-2, -5),
            Direction.UP: (-5, -2),
            Direction.DOWN: (5, 2),
        }[direction]
        eye = (round(center + eye_offset[0]), round(center + eye_offset[1]))
        pygame.draw.circle(surface, (35, 29, 25), eye, max(1, self.size // 18))
        return surface

    def _make_death_frame(self, progress: float, direction: Direction) -> pygame.Surface:
        if progress >= 1:
            return pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        # The mouth opens until the body becomes a thin arc, then fades away.
        frame = self._make_frame(min(174, round(8 + progress * 178)), direction)
        if progress > 0.72:
            frame.set_alpha(round(255 * (1 - (progress - 0.72) / 0.28)))
        return frame

    def set_direction(self, direction: Direction) -> None:
        self.direction = direction
        self.frames = self._direction_frames[direction]

    def draw(
        self,
        surface: pygame.Surface,
        center: Tuple[float, float],
        dt: float = 0.0,
        death_progress: float | None = None,
    ) -> None:
        self.update_animation(dt)
        if death_progress is None:
            self.image = self._direction_frames[self.direction][self.frame_index]
        else:
            index = min(11, max(0, int(death_progress * 12)))
            self.image = self._death_frames[self.direction][index]
        self.rect = self.image.get_rect(center=(round(center[0]), round(center[1])))
        surface.blit(self.image, self.rect)


class GhostSprite(AnimatedSprite):
    """Ghost body, frightened flash, and returning-eyes sprite states."""

    def __init__(self, color: Tuple[int, int, int], size: int = 24):
        self.color = color
        self.size = size
        self.direction = Direction.LEFT
        self.frightened = False
        self.flashing = False
        self.eaten = False
        self._cache: dict[tuple[Direction, str, bool], list[pygame.Surface]] = {}
        frames = self._frames_for(self.direction, "normal")
        super().__init__(frames, frame_seconds=0.14)

    def _frames_for(self, direction: Direction, state: str, flash: bool = False) -> list[pygame.Surface]:
        key = (direction, state, flash)
        if key not in self._cache:
            self._cache[key] = [self._make_frame(phase, direction, state, flash) for phase in (0, 1)]
        return self._cache[key]

    def _draw_eyes(self, surface: pygame.Surface, direction: Direction) -> None:
        dx, dy = direction.vector
        eye_y = round(self.size * 0.42)
        radius = max(3, round(self.size * 0.17))
        pupil_radius = max(1, round(self.size * 0.08))
        for eye_x in (round(self.size * 0.34), round(self.size * 0.68)):
            pygame.draw.circle(surface, (249, 249, 255), (eye_x, eye_y), radius)
            pupil = (eye_x + dx * max(1, radius // 2), eye_y + dy * max(1, radius // 2))
            pygame.draw.circle(surface, (42, 72, 188), pupil, pupil_radius)

    def _make_frame(
        self,
        phase: int,
        direction: Direction,
        state: str,
        flash: bool,
    ) -> pygame.Surface:
        surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        if state == "eaten":
            self._draw_eyes(surface, direction)
            return surface

        if state == "frightened":
            body = (238, 241, 255) if flash else (38, 74, 213)
        else:
            body = self.color

        center_x = self.size // 2
        head_y = self.size // 2
        radius = self.size // 2 - 1
        pygame.draw.circle(surface, body, (center_x, head_y), radius)
        pygame.draw.rect(surface, body, (1, head_y, self.size - 2, self.size - head_y - 3))

        # Four independently animated scallops make movement readable at the
        # small native resolution.
        foot_width = self.size / 4
        for index in range(4):
            foot_x = round((index + 0.5) * foot_width)
            foot_y = self.size - 3 - ((index + phase) % 2) * 2
            pygame.draw.circle(surface, body, (foot_x, foot_y), max(2, round(foot_width / 2)))

        if state == "frightened":
            face = (221, 42, 69) if flash else (247, 244, 225)
            eye_y = round(self.size * 0.43)
            for eye_x in (round(self.size * 0.34), round(self.size * 0.68)):
                pygame.draw.circle(surface, face, (eye_x, eye_y), max(1, self.size // 12))
            zigzag = []
            mouth_y = round(self.size * 0.66)
            for index in range(6):
                zigzag.append((round(self.size * 0.25 + index * self.size * 0.1), mouth_y + (index % 2) * 3))
            pygame.draw.lines(surface, face, False, zigzag, max(1, self.size // 12))
        else:
            self._draw_eyes(surface, direction)
            highlight = tuple(min(255, channel + 28) for channel in body)
            pygame.draw.circle(surface, highlight, (round(self.size * 0.32), round(self.size * 0.2)), 2)
        return surface

    def set_state(
        self,
        direction: Direction,
        frightened: bool,
        flashing: bool = False,
        eaten: bool = False,
    ) -> None:
        self.direction = direction
        self.frightened = frightened
        self.flashing = flashing
        self.eaten = eaten
        state = "eaten" if eaten else "frightened" if frightened else "normal"
        self.frames = self._frames_for(direction, state)

    def draw(self, surface: pygame.Surface, center: Tuple[float, float], dt: float = 0.0) -> None:
        self.update_animation(dt)
        state = "eaten" if self.eaten else "frightened" if self.frightened else "normal"
        flash_on = self.flashing and int(self.elapsed * 7) % 2 == 1
        frames = self._frames_for(self.direction, state, flash_on)
        self.image = frames[self.frame_index]
        self.rect = self.image.get_rect(center=(round(center[0]), round(center[1])))
        surface.blit(self.image, self.rect)
