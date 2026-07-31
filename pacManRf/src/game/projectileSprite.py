"""Cached animated sprites for the two ghost projectile abilities."""

from __future__ import annotations

import math
from typing import Tuple

import pygame

from .projectiles import ProjectileKind, ProjectileSpec


class ProjectileSprite(pygame.sprite.Sprite):
    """Render a projectile from cached frames without allocating per frame."""

    def __init__(
        self, spec: ProjectileSpec, tile_size: int, frame_seconds: float = 0.08
    ):
        super().__init__()
        self.spec = spec
        self.tile_size = int(tile_size)
        self.frame_seconds = float(frame_seconds)
        self.frames = tuple(self._make_frame(index) for index in range(8))
        self.impact_frames = tuple(
            self._make_impact_frame(index / 7) for index in range(8)
        )
        self.image = self.frames[0]
        self.rect = self.image.get_rect()

    def _make_frame(self, phase: int) -> pygame.Surface:
        size = max(18, round(self.tile_size * 0.92))
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        center = pygame.Vector2(size / 2, size / 2)
        radius = max(3, round(self.tile_size * self.spec.radius_tiles))
        angle = phase * math.tau / 8

        if self.spec.kind == ProjectileKind.FIREBALL:
            # A rotating four-lobed flame makes direction and motion readable
            # even at the maze's small native tile size.
            for index in range(4):
                theta = angle + index * math.tau / 4
                offset = pygame.Vector2(math.cos(theta), math.sin(theta)) * (radius + 2)
                pygame.draw.circle(
                    surface,
                    (*self.spec.glow_color, 90),
                    (round(center.x + offset.x), round(center.y + offset.y)),
                    max(2, radius - 1),
                )
            pygame.draw.circle(surface, (*self.spec.glow_color, 75), center, radius + 4)
            pygame.draw.circle(surface, self.spec.color, center, radius + 1)
            pygame.draw.circle(
                surface,
                (255, 236, 128),
                (round(center.x - radius * 0.35), round(center.y - radius * 0.35)),
                max(1, radius // 2),
            )
        else:
            pygame.draw.circle(surface, (*self.spec.glow_color, 72), center, radius + 5)
            points = []
            for index in range(8):
                theta = angle + index * math.tau / 8
                reach = radius + (4 if index % 2 == 0 else 1)
                points.append(
                    (
                        round(center.x + math.cos(theta) * reach),
                        round(center.y + math.sin(theta) * reach),
                    )
                )
            pygame.draw.polygon(surface, self.spec.color, points)
            pygame.draw.circle(surface, (230, 253, 255), center, max(2, radius - 1))
            pygame.draw.circle(surface, (93, 180, 255), center, max(1, radius // 2), 1)
        return surface

    def _make_impact_frame(self, progress: float) -> pygame.Surface:
        maximum_radius = round(4 + self.tile_size * 0.55)
        diameter = maximum_radius * 2 + 10
        surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        center = (diameter // 2, diameter // 2)
        radius = round(4 + progress * self.tile_size * 0.55)
        alpha = round(210 * (1.0 - progress))
        if alpha > 0:
            pygame.draw.circle(
                surface,
                (*self.spec.glow_color, alpha // 3),
                center,
                radius + 2,
            )
            pygame.draw.circle(
                surface,
                (*self.spec.color, alpha),
                center,
                radius,
                max(1, round(3 * (1.0 - progress))),
            )
        return surface

    def draw(
        self,
        surface: pygame.Surface,
        center: Tuple[float, float],
        animation_time: float,
    ) -> None:
        index = int(max(0.0, animation_time) / self.frame_seconds) % len(self.frames)
        self.image = self.frames[index]
        self.rect = self.image.get_rect(center=(round(center[0]), round(center[1])))
        surface.blit(self.image, self.rect)

    def draw_impact(
        self,
        surface: pygame.Surface,
        center: Tuple[float, float],
        progress: float,
    ) -> None:
        """Draw a short expanding collision ring for this projectile kind."""

        progress = max(0.0, min(1.0, float(progress)))
        index = min(
            len(self.impact_frames) - 1, int(progress * len(self.impact_frames))
        )
        burst = self.impact_frames[index]
        surface.blit(
            burst,
            burst.get_rect(center=(round(center[0]), round(center[1]))),
        )


__all__ = ("ProjectileSprite",)
