"""Cached circuit drawing and educational telemetry HUD."""

from __future__ import annotations

import math
from pathlib import Path
import random

import pygame

from .circuits import Circuit
from .environment import DrivingEnv
from .math2d import Vec2
from .terrain import TerrainKind, terrain
from .vehicle import DriverControls


TRACK_VIEW_WIDTH = 800
WINDOW_HEIGHT = 700
HUD_WIDTH = 300
WINDOW_WIDTH = TRACK_VIEW_WIDTH + HUD_WIDTH

COLORS = {
    "background": (13, 20, 27),
    "panel": (18, 27, 37),
    "panel_alt": (23, 35, 47),
    "text": (229, 238, 245),
    "muted": (136, 158, 174),
    "cyan": (35, 202, 228),
    "yellow": (246, 191, 38),
    "green": (83, 210, 134),
    "red": (242, 83, 74),
    "barrier": (24, 28, 34),
}


def _font_path() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "fonts" / "arial.ttf"


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    path = _font_path()
    font = pygame.font.Font(str(path) if path.exists() else None, size)
    font.set_bold(bold)
    return font


def _draw_closed_path(
    surface: pygame.Surface,
    points: tuple[Vec2, ...] | list[Vec2],
    color: tuple[int, int, int],
    width: int,
) -> None:
    pixel_points = [(round(point.x), round(point.y)) for point in points]
    pygame.draw.lines(surface, color, True, pixel_points, width)
    radius = max(1, width // 2)
    for point in pixel_points:
        pygame.draw.circle(surface, color, point, radius)


class CircuitRenderer:
    """Renders each static track once; physics remains completely analytic."""

    def __init__(self):
        self._cache: dict[str, pygame.Surface] = {}

    def surface_for(self, circuit: Circuit) -> pygame.Surface:
        cached = self._cache.get(circuit.slug)
        if cached is None:
            cached = self._build(circuit)
            self._cache[circuit.slug] = cached
        return cached

    def _build(self, circuit: Circuit) -> pygame.Surface:
        surface = pygame.Surface((TRACK_VIEW_WIDTH, WINDOW_HEIGHT))
        runoff = terrain(circuit.runoff)
        base = {
            TerrainKind.GRASS: (28, 70, 47),
            TerrainKind.MUD: (53, 47, 36),
            TerrainKind.GRAVEL: (101, 84, 57),
        }.get(circuit.runoff, runoff.color)
        surface.fill(base)

        # Deterministic low-cost ground texture distinguishes each circuit.
        seed = sum((index + 1) * ord(char) for index, char in enumerate(circuit.slug))
        randomizer = random.Random(seed)
        texture = tuple(min(255, channel + 18) for channel in base)
        for _ in range(1250):
            point = (
                randomizer.randrange(TRACK_VIEW_WIDTH),
                randomizer.randrange(WINDOW_HEIGHT),
            )
            pygame.draw.circle(surface, texture, point, randomizer.choice((1, 1, 2)))

        barrier_width = round(circuit.track_width + circuit.runoff_width * 2 + 8)
        runoff_width = round(circuit.track_width + circuit.runoff_width * 2)
        _draw_closed_path(surface, circuit.points, COLORS["barrier"], barrier_width)
        _draw_closed_path(surface, circuit.points, runoff.color, runoff_width)

        samples = 320
        centerline = [
            circuit.point_tangent_at(index / samples)[0] for index in range(samples)
        ]
        _draw_closed_path(
            surface,
            centerline,
            terrain(TerrainKind.ASPHALT).color,
            round(circuit.track_width),
        )
        for index, point in enumerate(centerline):
            progress = index / samples
            kind = circuit.road_kind_at_progress(progress)
            if kind is TerrainKind.ASPHALT:
                continue
            next_point = centerline[(index + 1) % samples]
            pygame.draw.line(
                surface,
                terrain(kind).color,
                (round(point.x), round(point.y)),
                (round(next_point.x), round(next_point.y)),
                round(circuit.track_width),
            )

        # Painted edge markers make steering and slip easy to judge.
        for index in range(0, samples, 5):
            point, tangent = circuit.point_tangent_at(index / samples)
            normal = tangent.perpendicular()
            edge = circuit.track_width * 0.5 - 2
            color = (224, 55, 49) if (index // 5) % 2 else (235, 237, 229)
            for side in (-1, 1):
                location = point + normal * (edge * side)
                pygame.draw.circle(
                    surface, color, (round(location.x), round(location.y)), 3
                )

        # Center dashes and a checkered start line communicate lap direction.
        for index in range(0, samples, 10):
            start, _ = circuit.point_tangent_at(index / samples)
            end, _ = circuit.point_tangent_at((index + 4) / samples)
            pygame.draw.line(
                surface, (206, 211, 211), (start.x, start.y), (end.x, end.y), 2
            )
        start, tangent = circuit.point_tangent_at(0.0)
        normal = tangent.perpendicular()
        for row in range(-4, 4):
            for column in range(2):
                center = start + normal * (row * 5.0 + 2.5) + tangent * (column * 5.0)
                color = (242, 242, 235) if (row + column) % 2 else (25, 28, 31)
                pygame.draw.rect(
                    surface, color, (round(center.x - 3), round(center.y - 3), 6, 6)
                )
        return surface


class TelemetryHUD:
    def __init__(self):
        self.title_font = _font(23, bold=True)
        self.section_font = _font(15, bold=True)
        self.body_font = _font(14)
        self.small_font = _font(12)

    @staticmethod
    def _text(
        target: pygame.Surface,
        font: pygame.font.Font,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = COLORS["text"],
    ) -> None:
        target.blit(font.render(text, True, color), position)

    def _bar(
        self,
        target: pygame.Surface,
        label: str,
        value: float,
        maximum: float,
        y: int,
        color: tuple[int, int, int],
        value_text: str,
    ) -> None:
        x = TRACK_VIEW_WIDTH + 18
        width = HUD_WIDTH - 36
        self._text(target, self.small_font, label, (x, y), COLORS["muted"])
        rendered = self.small_font.render(value_text, True, COLORS["text"])
        target.blit(rendered, (x + width - rendered.get_width(), y))
        y += 17
        pygame.draw.rect(target, (38, 52, 65), (x, y, width, 7), border_radius=4)
        fill = round(width * max(0.0, min(1.0, value / max(1e-9, maximum))))
        if fill:
            pygame.draw.rect(target, color, (x, y, fill, 7), border_radius=4)

    def draw(
        self,
        target: pygame.Surface,
        env: DrivingEnv,
        controls: DriverControls,
        particle_count: int,
    ) -> None:
        panel = pygame.Rect(TRACK_VIEW_WIDTH, 0, HUD_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(target, COLORS["panel"], panel)
        pygame.draw.line(
            target,
            (42, 60, 75),
            (TRACK_VIEW_WIDTH, 0),
            (TRACK_VIEW_WIDTH, WINDOW_HEIGHT),
            2,
        )
        snapshot = env.telemetry()
        components = snapshot["components"]
        capabilities = snapshot["capabilities"]

        x = TRACK_VIEW_WIDTH + 18
        self._text(target, self.title_font, "DRIVING LAB", (x, 16), COLORS["cyan"])
        self._text(target, self.body_font, env.circuit.name, (x, 45))
        description = env.circuit.description
        max_description_width = HUD_WIDTH - 36
        while (
            description
            and self.small_font.size(description + "...")[0] > max_description_width
        ):
            description = description[:-1].rstrip()
        if description != env.circuit.description:
            description += "..."
        self._text(target, self.small_font, description, (x, 65), COLORS["muted"])

        self._bar(
            target,
            "SPEED / MAX",
            float(snapshot["speed"]),
            float(capabilities["max_speed"]),
            92,
            COLORS["cyan"],
            f"{snapshot['speed']:.1f} / {capabilities['max_speed']:.0f} u/s",
        )
        self._bar(
            target,
            "EFFECTIVE GRIP",
            float(snapshot["effective_grip"]),
            1.15,
            130,
            COLORS["green"],
            f"{snapshot['effective_grip']:.2f}",
        )
        self._bar(
            target,
            "TRACK OFFSET",
            abs(float(snapshot["track_offset"])),
            env.circuit.collision_radius,
            168,
            COLORS["yellow"],
            f"{snapshot['track_offset']:+.1f} u",
        )

        pygame.draw.rect(
            target, COLORS["panel_alt"], (x, 208, HUD_WIDTH - 36, 113), border_radius=8
        )
        self._text(
            target, self.section_font, "LIVE PHYSICS", (x + 10, 217), COLORS["yellow"]
        )
        rows = (
            ("Terrain", str(snapshot["terrain"])),
            ("Longitudinal", f"{snapshot['longitudinal_speed']:+.1f} u/s"),
            (
                "Lateral / slip",
                f"{snapshot['lateral_speed']:+.1f} / {snapshot['slip_degrees']:+.1f} deg",
            ),
            ("Acceleration", f"{snapshot['acceleration']:+.1f} u/s2"),
            (
                "Progress / laps",
                f"{snapshot['progress'] * 100:05.1f}% / {snapshot['laps']}",
            ),
        )
        for row, (label, value) in enumerate(rows):
            y = 242 + row * 15
            self._text(target, self.small_font, label, (x + 10, y), COLORS["muted"])
            rendered = self.small_font.render(value, True, COLORS["text"])
            target.blit(
                rendered, (TRACK_VIEW_WIDTH + HUD_WIDTH - 27 - rendered.get_width(), y)
            )

        self._text(
            target, self.section_font, "UPGRADE PARTS  [0-5]", (x, 338), COLORS["cyan"]
        )
        labels = (
            ("1 MOTOR", "motor"),
            ("2 WHEELS", "wheels"),
            ("3 SUSPENSION", "suspension"),
            ("4 GRIP", "grip"),
        )
        for index, (label, key) in enumerate(labels):
            y = 365 + index * 27
            level = int(components[key])
            self._text(target, self.small_font, label, (x, y), COLORS["muted"])
            for block in range(5):
                block_color = COLORS["green"] if block < level else (42, 57, 69)
                pygame.draw.rect(
                    target,
                    block_color,
                    (x + 105 + block * 28, y + 1, 22, 9),
                    border_radius=2,
                )

        self._text(
            target, self.section_font, "DRIVER INPUT", (x, 482), COLORS["yellow"]
        )
        inputs = (
            ("Throttle", controls.throttle, COLORS["green"]),
            ("Steering", controls.steering, COLORS["cyan"]),
            ("Brake", controls.brake, COLORS["red"]),
        )
        for index, (label, value, color) in enumerate(inputs):
            y = 506 + index * 22
            self._text(target, self.small_font, label, (x, y), COLORS["muted"])
            center = x + 178
            pygame.draw.line(target, (48, 63, 75), (x + 92, y + 6), (x + 264, y + 6), 5)
            normalized = (
                value if label in ("Throttle", "Steering") else value * 2.0 - 1.0
            )
            normalized = max(-1.0, min(1.0, normalized))
            marker = round(center + normalized * 82)
            pygame.draw.circle(target, color, (marker, y + 6), 5)

        reward = sum(float(value) for value in snapshot["reward_terms"].values())
        self._text(
            target, self.small_font, f"Reward {reward:+.3f}", (x, 580), COLORS["text"]
        )
        self._text(
            target,
            self.small_font,
            f"Damage {snapshot['damage']:.1f}%",
            (x + 118, 580),
            COLORS["text"],
        )
        self._text(
            target,
            self.small_font,
            f"Particles {particle_count}",
            (x, 598),
            COLORS["muted"],
        )
        self._text(
            target,
            self.small_font,
            f"Collisions {snapshot['collisions']}",
            (x + 118, 598),
            COLORS["muted"],
        )

        pygame.draw.line(
            target, (42, 60, 75), (x, 625), (TRACK_VIEW_WIDTH + HUD_WIDTH - 18, 625), 1
        )
        self._text(
            target,
            self.small_font,
            "WASD/arrows drive   Space brake",
            (x, 637),
            COLORS["muted"],
        )
        self._text(
            target,
            self.small_font,
            "1-4 upgrade   C circuit   R reset",
            (x, 654),
            COLORS["muted"],
        )
        self._text(
            target,
            self.small_font,
            "V sensors   F12 screenshot   Esc quit",
            (x, 671),
            COLORS["muted"],
        )


def draw_sensor_rays(target: pygame.Surface, env: DrivingEnv) -> None:
    state = env.vehicle.state
    angles = (-math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2)
    readings = env.observation()[-5:]
    origin = (round(state.position.x), round(state.position.y))
    for angle, reading in zip(angles, readings):
        distance = float(reading) * 150.0
        endpoint = state.position + Vec2.from_angle(state.heading + angle) * distance
        color = (
            COLORS["green"]
            if reading > 0.55
            else COLORS["yellow"] if reading > 0.25 else COLORS["red"]
        )
        pygame.draw.line(
            target, (*color, 110), origin, (round(endpoint.x), round(endpoint.y)), 1
        )
        pygame.draw.circle(target, color, (round(endpoint.x), round(endpoint.y)), 3)
