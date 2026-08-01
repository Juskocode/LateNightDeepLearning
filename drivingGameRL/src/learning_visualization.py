"""Deterministic, telemetry-driven views for the Driving Lab learner.

The renderer deliberately owns no training state.  A training loop can pass a
plain mapping every frame, which keeps the visualization useful for DQN,
Double-DQN, and population-based variants without coupling it to one trainer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import pygame

from .environment import DrivingEnv
from .rendering import CircuitRenderer, TRACK_VIEW_WIDTH, WINDOW_HEIGHT as TRACK_HEIGHT


LEARNING_WINDOW_WIDTH = 1_400
LEARNING_WINDOW_HEIGHT = 760
LEARNING_WINDOW_SIZE = (LEARNING_WINDOW_WIDTH, LEARNING_WINDOW_HEIGHT)

ACTION_LABELS = ("COAST", "THROTTLE", "BRAKE", "LEFT", "RIGHT")

SENSOR_RAY_COUNT = 5
SENSOR_MAX_DISTANCE = 150.0
SENSOR_ANGLE_OFFSETS = (
    -math.pi / 2,
    -math.pi / 4,
    0.0,
    math.pi / 4,
    math.pi / 2,
)
MAX_POPULATION_CARS = 12

# Deliberately fixed rather than generated from fitness or list order.  A member
# keeps the same identity color while rankings and positions change.
POPULATION_CAR_COLORS = (
    (255, 105, 135),
    (111, 225, 151),
    (255, 158, 75),
    (155, 117, 255),
    (90, 176, 255),
    (255, 108, 216),
    (109, 231, 213),
    (250, 219, 86),
    (211, 128, 255),
    (255, 176, 188),
    (119, 202, 255),
    (174, 229, 103),
)

COLORS = {
    "background": (7, 13, 22),
    "panel": (14, 24, 37),
    "panel_alt": (19, 32, 48),
    "panel_high": (24, 41, 60),
    "edge": (44, 65, 84),
    "grid": (33, 50, 66),
    "text": (231, 241, 248),
    "muted": (130, 153, 171),
    "cyan": (42, 214, 238),
    "cyan_dim": (37, 132, 154),
    "blue": (74, 132, 255),
    "green": (75, 220, 137),
    "yellow": (250, 194, 55),
    "orange": (255, 132, 55),
    "red": (246, 80, 91),
    "magenta": (231, 81, 194),
    "human": (250, 197, 49),
    "champion": (46, 220, 239),
}


def _font_path() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "fonts" / "arial.ttf"


def _finite(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
    return default


def _integer(value: object, default: int = 0) -> int:
    numeric = _finite(value, float(default))
    return int(numeric)


def _sequence(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _flag(value: object, default: bool = False) -> bool:
    """Interpret a telemetry toggle without treating ``"false"`` as true."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and float(value) != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled", ""}:
            return False
    return default


def _point(value: object) -> tuple[float, float] | None:
    """Return a finite world-space point from mappings, vectors, or pairs."""

    if isinstance(value, Mapping):
        raw_x, raw_y = value.get("x"), value.get("y")
    elif hasattr(value, "x") and hasattr(value, "y"):
        raw_x, raw_y = getattr(value, "x"), getattr(value, "y")
    else:
        values = _sequence(value)
        if len(values) < 2:
            return None
        raw_x, raw_y = values[:2]
    x = _finite(raw_x, math.nan)
    y = _finite(raw_y, math.nan)
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _compact_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}K"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


class DrivingLearningVisualization:
    """Draw a complete learning observatory onto one reusable Pygame surface."""

    TABS = ("OVERVIEW", "NETWORK", "MEMORY")
    WIDTH = LEARNING_WINDOW_WIDTH
    HEIGHT = LEARNING_WINDOW_HEIGHT

    def __init__(
        self,
        env: DrivingEnv,
        telemetry: Mapping[str, Any] | None = None,
        *,
        surface: pygame.Surface | None = None,
    ):
        pygame.font.init()
        self.env = env
        self.telemetry: Mapping[str, Any] = _mapping(telemetry)
        self.surface = surface or pygame.Surface(LEARNING_WINDOW_SIZE)
        if self.surface.get_size() != LEARNING_WINDOW_SIZE:
            raise ValueError(f"learning surface must be {LEARNING_WINDOW_SIZE}")
        self.active_tab = self.TABS[0]
        self.return_to_training_requested = False
        self._fonts: dict[tuple[int, bool], pygame.font.Font] = {}
        self._track_renderer = CircuitRenderer()
        self._scaled_tracks: dict[tuple[str, int, int], pygame.Surface] = {}
        self._car_bodies: dict[tuple[int, int, int], pygame.Surface] = {}
        self._car_rotations: dict[tuple[tuple[int, int, int], int], pygame.Surface] = {}
        self.tab_rects: dict[str, pygame.Rect] = {
            name: pygame.Rect(796 + index * 184, 18, 170, 42)
            for index, name in enumerate(self.TABS)
        }

    def _font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        cached = self._fonts.get(key)
        if cached is None:
            path = _font_path()
            cached = pygame.font.Font(str(path) if path.exists() else None, size)
            cached.set_bold(bold)
            self._fonts[key] = cached
        return cached

    def _text(
        self,
        text: object,
        position: tuple[int, int],
        *,
        size: int = 14,
        color: tuple[int, int, int] = COLORS["text"],
        bold: bool = False,
        target: pygame.Surface | None = None,
    ) -> pygame.Rect:
        destination = target or self.surface
        rendered = self._font(size, bold).render(str(text), True, color)
        return destination.blit(rendered, position)

    def _right_text(
        self,
        text: object,
        right: int,
        y: int,
        *,
        size: int = 14,
        color: tuple[int, int, int] = COLORS["text"],
        bold: bool = False,
    ) -> pygame.Rect:
        rendered = self._font(size, bold).render(str(text), True, color)
        return self.surface.blit(rendered, (right - rendered.get_width(), y))

    def _fit_text(self, text: object, width: int, size: int, bold: bool = False) -> str:
        result = str(text)
        font = self._font(size, bold)
        if font.size(result)[0] <= width:
            return result
        suffix = "..."
        while result and font.size(result + suffix)[0] > width:
            result = result[:-1]
        return result.rstrip() + suffix

    def _panel(
        self,
        rect: pygame.Rect,
        title: str | None = None,
        *,
        accent: tuple[int, int, int] = COLORS["cyan"],
        high: bool = False,
    ) -> None:
        pygame.draw.rect(
            self.surface,
            COLORS["panel_high"] if high else COLORS["panel"],
            rect,
            border_radius=11,
        )
        pygame.draw.rect(self.surface, COLORS["edge"], rect, 1, border_radius=11)
        if title:
            pygame.draw.rect(
                self.surface,
                accent,
                (rect.x, rect.y, 4, min(38, rect.height)),
                border_radius=2,
            )
            self._text(
                title,
                (rect.x + 14, rect.y + 10),
                size=13,
                color=COLORS["muted"],
                bold=True,
            )

    def _combined_telemetry(
        self, env: DrivingEnv, supplied: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        try:
            combined: dict[str, Any] = dict(env.telemetry())
        except (AttributeError, TypeError, ValueError):
            combined = {}
        source = _mapping(self.telemetry if supplied is None else supplied)
        # Common containers are flattened for algorithms that keep environment,
        # learner, and evolutionary statistics in separate dictionaries.
        for name in ("environment", "env", "agent", "learning", "metrics"):
            combined.update(_mapping(source.get(name)))
        population = source.get("population")
        if isinstance(population, Mapping):
            combined.update(population)
        combined.update(source)
        return combined

    def set_tab(self, tab: str | int) -> str:
        if isinstance(tab, int) and not isinstance(tab, bool):
            self.active_tab = self.TABS[tab % len(self.TABS)]
        else:
            name = str(tab).upper()
            if name not in self.TABS:
                raise ValueError(f"unknown learning tab: {tab!r}")
            self.active_tab = name
        return self.active_tab

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle tab navigation and report whether the event was consumed."""

        if event.type == pygame.KEYDOWN:
            direct = {
                pygame.K_1: "OVERVIEW",
                pygame.K_2: "NETWORK",
                pygame.K_3: "MEMORY",
            }
            if event.key in direct:
                self.set_tab(direct[event.key])
                return True
            if event.key in (pygame.K_TAB, pygame.K_RIGHT):
                self.set_tab(self.TABS.index(self.active_tab) + 1)
                return True
            if event.key == pygame.K_LEFT:
                self.set_tab(self.TABS.index(self.active_tab) - 1)
                return True
            if event.key == pygame.K_p:
                self.return_to_training_requested = True
                return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for name, rect in self.tab_rects.items():
                if rect.collidepoint(event.pos):
                    self.set_tab(name)
                    return True
        return False

    def handle_events(self, events: Sequence[pygame.event.Event] | None = None) -> bool:
        consumed = False
        for event in pygame.event.get() if events is None else events:
            consumed = self.handle_event(event) or consumed
        return consumed

    def _header(self, data: Mapping[str, Any], *, race: bool = False) -> None:
        pygame.draw.rect(self.surface, COLORS["panel"], (0, 0, self.WIDTH, 76))
        pygame.draw.line(self.surface, COLORS["edge"], (0, 75), (self.WIDTH, 75), 1)
        title = (
            "HUMAN VS GENERATION CHAMPION" if race else "EVOLUTIONARY DQN OBSERVATORY"
        )
        self._text(title, (22, 14), size=22, color=COLORS["cyan"], bold=True)
        subtitle = (
            "Same circuit · deterministic simulation · one fair race"
            if race
            else "Live policy, population, replay memory, and neural activity"
        )
        self._text(subtitle, (23, 44), size=12, color=COLORS["muted"])
        if race:
            ray_state = "ON" if _flag(data.get("show_sensor_rays")) else "OFF"
            self._right_text(
                f"P  TRAINING   ·   V RAYS {ray_state}",
                1_378,
                46,
                size=10,
                color=COLORS["muted"],
                bold=True,
            )
            return
        speed = str(data.get("training_speed_label", "16x")).upper()
        ray_state = "ON" if _flag(data.get("show_sensor_rays")) else "OFF"
        cars_state = "ON" if _flag(data.get("show_population_cars")) else "OFF"
        self._right_text(
            (
                f"P RACE  ·  V RAYS {ray_state}  ·  "
                f"M GEN CARS {cars_state}  ·  [ ] {speed}"
            ),
            778,
            46,
            size=9,
            color=COLORS["muted"],
            bold=True,
        )
        for name, rect in self.tab_rects.items():
            selected = name == self.active_tab
            pygame.draw.rect(
                self.surface,
                COLORS["cyan"] if selected else COLORS["panel_alt"],
                rect,
                border_radius=8,
            )
            pygame.draw.rect(
                self.surface,
                COLORS["cyan_dim"] if selected else COLORS["edge"],
                rect,
                1,
                border_radius=8,
            )
            label = f"{self.TABS.index(name) + 1}  {name}"
            label_surface = self._font(13, True).render(
                label, True, COLORS["background"] if selected else COLORS["text"]
            )
            self.surface.blit(label_surface, label_surface.get_rect(center=rect.center))
        phase = str(
            data.get("phase", data.get("status", data.get("event", "TRAINING")))
        ).upper()
        self._right_text(phase, 1_380, 63, size=10, color=COLORS["green"], bold=True)

    def _track_surface(self, env: DrivingEnv, size: tuple[int, int]) -> pygame.Surface:
        circuit = env.circuit
        key = (circuit.slug, size[0], size[1])
        cached = self._scaled_tracks.get(key)
        if cached is None:
            native = self._track_renderer.surface_for(circuit)
            cached = pygame.transform.smoothscale(native, size)
            self._scaled_tracks[key] = cached
        return cached

    def _car_body(self, color: tuple[int, int, int]) -> pygame.Surface:
        body = self._car_bodies.get(color)
        if body is None:
            body = pygame.Surface((24, 40), pygame.SRCALPHA)
            dark = tuple(max(0, channel - 105) for channel in color)
            pygame.draw.rect(body, (9, 13, 18), (1, 6, 4, 11), border_radius=2)
            pygame.draw.rect(body, (9, 13, 18), (19, 6, 4, 11), border_radius=2)
            pygame.draw.rect(body, (9, 13, 18), (1, 27, 4, 10), border_radius=2)
            pygame.draw.rect(body, (9, 13, 18), (19, 27, 4, 10), border_radius=2)
            pygame.draw.polygon(
                body,
                color,
                (
                    (7, 1),
                    (17, 1),
                    (22, 9),
                    (20, 35),
                    (16, 39),
                    (8, 39),
                    (4, 35),
                    (2, 9),
                ),
            )
            pygame.draw.polygon(body, dark, ((6, 11), (18, 11), (17, 23), (7, 23)))
            pygame.draw.line(body, (242, 249, 250), (7, 5), (17, 5), 2)
            self._car_bodies[color] = body
        return body

    def _draw_car(
        self,
        center: tuple[int, int],
        heading: float,
        color: tuple[int, int, int],
        *,
        label: str | None = None,
    ) -> None:
        angle = round((-math.degrees(heading) - 90.0) / 3.0) * 3
        key = (color, angle)
        image = self._car_rotations.get(key)
        if image is None:
            image = pygame.transform.rotozoom(self._car_body(color), angle, 1.0)
            self._car_rotations[key] = image
        pygame.draw.circle(self.surface, (*color,), center, 18, 2)
        self.surface.blit(image, image.get_rect(center=center))
        if label:
            text = self._font(10, True).render(label, True, color)
            box = text.get_rect(midbottom=(center[0], center[1] - 22))
            backdrop = box.inflate(8, 4)
            pygame.draw.rect(
                self.surface, COLORS["background"], backdrop, border_radius=4
            )
            self.surface.blit(text, box)

    @staticmethod
    def _pose(env: DrivingEnv, explicit: object = None) -> tuple[float, float, float]:
        if explicit is None:
            state = env.vehicle.state
            return state.position.x, state.position.y, state.heading
        if isinstance(explicit, Mapping):
            position = explicit.get("position", explicit.get("pos", (0.0, 0.0)))
            heading = explicit.get("heading", explicit.get("angle", 0.0))
        elif hasattr(explicit, "position"):
            position = explicit.position
            heading = getattr(explicit, "heading", 0.0)
        else:
            values = _sequence(explicit)
            if len(values) >= 3:
                return _finite(values[0]), _finite(values[1]), _finite(values[2])
            position, heading = (0.0, 0.0), 0.0
        if hasattr(position, "x") and hasattr(position, "y"):
            return _finite(position.x), _finite(position.y), _finite(heading)
        values = _sequence(position)
        return (
            _finite(values[0]) if values else 0.0,
            _finite(values[1]) if len(values) > 1 else 0.0,
            _finite(heading),
        )

    @staticmethod
    def _field(value: object, name: str, default: object = None) -> object:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    def _coerce_rays(self, source: object) -> list[dict[str, Any]]:
        """Normalize SensorRay objects and serialized ray mappings for drawing."""

        if isinstance(source, Mapping):
            for key in ("rays", "sensor_rays", "vision_rays"):
                if key in source:
                    source = source.get(key)
                    break
            else:
                source = [source] if "endpoint" in source else []
        elif not isinstance(source, Sequence) and source is not None:
            nested = getattr(source, "rays", getattr(source, "sensor_rays", None))
            if nested is not None:
                source = nested

        rays: list[dict[str, Any]] = []
        for candidate in _sequence(source):
            origin = _point(self._field(candidate, "origin"))
            endpoint = _point(
                self._field(
                    candidate,
                    "endpoint",
                    self._field(candidate, "end"),
                )
            )
            if origin is None or endpoint is None:
                continue
            normalized = _finite(
                self._field(candidate, "normalized_distance"), math.nan
            )
            distance = _finite(self._field(candidate, "distance"), math.nan)
            maximum = _finite(
                self._field(candidate, "max_distance"), SENSOR_MAX_DISTANCE
            )
            if not math.isfinite(normalized):
                normalized = (
                    distance / maximum
                    if math.isfinite(distance) and maximum > 0.0
                    else 1.0
                )
            rays.append(
                {
                    "origin": origin,
                    "endpoint": endpoint,
                    "distance": distance,
                    "normalized_distance": max(0.0, min(1.0, normalized)),
                    "hit": _flag(self._field(candidate, "hit"), normalized < 1.0),
                }
            )
        return rays

    def _environment_rays(
        self, env: DrivingEnv, explicit_pose: object = None
    ) -> list[dict[str, Any]]:
        """Read the environment's exact ray snapshots, with a legacy fallback."""

        environment_pose = self._pose(env)
        requested_pose = self._pose(env, explicit_pose)
        pose_matches_environment = explicit_pose is None or all(
            math.isclose(actual, requested, abs_tol=1e-7)
            for actual, requested in zip(environment_pose, requested_pose)
        )
        sensor_api = getattr(env, "sensor_rays", None)
        if sensor_api is not None and pose_matches_environment:
            try:
                snapshots = sensor_api() if callable(sensor_api) else sensor_api
                rays = self._coerce_rays(snapshots)
                if len(rays) >= SENSOR_RAY_COUNT:
                    return rays[:SENSOR_RAY_COUNT]
            except (AttributeError, TypeError, ValueError):
                pass

        # Older environments expose only the normalized observation values.
        # Rebuilding endpoints from the same fixed angles remains read-only and
        # keeps this dashboard compatible while SensorRay rolls out.
        try:
            readings = [_finite(value) for value in env.observation()[-5:]]
        except (AttributeError, TypeError, ValueError):
            readings = []
        if len(readings) != SENSOR_RAY_COUNT:
            return []
        x, y, heading = requested_pose
        result: list[dict[str, Any]] = []
        for angle, reading in zip(SENSOR_ANGLE_OFFSETS, readings):
            normalized = max(0.0, min(1.0, reading))
            distance = normalized * SENSOR_MAX_DISTANCE
            endpoint = (
                x + math.cos(heading + angle) * distance,
                y + math.sin(heading + angle) * distance,
            )
            result.append(
                {
                    "origin": (x, y),
                    "endpoint": endpoint,
                    "distance": distance,
                    "normalized_distance": normalized,
                    "hit": normalized < 1.0,
                }
            )
        return result

    @staticmethod
    def _stable_member_key(value: object, fallback: int) -> int:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return int(numeric)
        text = str(value)
        if not text or text == "None":
            return fallback
        return sum((index + 1) * ord(character) for index, character in enumerate(text))

    def _population_rollouts(
        self, data: Mapping[str, Any]
    ) -> list[tuple[Mapping[str, Any], tuple[int, int, int], str]]:
        raw = _sequence(data.get("population_rollouts"))
        if not raw:
            return []

        target_raw = data.get(
            "population_rollout_generation",
            data.get("generation", data.get("generation_index")),
        )
        target_generation: int | None = None
        if isinstance(target_raw, (int, float)) and not isinstance(target_raw, bool):
            if math.isfinite(float(target_raw)):
                target_generation = int(target_raw)
        if target_generation is None:
            for candidate in raw:
                generation = _mapping(candidate).get("generation")
                if isinstance(generation, (int, float)) and not isinstance(
                    generation, bool
                ):
                    if math.isfinite(float(generation)):
                        target_generation = int(generation)
                        break

        prepared: list[
            tuple[int, int, Mapping[str, Any], tuple[int, int, int], str]
        ] = []
        for ordinal, candidate in enumerate(raw):
            rollout = _mapping(candidate)
            if (
                not rollout
                or _point(rollout.get("position", rollout.get("pos"))) is None
            ):
                continue
            generation = rollout.get("generation")
            if (
                target_generation is not None
                and isinstance(generation, (int, float))
                and not isinstance(generation, bool)
                and math.isfinite(float(generation))
                and int(generation) != target_generation
            ):
                continue
            member_value = rollout.get(
                "member_id", rollout.get("index", rollout.get("member", ordinal))
            )
            member_key = self._stable_member_key(member_value, ordinal)
            color = POPULATION_CAR_COLORS[member_key % len(POPULATION_CAR_COLORS)]
            label_value = rollout.get("index", member_value)
            label_number = self._stable_member_key(label_value, ordinal)
            label = f"M{label_number + 1:02d}"[-5:]
            prepared.append((label_number, ordinal, rollout, color, label))
        prepared.sort(key=lambda item: (item[0], item[1]))
        return [
            (rollout, color, label)
            for _, _, rollout, color, label in prepared[:MAX_POPULATION_CARS]
        ]

    @staticmethod
    def _ray_color(ray: Mapping[str, Any]) -> tuple[int, int, int]:
        reading = _finite(ray.get("normalized_distance"), 1.0)
        if reading > 0.55:
            return COLORS["green"]
        if reading > 0.25:
            return COLORS["yellow"]
        return COLORS["red"]

    def _draw_ray_set(
        self,
        layer: pygame.Surface,
        viewport: pygame.Rect,
        rays: Sequence[Mapping[str, Any]],
        *,
        color: tuple[int, int, int] | None = None,
        alpha: int = 150,
        width: int = 1,
    ) -> None:
        scale_x = viewport.width / TRACK_VIEW_WIDTH
        scale_y = viewport.height / TRACK_HEIGHT

        def local(point: tuple[float, float]) -> tuple[int, int]:
            # Bounding extreme malformed-but-finite inputs avoids passing huge
            # integers into SDL while still letting normal off-track rays clip.
            x = max(-TRACK_VIEW_WIDTH * 4, min(TRACK_VIEW_WIDTH * 5, point[0]))
            y = max(-TRACK_HEIGHT * 4, min(TRACK_HEIGHT * 5, point[1]))
            return round(x * scale_x), round(y * scale_y)

        for ray in list(rays)[:SENSOR_RAY_COUNT]:
            origin = _point(ray.get("origin"))
            endpoint = _point(ray.get("endpoint"))
            if origin is None or endpoint is None:
                continue
            ray_color = color or self._ray_color(ray)
            start, end = local(origin), local(endpoint)
            pygame.draw.line(layer, (*ray_color, alpha), start, end, width)
            if (
                -4 <= end[0] <= viewport.width + 4
                and -4 <= end[1] <= viewport.height + 4
            ):
                pygame.draw.circle(layer, (*ray_color, min(255, alpha + 55)), end, 3)

    def _draw_track(
        self,
        rect: pygame.Rect,
        env: DrivingEnv,
        *,
        title: str = "CURRENT POLICY ON TRACK",
        cars: Sequence[tuple[object, tuple[int, int, int], str]] = (),
        telemetry: Mapping[str, Any] | None = None,
        include_population: bool = True,
        ray_sources: Sequence[
            tuple[object, DrivingEnv, object, tuple[int, int, int]]
        ] = (),
    ) -> None:
        self._panel(rect, title, accent=COLORS["green"])
        viewport = pygame.Rect(
            rect.x + 10, rect.y + 38, rect.width - 20, rect.height - 48
        )
        self.surface.blit(self._track_surface(env, viewport.size), viewport)
        data = _mapping(telemetry)
        population = (
            self._population_rollouts(data)
            if include_population and _flag(data.get("show_population_cars"))
            else []
        )
        if _flag(data.get("show_sensor_rays")):
            ray_layer = pygame.Surface(viewport.size, pygame.SRCALPHA)
            if ray_sources:
                for source, source_env, pose, color in ray_sources:
                    rays = self._coerce_rays(source)
                    if len(rays) < SENSOR_RAY_COUNT:
                        rays = self._environment_rays(source_env, pose)
                    self._draw_ray_set(
                        ray_layer, viewport, rays, color=color, alpha=145, width=2
                    )
            else:
                self._draw_ray_set(
                    ray_layer,
                    viewport,
                    self._environment_rays(env),
                    alpha=165,
                    width=2,
                )
            for rollout, color, _ in population:
                self._draw_ray_set(
                    ray_layer,
                    viewport,
                    self._coerce_rays(rollout.get("rays")),
                    color=color,
                    alpha=58,
                    width=1,
                )
            self.surface.blit(ray_layer, viewport.topleft)
        if not cars:
            cars = ((None, COLORS["cyan"], "POLICY"),)
        all_cars = [*population, *cars]
        scale_x = viewport.width / TRACK_VIEW_WIDTH
        scale_y = viewport.height / TRACK_HEIGHT
        for explicit, color, label in all_cars:
            x, y, heading = self._pose(env, explicit)
            center = (
                round(viewport.x + x * scale_x),
                round(viewport.y + y * scale_y),
            )
            self._draw_car(center, heading, color, label=label)
        try:
            snapshot = env.telemetry()
        except (AttributeError, TypeError, ValueError):
            snapshot = {}
        terrain = str(snapshot.get("terrain", "unknown")).replace("_", " ").upper()
        footer = (
            f"{env.circuit.name.upper()}   ·   {terrain}   ·   "
            f"{_finite(snapshot.get('speed')):05.1f} U/S   ·   "
            f"LAP {_integer(snapshot.get('laps')) + 1}"
        )
        label_surface = self._font(11, True).render(footer, True, COLORS["text"])
        label_rect = label_surface.get_rect(
            bottomleft=(viewport.x + 10, viewport.bottom - 9)
        )
        pygame.draw.rect(
            self.surface,
            (*COLORS["background"],),
            label_rect.inflate(14, 8),
            border_radius=5,
        )
        self.surface.blit(label_surface, label_rect)

    def _metric_card(
        self,
        rect: pygame.Rect,
        label: str,
        value: str,
        detail: str = "",
        *,
        color: tuple[int, int, int] = COLORS["cyan"],
    ) -> None:
        self._panel(rect, high=True)
        self._text(
            label, (rect.x + 12, rect.y + 9), size=10, color=COLORS["muted"], bold=True
        )
        self._text(
            self._fit_text(value, rect.width - 24, 19, True),
            (rect.x + 12, rect.y + 25),
            size=19,
            color=color,
            bold=True,
        )
        if detail:
            self._text(
                self._fit_text(detail, rect.width - 24, 10),
                (rect.x + 12, rect.bottom - 18),
                size=10,
                color=COLORS["muted"],
            )

    def _population_rows(self, data: Mapping[str, Any]) -> list[tuple[int, float]]:
        candidates: object = data.get("ranking")
        if not _sequence(candidates):
            candidates = data.get("population_fitness", data.get("fitnesses"))
        if not _sequence(candidates) and not isinstance(
            data.get("population"), Mapping
        ):
            candidates = data.get("population")
        rows: list[tuple[int, float]] = []
        for index, item in enumerate(_sequence(candidates)):
            if isinstance(item, Mapping):
                member = _integer(
                    item.get("member", item.get("index", item.get("id", index))), index
                )
                raw_fitness = item.get("fitness", item.get("score"))
                if raw_fitness is None:
                    continue
                fitness = _finite(raw_fitness)
            else:
                member, fitness = index, _finite(item)
            rows.append((member, fitness))
        if not rows:
            best = _finite(data.get("best_fitness", data.get("fitness")))
            mean = _finite(data.get("mean_fitness", best))
            rows = [(0, best), (1, mean)]
        return sorted(rows, key=lambda row: (-row[1], row[0]))

    def _population_panel(self, rect: pygame.Rect, data: Mapping[str, Any]) -> None:
        self._panel(rect, "POPULATION RANKING", accent=COLORS["yellow"])
        rows = self._population_rows(data)[:7]
        current = _integer(data.get("member", data.get("member_index", -1)), -1)
        values = [value for _, value in rows]
        low, high = min(values, default=0.0), max(values, default=1.0)
        span = max(1e-9, high - low)
        y = rect.y + 42
        for rank, (member, fitness) in enumerate(rows, 1):
            selected = member == current
            label_color = (
                COLORS["yellow"]
                if rank == 1
                else (COLORS["cyan"] if selected else COLORS["muted"])
            )
            self._text(
                f"#{rank:02d}", (rect.x + 13, y), size=10, color=label_color, bold=True
            )
            self._text(
                f"M{member + 1:03d}", (rect.x + 47, y), size=10, color=COLORS["text"]
            )
            self._right_text(
                _compact_number(fitness), rect.right - 12, y, size=10, color=label_color
            )
            bar = pygame.Rect(rect.x + 92, y + 4, rect.width - 170, 7)
            pygame.draw.rect(self.surface, COLORS["grid"], bar, border_radius=4)
            fraction = 1.0 if len(rows) == 1 else 0.12 + 0.88 * (fitness - low) / span
            fill = bar.copy()
            fill.width = max(2, round(bar.width * max(0.0, min(1.0, fraction))))
            pygame.draw.rect(self.surface, label_color, fill, border_radius=4)
            y += 27

    def _history(self, data: Mapping[str, Any]) -> tuple[list[float], list[float]]:
        best = [_finite(value) for value in _sequence(data.get("best_history"))]
        mean = [_finite(value) for value in _sequence(data.get("mean_history"))]
        generation_history = _sequence(
            data.get("generation_history", data.get("history"))
        )
        if generation_history and not best:
            best = [
                _finite(_mapping(item).get("best", _mapping(item).get("best_fitness")))
                for item in generation_history
            ]
        if generation_history and not mean:
            mean = [
                _finite(_mapping(item).get("mean", _mapping(item).get("mean_fitness")))
                for item in generation_history
            ]
        return best[-80:], mean[-80:]

    def _history_panel(self, rect: pygame.Rect, data: Mapping[str, Any]) -> None:
        self._panel(rect, "FITNESS BY GENERATION", accent=COLORS["green"])
        best, mean = self._history(data)
        chart = pygame.Rect(rect.x + 34, rect.y + 49, rect.width - 48, rect.height - 77)
        pygame.draw.line(
            self.surface, COLORS["grid"], chart.bottomleft, chart.bottomright
        )
        pygame.draw.line(self.surface, COLORS["grid"], chart.topleft, chart.bottomleft)
        for fraction in (0.25, 0.5, 0.75):
            y = round(chart.bottom - chart.height * fraction)
            pygame.draw.line(
                self.surface, COLORS["grid"], (chart.x, y), (chart.right, y), 1
            )
        all_values = best + mean
        if not all_values:
            self._text(
                "No completed generation yet",
                (chart.x + 8, chart.centery - 7),
                size=11,
                color=COLORS["muted"],
            )
            return
        low, high = min(all_values), max(all_values)
        if math.isclose(low, high):
            low, high = low - 1.0, high + 1.0

        def points(values: list[float]) -> list[tuple[int, int]]:
            if len(values) == 1:
                xs = [chart.centerx]
            else:
                xs = [
                    round(chart.x + index * chart.width / (len(values) - 1))
                    for index in range(len(values))
                ]
            return [
                (x, round(chart.bottom - (value - low) / (high - low) * chart.height))
                for x, value in zip(xs, values)
            ]

        for values, color in ((mean, COLORS["cyan"]), (best, COLORS["green"])):
            line = points(values)
            if len(line) > 1:
                pygame.draw.lines(self.surface, color, False, line, 2)
            elif line:
                pygame.draw.circle(self.surface, color, line[0], 3)
        self._text(
            "BEST",
            (rect.x + 14, rect.bottom - 20),
            size=9,
            color=COLORS["green"],
            bold=True,
        )
        self._text(
            "MEAN",
            (rect.x + 67, rect.bottom - 20),
            size=9,
            color=COLORS["cyan"],
            bold=True,
        )
        self._right_text(
            f"{low:.2f} — {high:.2f}",
            rect.right - 12,
            rect.bottom - 20,
            size=9,
            color=COLORS["muted"],
        )

    def _q_values(self, data: Mapping[str, Any]) -> list[tuple[str, float]]:
        raw = data.get("q_values", _mapping(data.get("network")).get("q_values"))
        if isinstance(raw, Mapping):
            return [
                (str(label).upper(), _finite(value)) for label, value in raw.items()
            ]
        values = [_finite(value) for value in _sequence(raw)]
        return [
            (ACTION_LABELS[index] if index < len(ACTION_LABELS) else f"A{index}", value)
            for index, value in enumerate(values)
        ]

    def _q_panel(self, rect: pygame.Rect, data: Mapping[str, Any]) -> None:
        self._panel(rect, "LIVE ACTION VALUES  Q(s, a)", accent=COLORS["blue"])
        values = self._q_values(data)
        if not values:
            self._text(
                "Waiting for the first policy step",
                (rect.x + 15, rect.y + 53),
                size=11,
                color=COLORS["muted"],
            )
            return
        strongest = max(abs(value) for _, value in values) or 1.0
        greedy = max(range(len(values)), key=lambda index: values[index][1])
        chosen = _integer(
            data.get("last_action", data.get("selected_action", greedy)), greedy
        )
        y = rect.y + 42
        row_height = max(22, min(30, (rect.height - 51) // max(1, len(values))))
        label_width = 69
        value_width = 48
        bar_x = rect.x + label_width + 12
        bar_width = rect.width - label_width - value_width - 28
        middle = bar_x + bar_width // 2
        for index, (label, value) in enumerate(values):
            selected = index == chosen
            color = COLORS["green"] if index == greedy else COLORS["cyan"]
            if selected:
                pygame.draw.rect(
                    self.surface,
                    COLORS["panel_high"],
                    (rect.x + 7, y - 4, rect.width - 14, row_height - 2),
                    border_radius=5,
                )
            self._text(
                label,
                (rect.x + 13, y),
                size=9,
                color=color if selected else COLORS["muted"],
                bold=selected,
            )
            pygame.draw.line(
                self.surface, COLORS["grid"], (middle, y + 6), (middle, y + 14), 1
            )
            extent = round((bar_width / 2 - 2) * min(1.0, abs(value) / strongest))
            start = middle if value >= 0 else middle - extent
            pygame.draw.rect(
                self.surface,
                color if value >= 0 else COLORS["magenta"],
                (start, y + 7, max(1, extent), 6),
                border_radius=3,
            )
            self._right_text(
                f"{value:+.3f}", rect.right - 11, y, size=9, color=COLORS["text"]
            )
            y += row_height
        raw_counts = data.get("action_counts")
        if isinstance(raw_counts, Mapping):
            counts = [
                max(
                    0,
                    _integer(
                        raw_counts.get(
                            label,
                            raw_counts.get(label.lower(), raw_counts.get(index, 0)),
                        )
                    ),
                )
                for index, (label, _) in enumerate(values)
            ]
        else:
            counts = [max(0, _integer(value)) for value in _sequence(raw_counts)]
        counts = counts[: len(values)]
        total = sum(counts)
        if total and rect.bottom - y >= 24:
            self._text(
                "ACTION USE",
                (rect.x + 13, rect.bottom - 30),
                size=8,
                color=COLORS["muted"],
                bold=True,
            )
            self._right_text(
                f"{total:,} DECISIONS",
                rect.right - 12,
                rect.bottom - 30,
                size=8,
                color=COLORS["muted"],
            )
            strip = pygame.Rect(rect.x + 13, rect.bottom - 14, rect.width - 26, 6)
            pygame.draw.rect(self.surface, COLORS["grid"], strip, border_radius=3)
            palette = (
                COLORS["cyan"],
                COLORS["green"],
                COLORS["red"],
                COLORS["magenta"],
                COLORS["yellow"],
            )
            cursor = strip.x
            cumulative = 0
            for index, count in enumerate(counts):
                cumulative += count
                right = (
                    strip.right
                    if index == len(counts) - 1
                    else strip.x + round(strip.width * cumulative / total)
                )
                if right > cursor:
                    pygame.draw.rect(
                        self.surface,
                        palette[index % len(palette)],
                        (cursor, strip.y, right - cursor, strip.height),
                    )
                cursor = right

    def _observations(
        self, env: DrivingEnv, data: Mapping[str, Any]
    ) -> list[tuple[str, float]]:
        raw = data.get("observations", data.get("observation"))
        if isinstance(raw, Mapping):
            return [
                (str(label).replace("_", " ").upper(), _finite(value))
                for label, value in raw.items()
            ]
        values = _sequence(raw)
        if not values:
            try:
                values = list(env.observation())
            except (AttributeError, TypeError, ValueError):
                values = []
        labels = getattr(env, "OBSERVATION_LABELS", ())
        return [
            (
                (
                    str(labels[index]).replace("_", " ").upper()
                    if index < len(labels)
                    else f"OBS {index}"
                ),
                _finite(value),
            )
            for index, value in enumerate(values)
        ]

    def _observation_panel(
        self, rect: pygame.Rect, env: DrivingEnv, data: Mapping[str, Any]
    ) -> None:
        self._panel(rect, "OBSERVATION VECTOR", accent=COLORS["magenta"])
        observations = self._observations(env, data)
        if not observations:
            self._text(
                "Observation unavailable",
                (rect.x + 14, rect.y + 48),
                size=11,
                color=COLORS["muted"],
            )
            return
        columns = 2 if rect.width >= 280 else 1
        rows = math.ceil(min(len(observations), 16) / columns)
        row_height = max(21, min(28, (rect.height - 47) // max(1, rows)))
        column_width = (rect.width - 20) // columns
        for index, (label, value) in enumerate(observations[: columns * rows]):
            column = index // rows
            row = index % rows
            x = rect.x + 12 + column * column_width
            y = rect.y + 42 + row * row_height
            clipped = self._fit_text(label, column_width - 60, 8, False)
            self._text(clipped, (x, y), size=8, color=COLORS["muted"])
            width = max(16, column_width - 72)
            bar = pygame.Rect(x, y + 12, width, 5)
            pygame.draw.rect(self.surface, COLORS["grid"], bar, border_radius=3)
            midpoint = bar.centerx
            extent = round((width / 2) * min(1.0, abs(value)))
            start = midpoint if value >= 0 else midpoint - extent
            pygame.draw.rect(
                self.surface,
                COLORS["cyan"] if value >= 0 else COLORS["magenta"],
                (start, bar.y, max(1, extent), bar.height),
                border_radius=3,
            )
            value_text = f"{value:+.2f}"
            self._right_text(
                value_text, x + column_width - 5, y, size=8, color=COLORS["text"]
            )

    def _overview(self, env: DrivingEnv, data: Mapping[str, Any]) -> None:
        self._draw_track(pygame.Rect(20, 94, 752, 646), env, telemetry=data)
        x, top, width, gap = 792, 94, 588, 8
        card_width = (width - gap * 3) // 4
        algorithm = (
            str(data.get("algorithm", data.get("method", "DQN + GENETIC")))
            .replace("_", " ")
            .upper()
        )
        algorithm = {
            "GENETIC DOUBLE DQN": "GA + DDQN",
            "GENETIC DQN": "GA + DDQN",
            "DOUBLE DQN": "DOUBLE DQN",
        }.get(algorithm, algorithm)
        algorithm_detail = {
            "DQN": "replay learner",
            "DOUBLE DQN": "replay learner",
            "GENETIC": "weight evolution",
            "GA + DDQN": "hybrid learner",
        }.get(algorithm, "value learner")
        generation = _integer(data.get("generation", data.get("generation_index", 0)))
        member = _integer(data.get("member", data.get("member_index", 0)))
        population_size = _integer(
            data.get("population_size", len(self._population_rows(data)))
        )
        fitness = _finite(data.get("current_fitness", data.get("fitness")))
        epsilon = _finite(data.get("epsilon"))
        cards = (
            ("ALGORITHM", algorithm, algorithm_detail, COLORS["cyan"]),
            (
                "GENERATION",
                f"{generation:04d}",
                f"member {member + 1}/{max(1, population_size)}",
                COLORS["yellow"],
            ),
            (
                "FITNESS",
                _compact_number(fitness),
                f"best {_compact_number(_finite(data.get('best_fitness', fitness)))}",
                COLORS["green"],
            ),
            (
                "EXPLORATION",
                f"{epsilon * 100:05.1f}%",
                f"loss {_finite(data.get('last_loss', data.get('loss'))):.4f}",
                COLORS["magenta"],
            ),
        )
        for index, (label, value, detail, color) in enumerate(cards):
            self._metric_card(
                pygame.Rect(x + index * (card_width + gap), top, card_width, 86),
                label,
                value,
                detail,
                color=color,
            )
        self._population_panel(pygame.Rect(x, 190, 286, 252), data)
        self._history_panel(pygame.Rect(x + 296, 190, 292, 252), data)
        self._q_panel(pygame.Rect(x, 452, 286, 288), data)
        self._observation_panel(pygame.Rect(x + 296, 452, 292, 288), env, data)

    def _network_snapshot(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        for key in ("network", "network_snapshot", "nn", "model"):
            candidate = data.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if "layers" in data or "architecture" in data:
            return data
        return {}

    def _network_layers(
        self, env: DrivingEnv, data: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        snapshot = self._network_snapshot(data)
        raw_layers = _sequence(snapshot.get("layers"))
        layers: list[dict[str, Any]] = []
        if raw_layers and all(isinstance(layer, Mapping) for layer in raw_layers):
            for index, raw in enumerate(raw_layers):
                layer = _mapping(raw)
                activations = [
                    _finite(value) for value in _sequence(layer.get("activations"))
                ]
                size = _integer(layer.get("size", len(activations)), len(activations))
                weights = [
                    [_finite(value) for value in _sequence(row)]
                    for row in _sequence(layer.get("weights"))
                ]
                layers.append(
                    {
                        "name": str(layer.get("name", f"layer_{index}")),
                        "size": max(size, len(activations)),
                        "activations": activations,
                        "weights": weights,
                    }
                )
            return layers
        architecture = [
            max(1, _integer(value, 1))
            for value in _sequence(
                snapshot.get("architecture", data.get("architecture"))
            )
        ]
        if not architecture:
            observations = self._observations(env, data)
            q_values = self._q_values(data)
            architecture = [len(observations) or 12, len(q_values) or 5]
        all_activations = _sequence(snapshot.get("activations"))
        for index, size in enumerate(architecture):
            values = (
                _sequence(all_activations[index])
                if index < len(all_activations)
                else []
            )
            layers.append(
                {
                    "name": (
                        "observation"
                        if index == 0
                        else (
                            "q_values"
                            if index == len(architecture) - 1
                            else f"hidden_{index}"
                        )
                    ),
                    "size": size,
                    "activations": [_finite(value) for value in values],
                    "weights": [],
                }
            )
        raw_weights = _sequence(snapshot.get("weights", snapshot.get("connections")))
        for index, matrix in enumerate(raw_weights, 1):
            if index >= len(layers):
                break
            layers[index]["weights"] = [
                [_finite(value) for value in _sequence(row)]
                for row in _sequence(matrix)
            ]
        return layers

    @staticmethod
    def _sample_indices(size: int, maximum: int = 12) -> list[int]:
        if size <= maximum:
            return list(range(size))
        return sorted(
            {round(index * (size - 1) / (maximum - 1)) for index in range(maximum)}
        )

    def _network_panel(
        self, rect: pygame.Rect, env: DrivingEnv, data: Mapping[str, Any]
    ) -> None:
        self._panel(
            rect, "REAL NETWORK · LIVE WEIGHTS AND ACTIVATIONS", accent=COLORS["cyan"]
        )
        snapshot = self._network_snapshot(data)
        layers = self._network_layers(env, data)
        content = pygame.Rect(
            rect.x + 36, rect.y + 52, rect.width - 72, rect.height - 91
        )
        positions: list[list[tuple[int, int, int]]] = []
        for layer_index, layer in enumerate(layers):
            size = max(1, _integer(layer["size"], 1))
            indices = self._sample_indices(size)
            x = (
                content.centerx
                if len(layers) == 1
                else round(content.x + layer_index * content.width / (len(layers) - 1))
            )
            node_positions = [
                (
                    x,
                    round(content.y + (row + 0.5) * content.height / len(indices)),
                    source_index,
                )
                for row, source_index in enumerate(indices)
            ]
            positions.append(node_positions)

        connections = pygame.Surface(self.surface.get_size(), pygame.SRCALPHA)
        streamed_weights = False
        for destination_index in range(1, len(layers)):
            matrix = layers[destination_index]["weights"]
            if matrix:
                streamed_weights = True
            available = [abs(value) for row in matrix for value in row]
            scale = max(available, default=1.0) or 1.0
            for source_x, source_y, source_node in positions[destination_index - 1]:
                for destination_x, destination_y, destination_node in positions[
                    destination_index
                ]:
                    if destination_node >= len(matrix) or source_node >= len(
                        matrix[destination_node]
                    ):
                        continue
                    weight = matrix[destination_node][source_node]
                    intensity = min(1.0, abs(weight) / scale)
                    base = COLORS["cyan"] if weight >= 0 else COLORS["magenta"]
                    color = (*base, round(32 + intensity * 150))
                    pygame.draw.line(
                        connections,
                        color,
                        (source_x + 6, source_y),
                        (destination_x - 6, destination_y),
                        1 + int(intensity > 0.72),
                    )
        self.surface.blit(connections, (0, 0))

        observations = self._observations(env, data)
        q_values = self._q_values(data)
        for layer_index, (layer, nodes) in enumerate(zip(layers, positions)):
            activations = layer["activations"]
            activation_scale = (
                max((abs(value) for value in activations), default=1.0) or 1.0
            )
            for x, y, source_index in nodes:
                value = (
                    activations[source_index]
                    if source_index < len(activations)
                    else 0.0
                )
                level = min(1.0, abs(value) / activation_scale) if activations else 0.0
                color = COLORS["green"] if value >= 0 else COLORS["magenta"]
                fill = tuple(
                    round(
                        COLORS["panel_alt"][index] * (1 - level) + color[index] * level
                    )
                    for index in range(3)
                )
                pygame.draw.circle(self.surface, fill, (x, y), 8)
                pygame.draw.circle(
                    self.surface, color if activations else COLORS["edge"], (x, y), 8, 1
                )
                label = ""
                if layer_index == 0 and source_index < len(observations):
                    label = observations[source_index][0]
                    text = self._font(7).render(label[:14], True, COLORS["muted"])
                    self.surface.blit(text, (x - 14 - text.get_width(), y - 4))
                elif layer_index == len(layers) - 1 and source_index < len(q_values):
                    label = q_values[source_index][0]
                    self._text(label, (x + 13, y - 5), size=8, color=COLORS["muted"])
            label = str(layer["name"]).replace("_", " ").upper()
            self._text(
                label,
                (nodes[0][0] - 34, content.bottom + 9),
                size=9,
                color=COLORS["text"],
                bold=True,
            )
            self._text(
                f"{layer['size']} UNITS",
                (nodes[0][0] - 34, content.bottom + 22),
                size=8,
                color=COLORS["muted"],
            )
        parameter_count = _integer(
            snapshot.get("parameter_count", data.get("parameter_count"))
        )
        badge = (
            f"{parameter_count:,} TRAINABLE PARAMETERS"
            if parameter_count
            else "PARAMETER COUNT UNAVAILABLE"
        )
        self._right_text(
            badge,
            rect.right - 15,
            rect.y + 12,
            size=10,
            color=COLORS["muted"],
            bold=True,
        )
        if not streamed_weights:
            message = "Architecture available · waiting for live connection weights"
            text = self._font(10, True).render(message, True, COLORS["yellow"])
            box = text.get_rect(midtop=(rect.centerx, rect.y + 35))
            self.surface.blit(text, box)

    def _learning_stats(self, rect: pygame.Rect, data: Mapping[str, Any]) -> None:
        replay = _mapping(data.get("replay", data.get("memory")))
        size = _integer(replay.get("size", data.get("replay_size")))
        capacity = max(
            1, _integer(replay.get("capacity", data.get("replay_capacity", 1)), 1)
        )
        metrics = (
            ("REPLAY", f"{size:,} / {capacity:,}", size / capacity, COLORS["blue"]),
            (
                "EPSILON",
                f"{_finite(data.get('epsilon')):.3f}",
                _finite(data.get("epsilon")),
                COLORS["magenta"],
            ),
            (
                "LOSS",
                f"{_finite(data.get('last_loss', data.get('loss'))):.5f}",
                min(1.0, _finite(data.get("last_loss", data.get("loss")))),
                COLORS["red"],
            ),
            (
                "TD ERROR",
                f"{_finite(data.get('mean_absolute_td_error', data.get('td_error'))):.5f}",
                min(
                    1.0,
                    _finite(data.get("mean_absolute_td_error", data.get("td_error"))),
                ),
                COLORS["orange"],
            ),
            (
                "GRAD STEPS",
                f"{_integer(data.get('gradient_steps')):,}",
                1.0,
                COLORS["green"],
            ),
            (
                "TARGET SYNCS",
                f"{_integer(data.get('target_syncs')):,}",
                1.0 if _integer(data.get("target_syncs")) else 0.0,
                COLORS["cyan"],
            ),
        )
        self._panel(rect, "LEARNING SIGNAL", accent=COLORS["magenta"])
        columns = len(metrics) if rect.width >= 700 else 3
        rows = math.ceil(len(metrics) / columns)
        width = (rect.width - 28) // columns
        row_height = (rect.height - 46) // rows
        for index, (label, value, fraction, color) in enumerate(metrics):
            column = index % columns
            row = index // columns
            x = rect.x + 14 + column * width
            y = rect.y + 41 + row * row_height
            if column:
                pygame.draw.line(
                    self.surface,
                    COLORS["grid"],
                    (x, y),
                    (x, min(rect.bottom - 10, y + row_height - 5)),
                )
            if row and column == 0:
                pygame.draw.line(
                    self.surface,
                    COLORS["grid"],
                    (rect.x + 14, y - 5),
                    (rect.right - 14, y - 5),
                )
            self._text(
                self._fit_text(label, width - 18, 9, True),
                (x + 9, y + 3),
                size=9,
                color=COLORS["muted"],
                bold=True,
            )
            value_size = 14 if width >= 115 else 12
            self._text(
                self._fit_text(value, width - 18, value_size, True),
                (x + 9, y + 21),
                size=value_size,
                color=color,
                bold=True,
            )
            bar = pygame.Rect(
                x + 9,
                min(rect.bottom - 14, y + row_height - 15),
                width - 20,
                5,
            )
            pygame.draw.rect(self.surface, COLORS["grid"], bar, border_radius=3)
            fill = bar.copy()
            fill.width = max(1, round(bar.width * min(1.0, max(0.0, fraction))))
            pygame.draw.rect(self.surface, color, fill, border_radius=3)

    def _network(self, env: DrivingEnv, data: Mapping[str, Any]) -> None:
        self._draw_track(
            pygame.Rect(20, 94, 380, 292),
            env,
            title="POLICY POSITION",
            telemetry=data,
        )
        self._q_panel(pygame.Rect(20, 396, 380, 344), data)
        self._network_panel(pygame.Rect(420, 94, 960, 514), env, data)
        self._learning_stats(pygame.Rect(420, 618, 960, 122), data)

    def _memory_samples(self, data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        replay = _mapping(data.get("replay", data.get("memory")))
        raw = data.get(
            "memory_samples", data.get("replay_samples", replay.get("samples"))
        )
        return [_mapping(item) for item in _sequence(raw)]

    def _buffer_panel(self, rect: pygame.Rect, data: Mapping[str, Any]) -> None:
        self._panel(rect, "EXPERIENCE REPLAY", accent=COLORS["blue"])
        replay = _mapping(data.get("replay", data.get("memory")))
        size = max(0, _integer(replay.get("size", data.get("replay_size"))))
        capacity = max(
            1, _integer(replay.get("capacity", data.get("replay_capacity", 1)), 1)
        )
        ratio = min(1.0, size / capacity)
        self._text(
            f"{size:,}",
            (rect.x + 16, rect.y + 42),
            size=27,
            color=COLORS["cyan"],
            bold=True,
        )
        self._text(
            f"of {capacity:,} transitions",
            (rect.x + 17, rect.y + 75),
            size=11,
            color=COLORS["muted"],
        )
        ring = pygame.Rect(rect.x + 17, rect.y + 103, rect.width - 34, 18)
        pygame.draw.rect(self.surface, COLORS["grid"], ring, border_radius=9)
        fill = ring.copy()
        fill.width = max(2, round(ring.width * ratio))
        pygame.draw.rect(self.surface, COLORS["blue"], fill, border_radius=9)
        divisions = 20
        for index in range(1, divisions):
            x = round(ring.x + index * ring.width / divisions)
            pygame.draw.line(
                self.surface, COLORS["panel"], (x, ring.y + 2), (x, ring.bottom - 2), 1
            )
        stats = (
            ("FILL", f"{ratio * 100:.1f}%"),
            ("MEAN R", f"{_finite(replay.get('mean_reward')):+.3f}"),
            ("TERMINAL", f"{_integer(replay.get('terminal')):,}"),
            (
                "BATCH",
                f"{_integer(data.get('batch_size', replay.get('batch_size'))):,}",
            ),
        )
        for index, (label, value) in enumerate(stats):
            x = rect.x + 16 + index * (rect.width - 32) // len(stats)
            self._text(
                label, (x, rect.y + 140), size=8, color=COLORS["muted"], bold=True
            )
            self._text(
                value, (x, rect.y + 156), size=11, color=COLORS["text"], bold=True
            )

    def _sample_panel(self, rect: pygame.Rect, data: Mapping[str, Any]) -> None:
        self._panel(rect, "RECENT TRANSITIONS", accent=COLORS["orange"])
        samples = self._memory_samples(data)[-12:]
        if not samples:
            self._text(
                "Samples appear here once experience is collected",
                (rect.x + 16, rect.y + 49),
                size=11,
                color=COLORS["muted"],
            )
            return
        y = rect.y + 43
        row_height = max(22, (rect.height - 52) // len(samples))
        self._text("ACTION", (rect.x + 15, y), size=8, color=COLORS["muted"], bold=True)
        self._text(
            "REWARD", (rect.x + 112, y), size=8, color=COLORS["muted"], bold=True
        )
        self._text("NEXT", (rect.x + 205, y), size=8, color=COLORS["muted"], bold=True)
        y += 20
        reward_scale = (
            max((abs(_finite(sample.get("reward"))) for sample in samples), default=1.0)
            or 1.0
        )
        for sample in reversed(samples):
            action = _integer(sample.get("action"), -1)
            action_label = (
                ACTION_LABELS[action]
                if 0 <= action < len(ACTION_LABELS)
                else f"A{action}"
            )
            reward = _finite(sample.get("reward"))
            terminal = bool(sample.get("done", sample.get("terminal", False)))
            color = COLORS["green"] if reward >= 0 else COLORS["red"]
            self._text(action_label, (rect.x + 15, y), size=9, color=COLORS["text"])
            self._text(
                f"{reward:+.3f}", (rect.x + 112, y), size=9, color=color, bold=True
            )
            bar = pygame.Rect(rect.x + 205, y + 3, rect.width - 268, 6)
            pygame.draw.rect(self.surface, COLORS["grid"], bar, border_radius=3)
            fill = bar.copy()
            fill.width = max(1, round(bar.width * abs(reward) / reward_scale))
            pygame.draw.rect(self.surface, color, fill, border_radius=3)
            self._right_text(
                "END" if terminal else "→",
                rect.right - 14,
                y - 1,
                size=10,
                color=COLORS["orange"] if terminal else COLORS["muted"],
                bold=terminal,
            )
            y += row_height

    def _training_curves(self, rect: pygame.Rect, data: Mapping[str, Any]) -> None:
        self._panel(rect, "OPTIMIZATION TRACE", accent=COLORS["red"])
        loss = [_finite(value) for value in _sequence(data.get("loss_history"))][-100:]
        epsilon = [_finite(value) for value in _sequence(data.get("epsilon_history"))][
            -100:
        ]
        chart = pygame.Rect(rect.x + 20, rect.y + 47, rect.width - 38, rect.height - 70)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = round(chart.bottom - fraction * chart.height)
            pygame.draw.line(
                self.surface, COLORS["grid"], (chart.x, y), (chart.right, y), 1
            )

        def curve(values: list[float], color: tuple[int, int, int]) -> None:
            if not values:
                return
            high = max(values) or 1.0
            points = [
                (
                    (
                        chart.centerx
                        if len(values) == 1
                        else round(chart.x + index * chart.width / (len(values) - 1))
                    ),
                    round(
                        chart.bottom - min(1.0, max(0.0, value / high)) * chart.height
                    ),
                )
                for index, value in enumerate(values)
            ]
            if len(points) > 1:
                pygame.draw.lines(self.surface, color, False, points, 2)
            else:
                pygame.draw.circle(self.surface, color, points[0], 3)

        curve(loss, COLORS["red"])
        curve(epsilon, COLORS["magenta"])
        if not loss and not epsilon:
            self._text(
                "Optimization history not streamed",
                (chart.x + 12, chart.centery - 5),
                size=11,
                color=COLORS["muted"],
            )
        self._text(
            "LOSS",
            (rect.x + 16, rect.bottom - 19),
            size=8,
            color=COLORS["red"],
            bold=True,
        )
        self._text(
            "EPSILON",
            (rect.x + 60, rect.bottom - 19),
            size=8,
            color=COLORS["magenta"],
            bold=True,
        )

    def _memory(self, env: DrivingEnv, data: Mapping[str, Any]) -> None:
        self._buffer_panel(pygame.Rect(20, 94, 500, 194), data)
        self._training_curves(pygame.Rect(20, 298, 500, 211), data)
        self._draw_track(
            pygame.Rect(20, 519, 500, 221),
            env,
            title="EXPERIENCE SOURCE",
            telemetry=data,
        )
        self._sample_panel(pygame.Rect(540, 94, 400, 415), data)
        self._observation_panel(pygame.Rect(960, 94, 420, 415), env, data)
        self._q_panel(pygame.Rect(540, 519, 400, 221), data)
        self._learning_stats(pygame.Rect(960, 519, 420, 221), data)

    def draw(
        self,
        env: DrivingEnv | None = None,
        telemetry: Mapping[str, Any] | None = None,
    ) -> pygame.Surface:
        """Render the selected training tab and return the reusable surface."""

        if env is not None:
            self.env = env
        if telemetry is not None:
            self.telemetry = _mapping(telemetry)
        data = self._combined_telemetry(self.env, telemetry)
        self.surface.fill(COLORS["background"])
        self._header(data)
        if self.active_tab == "NETWORK":
            self._network(self.env, data)
        elif self.active_tab == "MEMORY":
            self._memory(self.env, data)
        else:
            self._overview(self.env, data)
        return self.surface

    render_training = draw

    def _race_data(
        self,
        env: DrivingEnv,
        explicit: object,
        supplied: Mapping[str, Any],
        prefix: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        try:
            result.update(env.telemetry())
        except (AttributeError, TypeError, ValueError):
            pass
        result.update(_mapping(explicit))
        for key, value in supplied.items():
            if key.startswith(prefix):
                result[key[len(prefix) :]] = value
        # Shared race keys remain useful when no nested human/champion maps are
        # supplied (for example generation and champion fitness).
        for key, value in supplied.items():
            result.setdefault(key, value)
        return result

    def _race_scoreboard(
        self,
        rect: pygame.Rect,
        human: Mapping[str, Any],
        champion: Mapping[str, Any],
        data: Mapping[str, Any],
    ) -> None:
        self._panel(rect, "LIVE HEAD-TO-HEAD", accent=COLORS["yellow"], high=True)
        rows = (
            ("YOU", human, COLORS["human"]),
            ("CHAMPION", champion, COLORS["champion"]),
        )
        y = rect.y + 48
        for name, values, color in rows:
            pygame.draw.rect(
                self.surface,
                COLORS["panel"],
                (rect.x + 13, y, rect.width - 26, 118),
                border_radius=9,
            )
            pygame.draw.circle(self.surface, color, (rect.x + 29, y + 20), 6)
            self._text(name, (rect.x + 43, y + 10), size=13, color=color, bold=True)
            lap = _integer(values.get("laps")) + 1
            progress = _finite(values.get("progress"))
            speed = _finite(values.get("speed"))
            self._text(
                f"LAP {lap}",
                (rect.x + 25, y + 43),
                size=10,
                color=COLORS["muted"],
                bold=True,
            )
            self._right_text(
                f"{speed:05.1f} U/S",
                rect.right - 25,
                y + 43,
                size=10,
                color=COLORS["text"],
                bold=True,
            )
            bar = pygame.Rect(rect.x + 25, y + 69, rect.width - 50, 10)
            pygame.draw.rect(self.surface, COLORS["grid"], bar, border_radius=5)
            fill = bar.copy()
            fill.width = max(2, round(bar.width * min(1.0, max(0.0, progress))))
            pygame.draw.rect(self.surface, color, fill, border_radius=5)
            self._text(
                f"{progress * 100:05.1f}%",
                (rect.x + 25, y + 87),
                size=10,
                color=color,
                bold=True,
            )
            finish_time = values.get("finish_time")
            best_lap = values.get("best_lap_time")
            if finish_time is not None:
                time_label = "FINISH"
                time_text = f"{_finite(finish_time):.3f}s"
            else:
                time_label = "BEST"
                time_text = "--" if best_lap is None else f"{_finite(best_lap):.3f}s"
            self._right_text(
                f"{time_label} {time_text}",
                rect.right - 25,
                y + 87,
                size=10,
                color=COLORS["muted"],
            )
            y += 130
        human_progress = _finite(human.get("progress"))
        champion_progress = _finite(champion.get("progress"))
        delta = (human_progress - champion_progress + 0.5) % 1.0 - 0.5
        winner = str(data.get("winner") or "").lower()
        if winner == "human":
            leader, leader_color = "YOU WIN!", COLORS["human"]
        elif winner == "champion":
            leader, leader_color = "CHAMPION WINS", COLORS["champion"]
        elif winner == "tie":
            leader, leader_color = "PHOTO FINISH · TIE", COLORS["green"]
        else:
            leader = "YOU LEAD" if delta >= 0 else "CHAMPION LEADS"
            leader_color = COLORS["human"] if delta >= 0 else COLORS["champion"]
        self._text(
            leader,
            (rect.x + 18, y + 2),
            size=16 if winner == "tie" else 18,
            color=leader_color,
            bold=True,
        )
        elapsed = _finite(data.get("elapsed"))
        self._right_text(
            f"{elapsed:06.2f}s" if winner else f"{abs(delta) * 100:.1f}% LAP",
            rect.right - 18,
            y + 6,
            size=11,
            color=COLORS["text"],
            bold=True,
        )
        generation = _integer(data.get("generation", data.get("generation_index")))
        fitness = _finite(data.get("best_fitness", data.get("champion_fitness")))
        member = _integer(data.get("champion_member", data.get("best_member", 0)))
        self._text(
            f"GEN {generation:04d}  ·  MEMBER {member + 1:03d}",
            (rect.x + 18, y + 35),
            size=10,
            color=COLORS["muted"],
            bold=True,
        )
        self._right_text(
            f"FITNESS {_compact_number(fitness)}",
            rect.right - 18,
            y + 35,
            size=10,
            color=COLORS["green"],
            bold=True,
        )

    def draw_race(
        self,
        human_env: DrivingEnv,
        champion_env: DrivingEnv | None = None,
        telemetry: Mapping[str, Any] | None = None,
        human_pose: object = None,
        champion_pose: object = None,
    ) -> pygame.Surface:
        """Draw a fair human/champion race, deriving omitted poses from each env."""

        champion_env = champion_env or human_env
        data = _mapping(telemetry)
        human = self._race_data(human_env, data.get("human"), data, "human_")
        champion = self._race_data(
            champion_env, data.get("champion"), data, "champion_"
        )
        if human_pose is None:
            human_pose = human_env.vehicle.state
        if champion_pose is None:
            champion_pose = champion_env.vehicle.state
        human_rays = human.get("rays", human.get("sensor_rays"))
        champion_rays = champion.get("rays", champion.get("sensor_rays"))
        self.surface.fill(COLORS["background"])
        self._header(data, race=True)
        self._draw_track(
            pygame.Rect(20, 94, 990, 646),
            human_env,
            title="RACE TRACK · IDENTICAL PHYSICS",
            cars=(
                (human_pose, COLORS["human"], "YOU"),
                (champion_pose, COLORS["champion"], "CHAMPION"),
            ),
            telemetry=data,
            include_population=False,
            ray_sources=(
                (human_rays, human_env, human_pose, COLORS["human"]),
                (champion_rays, champion_env, champion_pose, COLORS["champion"]),
            ),
        )
        scoreboard = pygame.Rect(1_030, 94, 350, 510)
        self._race_scoreboard(scoreboard, human, champion, data)
        hint = pygame.Rect(1_030, 620, 350, 120)
        pygame.draw.rect(self.surface, COLORS["cyan"], hint, border_radius=12)
        self._text(
            "P",
            (hint.x + 22, hint.y + 14),
            size=48,
            color=COLORS["background"],
            bold=True,
        )
        self._text(
            "RETURN TO",
            (hint.x + 87, hint.y + 23),
            size=13,
            color=COLORS["background"],
            bold=True,
        )
        self._text(
            "TRAINING",
            (hint.x + 87, hint.y + 43),
            size=24,
            color=COLORS["background"],
            bold=True,
        )
        self._text(
            "Race progress is not added to replay",
            (hint.x + 24, hint.bottom - 26),
            size=10,
            color=COLORS["cyan_dim"],
            bold=True,
        )
        return self.surface

    render_race = draw_race


# Short aliases keep integrations readable without introducing a second API.
LearningVisualization = DrivingLearningVisualization
DrivingLearningDashboard = DrivingLearningVisualization


__all__ = (
    "ACTION_LABELS",
    "DrivingLearningDashboard",
    "DrivingLearningVisualization",
    "LEARNING_WINDOW_HEIGHT",
    "LEARNING_WINDOW_SIZE",
    "LEARNING_WINDOW_WIDTH",
    "LearningVisualization",
)
