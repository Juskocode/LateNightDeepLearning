"""Pygame sprite presentation for cars and deterministic particles."""

from __future__ import annotations

import math
from pathlib import Path
import random

import pygame

from .math2d import Vec2
from .terrain import ParticleMode, Terrain
from .vehicle import CarBuild, DriverControls, Vehicle


DEFAULT_CAR_SPRITE = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "sprites"
    / "driving"
    / "car-body.png"
)


def _load_alpha_cropped(path: Path) -> pygame.Surface | None:
    try:
        image = pygame.image.load(str(path))
    except (FileNotFoundError, pygame.error, OSError):
        return None
    if image.get_flags() & pygame.SRCALPHA == 0:
        image = image.convert_alpha() if pygame.display.get_surface() else image.copy()
    bounds = image.get_bounding_rect(min_alpha=2)
    if bounds.width <= 1 or bounds.height <= 1:
        return None
    return image.subsurface(bounds).copy()


class CarSprite(pygame.sprite.Sprite):
    """Rotating car sprite with upgrade-dependent wheels and suspension.

    The generated repository image is used by default.  A missing or invalid
    custom path falls back to a crisp procedural body so headless experiments
    never depend on an asset loader succeeding.
    """

    def __init__(self, build: CarBuild, image_path: str | Path | None = None):
        super().__init__()
        self.image_path = (
            Path(image_path).expanduser() if image_path else DEFAULT_CAR_SPRITE
        )
        loaded = _load_alpha_cropped(self.image_path)
        self.using_external_image = loaded is not None
        self._source_body = loaded or self._procedural_body()
        self._build = build
        self._chassis = self._compose_chassis(build)
        self.image = self._chassis
        self.rect = self.image.get_rect()

    @staticmethod
    def _procedural_body() -> pygame.Surface:
        surface = pygame.Surface((30, 52), pygame.SRCALPHA)
        pygame.draw.polygon(
            surface,
            (16, 194, 222),
            ((7, 3), (23, 3), (28, 13), (27, 43), (21, 50), (9, 50), (3, 43), (2, 13)),
        )
        pygame.draw.rect(surface, (245, 188, 28), (8, 5, 14, 42), border_radius=4)
        pygame.draw.polygon(
            surface, (24, 43, 58), ((7, 17), (23, 17), (21, 29), (9, 29))
        )
        pygame.draw.line(surface, (224, 247, 250), (7, 8), (23, 8), 2)
        pygame.draw.circle(surface, (236, 67, 48), (8, 45), 2)
        pygame.draw.circle(surface, (236, 67, 48), (22, 45), 2)
        return surface

    def _compose_chassis(self, build: CarBuild) -> pygame.Surface:
        chassis = pygame.Surface((38, 58), pygame.SRCALPHA)
        wheel_width = 4 + build.grip // 2
        wheel_color = (18, 20, 24)
        spring_color = (83, 210, 232) if build.suspension >= 3 else (135, 145, 157)
        for y in (10, 40):
            pygame.draw.line(chassis, spring_color, (8, y + 4), (30, y + 4), 2)
            pygame.draw.rect(
                chassis, wheel_color, (2, y, wheel_width, 13), border_radius=2
            )
            pygame.draw.rect(
                chassis,
                wheel_color,
                (36 - wheel_width, y, wheel_width, 13),
                border_radius=2,
            )
            tread = (92, 103, 114)
            pygame.draw.line(
                chassis, tread, (3, y + 3), (3 + wheel_width - 2, y + 3), 1
            )
            pygame.draw.line(chassis, tread, (34, y + 9), (37 - wheel_width, y + 9), 1)

        body = pygame.transform.smoothscale(self._source_body, (28, 52))
        chassis.blit(body, body.get_rect(center=(19, 29)))
        if build.motor:
            intensity = min(255, 105 + build.motor * 25)
            pygame.draw.circle(
                chassis, (intensity, 197, 255), (15, 53), 2 + build.motor // 3
            )
            pygame.draw.circle(
                chassis, (intensity, 197, 255), (23, 53), 2 + build.motor // 3
            )
        return chassis

    def set_build(self, build: CarBuild) -> None:
        if build != self._build:
            self._build = build
            self._chassis = self._compose_chassis(build)

    def sync(self, vehicle: Vehicle) -> None:
        self.set_build(vehicle.build)
        # Source art points up. Physics heading 0 points right and positive
        # angles follow the screen's clockwise Y axis.
        angle = -math.degrees(vehicle.state.heading) - 90.0
        center = (round(vehicle.state.position.x), round(vehicle.state.position.y))
        self.image = pygame.transform.rotozoom(self._chassis, angle, 1.0)
        self.rect = self.image.get_rect(center=center)


class ParticleSprite(pygame.sprite.Sprite):
    def __init__(
        self,
        position: Vec2,
        velocity: Vec2,
        color: tuple[int, int, int],
        lifetime: float,
        radius: int,
    ):
        super().__init__()
        values = (position.x, position.y, velocity.x, velocity.y, lifetime)
        if not all(math.isfinite(value) for value in values) or lifetime <= 0.0:
            raise ValueError("Particle position, velocity, and lifetime must be finite")
        if not isinstance(radius, int) or radius <= 0:
            raise ValueError("Particle radius must be a positive integer")
        self.position = position
        self.velocity = velocity
        self.color = color
        self.lifetime = lifetime
        self.remaining = lifetime
        self.radius = radius
        diameter = radius * 2 + 2
        self.image = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(round(position.x), round(position.y)))
        self._redraw()

    def _redraw(self) -> None:
        self.image.fill((0, 0, 0, 0))
        alpha = round(210 * max(0.0, self.remaining / self.lifetime))
        center = (self.image.get_width() // 2, self.image.get_height() // 2)
        pygame.draw.circle(self.image, (*self.color, alpha), center, self.radius)

    def update(self, dt: float) -> None:
        self.remaining -= dt
        if self.remaining <= 0.0:
            self.kill()
            return
        self.position = self.position + self.velocity * dt
        self.velocity = self.velocity * max(0.0, 1.0 - 1.8 * dt)
        self.rect.center = (round(self.position.x), round(self.position.y))
        self._redraw()


class ParticleSystem:
    """Bounded visual-only particles driven by a seeded local RNG."""

    MAX_PARTICLES = 260

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.random = random.Random(seed)
        self.sprites = pygame.sprite.Group()
        self._emission_budget = 0.0

    def clear(self) -> None:
        self.sprites.empty()
        self._emission_budget = 0.0

    def reset(self) -> None:
        """Clear particles and rewind a configured deterministic seed."""

        self.clear()
        self.random.seed(self.seed)

    def _add(self, particle: ParticleSprite) -> None:
        overflow = len(self.sprites) - self.MAX_PARTICLES + 1
        if overflow > 0:
            for old in self.sprites.sprites()[:overflow]:
                old.kill()
        self.sprites.add(particle)

    def emit_drive(
        self,
        vehicle: Vehicle,
        surface: Terrain,
        controls: DriverControls,
        dt: float,
    ) -> None:
        telemetry = vehicle.last_telemetry
        speed = telemetry.speed
        slip = abs(telemetry.slip_angle)
        emits_material = surface.particle_mode is not ParticleMode.NONE
        skidding = slip > math.radians(5.0) or controls.brake > 0.55
        if speed < 12.0 or not (emits_material or skidding):
            return

        rate = min(44.0, 6.0 + speed * 0.08 + math.degrees(slip) * 0.8)
        self._emission_budget += rate * dt
        forward = Vec2.from_angle(vehicle.state.heading)
        right = forward.perpendicular()
        while self._emission_budget >= 1.0:
            self._emission_budget -= 1.0
            side = self.random.choice((-1.0, 1.0))
            position = (
                vehicle.state.position
                - forward * 14.0
                + right * (side * self.random.uniform(5.0, 8.0))
            )
            jitter = Vec2(self.random.uniform(-9, 9), self.random.uniform(-9, 9))
            velocity = vehicle.state.velocity * -0.08 + jitter
            color = surface.particle_color if emits_material else (164, 173, 182)
            self._add(
                ParticleSprite(
                    position,
                    velocity,
                    color,
                    self.random.uniform(0.35, 0.85),
                    self.random.randint(2, 4),
                )
            )

    def emit_collision(self, position: Vec2, impact_speed: float) -> None:
        if not math.isfinite(impact_speed) or impact_speed <= 0.0:
            return
        count = max(4, min(28, round(impact_speed * 0.22)))
        for _ in range(count):
            angle = self.random.uniform(0.0, math.tau)
            speed = self.random.uniform(25.0, 75.0 + min(impact_speed, 120.0))
            color = self.random.choice(
                ((255, 196, 48), (255, 112, 31), (238, 239, 212))
            )
            self._add(
                ParticleSprite(
                    position,
                    Vec2.from_angle(angle) * speed,
                    color,
                    self.random.uniform(0.18, 0.50),
                    self.random.randint(1, 3),
                )
            )

    def update(self, dt: float) -> None:
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("Particle time step must be finite and non-negative")
        self.sprites.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        self.sprites.draw(surface)

    def __len__(self) -> int:
        return len(self.sprites)
