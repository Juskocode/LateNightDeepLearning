"""Deterministic top-down vehicle model with observable component effects."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .math2d import Vec2, ZERO, clamp, wrap_angle
from .terrain import Terrain


MAX_UPGRADE_LEVEL = 5


@dataclass(frozen=True, slots=True)
class CarBuild:
    """Upgrade levels used to derive the car's physical capabilities."""

    motor: int = 0
    wheels: int = 0
    suspension: int = 0
    grip: int = 0

    def __post_init__(self) -> None:
        for name in ("motor", "wheels", "suspension", "grip"):
            level = getattr(self, name)
            if type(level) is not int or not 0 <= level <= MAX_UPGRADE_LEVEL:
                raise ValueError(
                    f"{name} level must be between 0 and {MAX_UPGRADE_LEVEL}"
                )

    @property
    def acceleration(self) -> float:
        return 88.0 + 13.0 * self.motor

    @property
    def max_speed(self) -> float:
        return 205.0 + 17.0 * self.motor

    @property
    def steering_rate(self) -> float:
        return math.radians(90.0 + 8.0 * self.wheels)

    @property
    def steering_response(self) -> float:
        return 5.0 + 0.75 * self.wheels

    @property
    def stability(self) -> float:
        return 0.82 + 0.065 * self.suspension

    @property
    def grip_multiplier(self) -> float:
        return 0.84 + 0.055 * self.grip

    @property
    def tires(self) -> int:
        """Readable alias: the grip upgrade represents the tire package."""

        return self.grip


@dataclass(frozen=True, slots=True)
class DriverControls:
    throttle: float = 0.0
    steering: float = 0.0
    brake: float = 0.0

    def clamped(self) -> "DriverControls":
        try:
            finite = all(
                math.isfinite(value)
                for value in (self.throttle, self.steering, self.brake)
            )
        except TypeError as error:
            raise ValueError("Driver controls must be finite numbers") from error
        if not finite:
            raise ValueError("Driver controls must be finite numbers")
        return DriverControls(
            throttle=clamp(self.throttle, -1.0, 1.0),
            steering=clamp(self.steering, -1.0, 1.0),
            brake=clamp(self.brake, 0.0, 1.0),
        )


@dataclass(slots=True)
class VehicleState:
    position: Vec2 = ZERO
    velocity: Vec2 = ZERO
    heading: float = 0.0
    steering_angle: float = 0.0
    damage: float = 0.0

    @property
    def speed(self) -> float:
        return self.velocity.length()


@dataclass(frozen=True, slots=True)
class VehicleTelemetry:
    speed: float
    longitudinal_speed: float
    lateral_speed: float
    slip_angle: float
    acceleration: float
    effective_grip: float
    max_speed: float


class Vehicle:
    """A compact bicycle-inspired model designed for stable fixed steps."""

    LENGTH = 34.0
    WIDTH = 18.0

    def __init__(self, build: CarBuild | None = None):
        self.build = build or CarBuild()
        self.state = VehicleState()
        self.last_telemetry = VehicleTelemetry(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.build.max_speed
        )

    def reset(self, position: Vec2, heading: float) -> VehicleState:
        if not all(math.isfinite(value) for value in (position.x, position.y, heading)):
            raise ValueError("Vehicle reset pose must be finite")
        self.state = VehicleState(position=position, heading=wrap_angle(heading))
        self.last_telemetry = VehicleTelemetry(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.build.max_speed
        )
        return self.state

    def set_build(self, build: CarBuild) -> None:
        self.build = build

    def step(
        self, controls: DriverControls, terrain: Terrain, dt: float
    ) -> VehicleTelemetry:
        if not 0.0 < dt <= 0.1:
            raise ValueError("Vehicle time step must be in the (0, 0.1] interval")
        controls = controls.clamped()
        state = self.state
        forward = Vec2.from_angle(state.heading)
        right = forward.perpendicular()
        longitudinal = state.velocity.dot(forward)
        lateral = state.velocity.dot(right)
        effective_grip = terrain.grip * self.build.grip_multiplier

        target_steering = controls.steering * self.build.steering_rate
        steering_blend = min(1.0, self.build.steering_response * dt)
        state.steering_angle += (
            target_steering - state.steering_angle
        ) * steering_blend

        speed_ratio = clamp(
            abs(longitudinal) / max(1.0, self.build.max_speed), 0.0, 1.0
        )
        direction_sign = -1.0 if longitudinal < -1.0 else 1.0
        motion_factor = clamp(abs(longitudinal) / 35.0, 0.0, 1.0)
        high_speed_stability = 1.0 - 0.35 * speed_ratio
        yaw_scale = (
            motion_factor * high_speed_stability * effective_grip * self.build.stability
        )
        state.heading = wrap_angle(
            state.heading + state.steering_angle * yaw_scale * direction_sign * dt
        )

        forward = Vec2.from_angle(state.heading)
        right = forward.perpendicular()
        longitudinal = state.velocity.dot(forward)
        lateral = state.velocity.dot(right)

        engine_acceleration = (
            self.build.acceleration
            * terrain.engine_efficiency
            * min(1.0, 0.50 + effective_grip * 0.65)
            * controls.throttle
        )
        brake_acceleration = 0.0
        if controls.brake > 0.0 and abs(longitudinal) > 0.05:
            brake_acceleration = -math.copysign(190.0 * controls.brake, longitudinal)

        aerodynamic_drag = 0.0018 * longitudinal * abs(longitudinal)
        rolling_drag = terrain.rolling_resistance * 115.0
        if abs(longitudinal) > 0.05:
            rolling_drag = math.copysign(rolling_drag, longitudinal)
        else:
            rolling_drag = 0.0
        acceleration = (
            engine_acceleration + brake_acceleration - aerodynamic_drag - rolling_drag
        )
        next_longitudinal = longitudinal + acceleration * dt
        if longitudinal * next_longitudinal < 0.0 and (
            controls.brake > 0.0 or controls.throttle == 0.0
        ):
            next_longitudinal = 0.0
        reverse_limit = self.build.max_speed * 0.34
        next_longitudinal = clamp(
            next_longitudinal, -reverse_limit, self.build.max_speed
        )

        # Tires and suspension dissipate lateral movement.  Low grip leaves a
        # visible, measurable slip angle rather than snapping onto the heading.
        lateral_recovery = 3.4 * effective_grip * self.build.stability
        next_lateral = lateral * max(0.0, 1.0 - lateral_recovery * dt)
        state.velocity = forward * next_longitudinal + right * next_lateral
        state.position = state.position + state.velocity * dt

        values = (
            state.position.x,
            state.position.y,
            state.velocity.x,
            state.velocity.y,
            state.heading,
            state.steering_angle,
        )
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Vehicle physics produced a non-finite state")

        slip_angle = math.atan2(next_lateral, max(1.0, abs(next_longitudinal)))
        self.last_telemetry = VehicleTelemetry(
            speed=state.velocity.length(),
            longitudinal_speed=next_longitudinal,
            lateral_speed=next_lateral,
            slip_angle=slip_angle,
            acceleration=acceleration,
            effective_grip=effective_grip,
            max_speed=self.build.max_speed,
        )
        return self.last_telemetry

    def resolve_collision(self, track_point: Vec2, collision_radius: float) -> float:
        """Clamp to a track barrier and reflect outward velocity.

        Returns impact speed so reward and particles can react without querying
        rendering state.
        """

        delta = self.state.position - track_point
        distance = delta.length()
        if distance <= collision_radius:
            return 0.0
        outward = delta.normalized()
        self.state.position = track_point + outward * collision_radius
        outward_speed = self.state.velocity.dot(outward)
        if outward_speed <= 0.0:
            return 0.0
        impact_speed = outward_speed
        self.state.velocity = self.state.velocity - outward * (1.28 * outward_speed)
        self.state.velocity = self.state.velocity * 0.68
        self._refresh_motion_telemetry()
        values = (
            self.state.position.x,
            self.state.position.y,
            self.state.velocity.x,
            self.state.velocity.y,
            self.state.damage,
        )
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Collision resolution produced a non-finite state")
        return impact_speed

    def _refresh_motion_telemetry(self) -> None:
        """Synchronize observable motion after an instantaneous impulse."""

        forward = Vec2.from_angle(self.state.heading)
        right = forward.perpendicular()
        longitudinal = self.state.velocity.dot(forward)
        lateral = self.state.velocity.dot(right)
        previous = self.last_telemetry
        self.last_telemetry = VehicleTelemetry(
            speed=self.state.speed,
            longitudinal_speed=longitudinal,
            lateral_speed=lateral,
            slip_angle=math.atan2(lateral, max(1.0, abs(longitudinal))),
            acceleration=previous.acceleration,
            effective_grip=previous.effective_grip,
            max_speed=self.build.max_speed,
        )

    def apply_impact_damage(self, impact_speed: float) -> None:
        """Apply damage once for a new contact episode."""

        self.state.damage = clamp(
            self.state.damage + max(0.0, impact_speed) * 0.012, 0.0, 100.0
        )
