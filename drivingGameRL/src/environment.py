"""Gym-like educational environment without a Gym dependency."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import random

from .circuits import Circuit, TrackProjection, get_circuit
from .math2d import Vec2, clamp, wrap_angle
from .terrain import TerrainKind
from .vehicle import CarBuild, DriverControls, Vehicle


class DrivingAction(IntEnum):
    COAST = 0
    ACCELERATE = 1
    BRAKE = 2
    STEER_LEFT = 3
    STEER_RIGHT = 4


ACTION_CONTROLS = {
    DrivingAction.COAST: DriverControls(),
    DrivingAction.ACCELERATE: DriverControls(throttle=1.0),
    DrivingAction.BRAKE: DriverControls(brake=1.0),
    DrivingAction.STEER_LEFT: DriverControls(throttle=0.72, steering=-1.0),
    DrivingAction.STEER_RIGHT: DriverControls(throttle=0.72, steering=1.0),
}


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]

    def __iter__(self):
        yield self.observation
        yield self.reward
        yield self.terminated
        yield self.truncated
        yield self.info


class DrivingEnv:
    """Fixed-step track environment with observable physics and shaped reward."""

    OBSERVATION_LABELS = (
        "speed",
        "longitudinal_speed",
        "lateral_speed",
        "heading_error",
        "track_offset",
        "terrain_grip",
        "lap_progress",
        "ray_left",
        "ray_left_forward",
        "ray_forward",
        "ray_right_forward",
        "ray_right",
    )

    def __init__(
        self,
        circuit: str | Circuit = "harbor_loop",
        *,
        build: CarBuild | None = None,
        seed: int | None = None,
        fixed_dt: float = 1.0 / 60.0,
        max_steps: int = 60 * 180,
    ):
        if not 0.0 < fixed_dt <= 0.1:
            raise ValueError("fixed_dt must be in the (0, 0.1] interval")
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.circuit = get_circuit(circuit) if isinstance(circuit, str) else circuit
        self.vehicle = Vehicle(build)
        self.fixed_dt = fixed_dt
        self.max_steps = max_steps
        self.random = random.Random(seed)
        self.seed = seed
        self.steps = 0
        self.laps = 0
        self.collisions = 0
        self._collision_contact = False
        self._lap_progress = 0.0
        self.previous_progress = 0.0
        self.last_projection: TrackProjection | None = None
        self.last_reward_terms: dict[str, float] = {}
        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None) -> tuple[float, ...]:
        if seed is not None:
            self.seed = seed
            self.random.seed(seed)
        position, heading = self.circuit.start_pose()
        self.vehicle.reset(position, heading)
        self.steps = 0
        self.laps = 0
        self.collisions = 0
        self._collision_contact = False
        self._lap_progress = 0.0
        self.last_projection = self.circuit.project(position)
        self.previous_progress = self.last_projection.progress
        self.last_reward_terms = {}
        return self.observation()

    def observation(self) -> tuple[float, ...]:
        state = self.vehicle.state
        telemetry = self.vehicle.last_telemetry
        projection = self.circuit.project(state.position)
        track_heading = math.atan2(projection.tangent.y, projection.tangent.x)
        heading_error = wrap_angle(state.heading - track_heading) / math.pi
        max_speed = max(1.0, self.vehicle.build.max_speed)
        sensor_angles = (-math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2)
        rays = tuple(
            self._ray_distance(state.heading + angle) for angle in sensor_angles
        )
        return (
            clamp(telemetry.speed / max_speed, 0.0, 1.5),
            clamp(telemetry.longitudinal_speed / max_speed, -1.0, 1.0),
            clamp(telemetry.lateral_speed / max_speed, -1.0, 1.0),
            heading_error,
            clamp(projection.signed_offset / self.circuit.collision_radius, -1.0, 1.0),
            clamp(self.circuit.terrain_at(state.position).grip, 0.0, 1.0),
            projection.progress,
            *rays,
        )

    def _ray_distance(self, angle: float, max_distance: float = 150.0) -> float:
        origin = self.vehicle.state.position
        direction = Vec2.from_angle(angle)
        step = 6.0
        distance = step
        while distance <= max_distance:
            sample = origin + direction * distance
            if self.circuit.project(sample).distance >= self.circuit.collision_radius:
                return distance / max_distance
            distance += step
        return 1.0

    def step(self, action: int | DrivingAction | DriverControls) -> StepResult:
        if isinstance(action, DriverControls):
            controls = action
        else:
            try:
                controls = ACTION_CONTROLS[DrivingAction(action)]
            except (ValueError, KeyError) as error:
                raise ValueError(f"Invalid driving action: {action!r}") from error
        return self.step_controls(controls)

    def step_controls(self, controls: DriverControls) -> StepResult:
        active_terrain = self.circuit.terrain_at(self.vehicle.state.position)
        telemetry = self.vehicle.step(controls, active_terrain, self.fixed_dt)
        after = self.circuit.project(self.vehicle.state.position)
        penetrated_barrier = after.distance > self.circuit.collision_radius
        impact_speed = self.vehicle.resolve_collision(
            after.point, self.circuit.collision_radius
        )
        collided = penetrated_barrier
        collision_started = collided and not self._collision_contact
        if collision_started:
            self.collisions += 1
            self.vehicle.apply_impact_damage(impact_speed)
        if collided:
            self._collision_contact = True
            after = self.circuit.project(self.vehicle.state.position)
            telemetry = self.vehicle.last_telemetry
        elif after.distance < self.circuit.collision_radius - 4.0:
            self._collision_contact = False

        delta_progress = after.progress - self.previous_progress
        if delta_progress < -0.5:
            delta_progress += 1.0
        elif delta_progress > 0.5:
            delta_progress -= 1.0

        self._lap_progress = max(0.0, self._lap_progress + delta_progress)
        lap_completed = self._lap_progress >= 1.0 - 1e-9
        if lap_completed:
            self.laps += 1
            self._lap_progress = max(0.0, self._lap_progress - 1.0)

        forward_distance = delta_progress * self.circuit.length
        on_road = after.distance <= self.circuit.track_width * 0.5
        reward_terms = {
            "progress": forward_distance * 0.12,
            "road": 0.025 if on_road else -0.08,
            "speed": 0.018
            * max(0.0, telemetry.longitudinal_speed)
            / self.vehicle.build.max_speed,
            "reverse": -0.05 if telemetry.longitudinal_speed < -2.0 else 0.0,
            "collision": (-min(5.0, impact_speed * 0.06) if collision_started else 0.0),
            "lap": 20.0 if lap_completed else 0.0,
        }
        reward = sum(reward_terms.values())
        self.last_reward_terms = reward_terms
        self.steps += 1
        self.previous_progress = after.progress
        self.last_projection = after
        truncated = self.steps >= self.max_steps
        info: dict[str, object] = {
            "circuit": self.circuit.slug,
            "terrain": active_terrain.kind.value,
            "on_road": on_road,
            "progress": after.progress,
            "laps": self.laps,
            "lap_completed": lap_completed,
            "collided": collided,
            "collision_started": collision_started,
            "impact_speed": impact_speed,
            "reward_terms": reward_terms.copy(),
            "telemetry": telemetry,
        }
        return StepResult(self.observation(), reward, False, truncated, info)

    def change_circuit(self, name: str) -> tuple[float, ...]:
        self.circuit = get_circuit(name)
        return self.reset()

    def telemetry(self) -> dict[str, object]:
        """Return a stable, serialization-friendly educational snapshot."""

        state = self.vehicle.state
        physics = self.vehicle.last_telemetry
        projection = self.circuit.project(state.position)
        active_terrain = self.circuit.terrain_at(state.position)
        build = self.vehicle.build
        return {
            "circuit": self.circuit.slug,
            "steps": self.steps,
            "laps": self.laps,
            "collisions": self.collisions,
            "position": (state.position.x, state.position.y),
            "heading_degrees": math.degrees(state.heading),
            "speed": physics.speed,
            "speed_ratio": physics.speed / max(1.0, build.max_speed),
            "longitudinal_speed": physics.longitudinal_speed,
            "lateral_speed": physics.lateral_speed,
            "slip_degrees": math.degrees(physics.slip_angle),
            "acceleration": physics.acceleration,
            "terrain": active_terrain.kind.value,
            "terrain_grip": active_terrain.grip,
            "effective_grip": active_terrain.grip * build.grip_multiplier,
            "track_offset": projection.signed_offset,
            "progress": projection.progress,
            "damage": state.damage,
            "components": {
                "motor": build.motor,
                "wheels": build.wheels,
                "suspension": build.suspension,
                "grip": build.grip,
            },
            "capabilities": {
                "acceleration": build.acceleration,
                "max_speed": build.max_speed,
                "steering_degrees": math.degrees(build.steering_rate),
                "stability": build.stability,
                "grip_multiplier": build.grip_multiplier,
            },
            "reward_terms": self.last_reward_terms.copy(),
        }
