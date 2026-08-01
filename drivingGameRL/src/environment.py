"""Gym-like educational environment without a Gym dependency."""

from __future__ import annotations

from bisect import bisect_right
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


@dataclass(frozen=True, slots=True)
class LapPose:
    """One deterministic sample from a completed best lap."""

    elapsed: float
    position: Vec2
    heading: float


@dataclass(frozen=True, slots=True)
class LapRecord:
    """Fastest in-session lap and the racing line used by its ghost."""

    circuit: str
    duration: float
    trajectory: tuple[LapPose, ...]


@dataclass(frozen=True, slots=True)
class SensorRay:
    """One immutable track-clearance ray used by the driving policy.

    ``angle`` is an absolute world-space angle in radians.  ``distance`` and
    ``endpoint`` describe the first sampled barrier contact, or the full ray
    when ``hit`` is false.  The normalized distance is the exact value placed
    in the neural-network observation.
    """

    angle: float
    max_distance: float
    distance: float
    normalized_distance: float
    origin: Vec2
    endpoint: Vec2
    hit: bool


class DrivingEnv:
    """Fixed-step track environment with observable physics and shaped reward."""

    MAX_GHOST_SAMPLES = 4_096
    GHOST_SAMPLE_INTERVAL = 1.0 / 30.0
    LAP_CHECKPOINTS = (0.25, 0.50, 0.75)
    MAX_LAP_PROGRESS_STEP = 0.075
    BEST_LAP_EPSILON = 1e-9
    SENSOR_MAX_DISTANCE = 150.0
    SENSOR_SAMPLE_STEP = 6.0
    SENSOR_RELATIVE_ANGLES = (
        -math.pi / 2,
        -math.pi / 4,
        0.0,
        math.pi / 4,
        math.pi / 2,
    )

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
        self.current_lap_time = 0.0
        self.last_lap_time: float | None = None
        self._best_laps: dict[str, LapRecord] = {}
        self._current_lap_trajectory: list[LapPose] = []
        self._record_interval = fixed_dt
        self._next_record_time = fixed_dt
        self._next_lap_checkpoint = 0
        self._lap_candidate_armed = True
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
        self.current_lap_time = 0.0
        self.last_lap_time = None
        self._collision_contact = False
        self._lap_progress = 0.0
        self._next_lap_checkpoint = 0
        self._lap_candidate_armed = True
        self.last_projection = self.circuit.project(position)
        self.previous_progress = self.last_projection.progress
        self.last_reward_terms = {}
        self._reset_lap_recording()
        return self.observation()

    @property
    def best_lap_record(self) -> LapRecord | None:
        """Return the current circuit's fastest in-session lap, if available."""

        return self._best_laps.get(self.circuit.slug)

    @property
    def best_lap_time(self) -> float | None:
        record = self.best_lap_record
        return None if record is None else record.duration

    @property
    def best_lap_trajectory(self) -> tuple[LapPose, ...]:
        record = self.best_lap_record
        return () if record is None else record.trajectory

    @property
    def current_trajectory_samples(self) -> int:
        return len(self._current_lap_trajectory)

    def _lap_pose(self, elapsed: float) -> LapPose:
        state = self.vehicle.state
        return LapPose(elapsed, state.position, state.heading)

    def _reset_lap_recording(self) -> None:
        self._record_interval = max(self.fixed_dt, self.GHOST_SAMPLE_INTERVAL)
        self._next_record_time = self._record_interval
        self._current_lap_trajectory = [self._lap_pose(0.0)]

    def restart_lap_candidate(self, *, wait_for_start: bool = False) -> None:
        """Invalidate partial timing without deleting any completed best lap.

        Component changes use ``wait_for_start=True`` so a mid-lap upgrade
        cannot turn the remaining fraction of the circuit into a best time.
        """

        self.current_lap_time = 0.0
        self._lap_progress = 0.0
        self._next_lap_checkpoint = 0
        projection = self.circuit.project(self.vehicle.state.position)
        exactly_on_start = min(projection.progress, 1.0 - projection.progress) <= 1e-9
        self._lap_candidate_armed = not wait_for_start or exactly_on_start
        self.previous_progress = projection.progress
        self.last_projection = projection
        self._reset_lap_recording()

    def _record_lap_pose(self, *, force: bool = False) -> None:
        if not force and self.current_lap_time + 1e-12 < self._next_record_time:
            return
        if len(self._current_lap_trajectory) >= self.MAX_GHOST_SAMPLES:
            # Deterministically halve temporal resolution instead of dropping
            # the beginning of a long lap. This keeps both memory and render
            # work bounded while retaining the complete racing line.
            self._current_lap_trajectory = self._current_lap_trajectory[::2]
            self._record_interval *= 2.0
        pose = self._lap_pose(self.current_lap_time)
        if self._current_lap_trajectory[-1].elapsed == pose.elapsed:
            self._current_lap_trajectory[-1] = pose
        else:
            self._current_lap_trajectory.append(pose)
        self._next_record_time = self.current_lap_time + self._record_interval

    def ghost_pose_at(self, elapsed: float | None = None) -> LapPose | None:
        """Interpolate the best-lap ghost at deterministic simulation time."""

        record = self.best_lap_record
        if record is None:
            return None
        if elapsed is None and not self._lap_candidate_armed:
            return None
        target = self.current_lap_time if elapsed is None else elapsed
        if not math.isfinite(target) or target < 0.0:
            raise ValueError("Ghost elapsed time must be finite and non-negative")
        if target > record.duration:
            return None
        trajectory = record.trajectory
        index = bisect_right(trajectory, target, key=lambda pose: pose.elapsed)
        if index <= 0:
            first = trajectory[0]
            return LapPose(target, first.position, first.heading)
        if index >= len(trajectory):
            last = trajectory[-1]
            return LapPose(target, last.position, last.heading)
        before = trajectory[index - 1]
        after = trajectory[index]
        span = after.elapsed - before.elapsed
        blend = 0.0 if span <= 1e-12 else (target - before.elapsed) / span
        position = before.position + (after.position - before.position) * blend
        heading_delta = wrap_angle(after.heading - before.heading)
        heading = wrap_angle(before.heading + heading_delta * blend)
        return LapPose(target, position, heading)

    def observation(self) -> tuple[float, ...]:
        state = self.vehicle.state
        telemetry = self.vehicle.last_telemetry
        projection = self.circuit.project(state.position)
        track_heading = math.atan2(projection.tangent.y, projection.tangent.x)
        heading_error = wrap_angle(state.heading - track_heading) / math.pi
        max_speed = max(1.0, self.vehicle.build.max_speed)
        rays = self.sensor_rays()
        return (
            clamp(telemetry.speed / max_speed, 0.0, 1.5),
            clamp(telemetry.longitudinal_speed / max_speed, -1.0, 1.0),
            clamp(telemetry.lateral_speed / max_speed, -1.0, 1.0),
            heading_error,
            clamp(projection.signed_offset / self.circuit.collision_radius, -1.0, 1.0),
            clamp(self.circuit.terrain_at(state.position).grip, 0.0, 1.0),
            projection.progress,
            *(ray.normalized_distance for ray in rays),
        )

    def sensor_rays(
        self, max_distance: float = SENSOR_MAX_DISTANCE
    ) -> tuple[SensorRay, ...]:
        """Return the five policy sensor rays in observation-label order.

        The tuple and its :class:`SensorRay` entries are immutable snapshots.
        Rendering can therefore consume the same distances as the policy
        without being able to alter environment state or drift from the
        observation contract.
        """

        if (
            isinstance(max_distance, bool)
            or not isinstance(max_distance, (int, float))
            or not math.isfinite(max_distance)
            or max_distance <= 0.0
        ):
            raise ValueError("max_distance must be finite and positive")
        maximum = float(max_distance)
        heading = self.vehicle.state.heading
        return tuple(
            self._sensor_ray(heading + relative_angle, maximum)
            for relative_angle in self.SENSOR_RELATIVE_ANGLES
        )

    def _sensor_ray(self, angle: float, max_distance: float) -> SensorRay:
        origin = self.vehicle.state.position
        direction = Vec2.from_angle(angle)
        distance = self.SENSOR_SAMPLE_STEP
        while distance <= max_distance:
            sample = origin + direction * distance
            if self.circuit.project(sample).distance >= self.circuit.collision_radius:
                return SensorRay(
                    angle=angle,
                    max_distance=max_distance,
                    distance=distance,
                    normalized_distance=distance / max_distance,
                    origin=origin,
                    endpoint=sample,
                    hit=True,
                )
            distance += self.SENSOR_SAMPLE_STEP
        endpoint = origin + direction * max_distance
        return SensorRay(
            angle=angle,
            max_distance=max_distance,
            distance=max_distance,
            normalized_distance=1.0,
            origin=origin,
            endpoint=endpoint,
            hit=False,
        )

    def _ray_distance(
        self, angle: float, max_distance: float = SENSOR_MAX_DISTANCE
    ) -> float:
        """Compatibility helper returning only a ray's normalized distance."""

        if (
            isinstance(max_distance, bool)
            or not isinstance(max_distance, (int, float))
            or not math.isfinite(max_distance)
            or max_distance <= 0.0
        ):
            raise ValueError("max_distance must be finite and positive")
        return self._sensor_ray(angle, float(max_distance)).normalized_distance

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

        raw_delta_progress = after.progress - self.previous_progress
        delta_progress = raw_delta_progress
        if delta_progress < -0.5:
            delta_progress += 1.0
        elif delta_progress > 0.5:
            delta_progress -= 1.0

        valid_forward_progress = 0.0 < delta_progress <= self.MAX_LAP_PROGRESS_STEP
        forward_start_crossed = valid_forward_progress and raw_delta_progress < -0.5
        progress_discontinuity = abs(delta_progress) > self.MAX_LAP_PROGRESS_STEP
        if self._lap_candidate_armed and progress_discontinuity:
            # A fixed-step car cannot legitimately traverse this much of the
            # center line at once. Projection switches between nearby hairpin
            # segments therefore invalidate, rather than shorten, the lap.
            self.restart_lap_candidate(wait_for_start=True)
        if self._lap_candidate_armed:
            self.current_lap_time += self.fixed_dt
            if abs(delta_progress) <= self.MAX_LAP_PROGRESS_STEP:
                self._lap_progress = clamp(
                    self._lap_progress + delta_progress, 0.0, 1.0
                )
            if valid_forward_progress and not forward_start_crossed:
                checkpoint = (
                    self.LAP_CHECKPOINTS[self._next_lap_checkpoint]
                    if self._next_lap_checkpoint < len(self.LAP_CHECKPOINTS)
                    else None
                )
                if (
                    checkpoint is not None
                    and self.previous_progress < checkpoint <= after.progress
                ):
                    self._next_lap_checkpoint += 1

        lap_completed = (
            forward_start_crossed
            and self._lap_candidate_armed
            and self._next_lap_checkpoint == len(self.LAP_CHECKPOINTS)
            and self._lap_progress >= 1.0 - 1e-9
        )
        if lap_completed:
            self._record_lap_pose(force=True)
            self.laps += 1
            self.last_lap_time = self.current_lap_time
            previous_best = self.best_lap_record
            if (
                previous_best is None
                or self.last_lap_time < previous_best.duration - self.BEST_LAP_EPSILON
            ):
                self._best_laps[self.circuit.slug] = LapRecord(
                    self.circuit.slug,
                    self.last_lap_time,
                    tuple(self._current_lap_trajectory),
                )
            self.current_lap_time = 0.0
            self._lap_progress = 0.0
            self._next_lap_checkpoint = 0
            self._reset_lap_recording()
        elif forward_start_crossed:
            # Crossing the line without all ordered gates rejects shortcuts
            # and starts a clean candidate for the following full lap.
            self.current_lap_time = 0.0
            self._lap_progress = 0.0
            self._next_lap_checkpoint = 0
            self._lap_candidate_armed = True
            self._reset_lap_recording()
        else:
            if self._lap_candidate_armed:
                self._record_lap_pose()

        reward_progress = (
            delta_progress if abs(delta_progress) <= self.MAX_LAP_PROGRESS_STEP else 0.0
        )
        forward_distance = reward_progress * self.circuit.length
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
            "current_lap_time": self.current_lap_time,
            "last_lap_time": self.last_lap_time,
            "best_lap_time": self.best_lap_time,
            "lap_candidate_valid": self._lap_candidate_armed,
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
            "current_lap_time": self.current_lap_time,
            "last_lap_time": self.last_lap_time,
            "best_lap_time": self.best_lap_time,
            "lap_candidate_valid": self._lap_candidate_armed,
            "ghost_available": self.best_lap_record is not None,
            "ghost_recording_samples": len(self._current_lap_trajectory),
            "best_trajectory_samples": len(self.best_lap_trajectory),
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
