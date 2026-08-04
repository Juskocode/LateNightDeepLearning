"""Gym-like educational environment without a Gym dependency."""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from collections.abc import Mapping
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
    # One discrete action remains useful on both sides of zero speed: it first
    # arrests forward motion, then becomes a deliberate reverse escape.  A
    # small brake component keeps the control readable as a brake without
    # overpowering the negative throttle once the car has stopped.
    DrivingAction.BRAKE: DriverControls(throttle=-1.0, brake=0.15),
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
    ORDERED_CHECKPOINT_REWARD = 15.0
    LAP_COMPLETION_REWARD = 300.0
    LAP_TIME_BONUS_MAX = 150.0
    LAP_TIME_MINIMUM_RATIO = 0.75
    MAX_LAP_TARGET = 12
    MAX_LAP_PROGRESS_STEP = 0.075
    BEST_LAP_EPSILON = 1e-9
    SENSOR_MAX_DISTANCE = 150.0
    SENSOR_SAMPLE_STEP = 6.0
    SENSOR_REFINEMENT_STEPS = 4
    STAGNATION_GRACE_STEPS = 90
    STAGNATION_LIMIT_STEPS = 240
    STAGNATION_PROGRESS_DISTANCE = 0.04
    CLEARANCE_USABLE_FLOOR = 0.18
    CLEARANCE_GREEN_THRESHOLD = 0.55
    CLEARANCE_FRONT_SHARE = 0.65
    CLEARANCE_RAY_WEIGHTS = (
        0.15,
        0.25,
        0.45,
        0.75,
        1.0,
        0.75,
        0.45,
        0.25,
        0.15,
    )
    CLEARANCE_GAIN_REWARD_SCALE = 6.0
    GREEN_DENSITY_REWARD_SCALE = 0.025
    CLEARANCE_CLOSING_PENALTY_SCALE = 32.0
    CLEARANCE_HAZARD_THRESHOLD = 0.42
    CLEARANCE_HAZARD_PENALTY_SCALE = 0.55
    WALL_CONTACT_PENALTY = 1.25
    COLLISION_ENTRY_PENALTY = 6.0
    COLLISION_IMPACT_PENALTY_SCALE = 0.12
    COLLISION_IMPACT_PENALTY_CAP = 12.0
    COLLISION_RECOVERY_CONFIRM_STEPS = 12
    COLLISION_RECOVERY_TIMEOUT_STEPS = 45
    # Compatibility name retained for dashboards and callers written before
    # collision recovery became an explicit state machine.
    WALL_CONTACT_TRUNCATION_STEPS = COLLISION_RECOVERY_TIMEOUT_STEPS
    COLLISION_LOOP_ENTRY_LIMIT = 4
    COLLISION_LOOP_WINDOW_STEPS = 180
    NORMAL_START_PROBABILITY = 0.80
    SENSOR_RELATIVE_ANGLES = (
        -math.pi / 2,
        -3 * math.pi / 8,
        -math.pi / 4,
        -math.pi / 8,
        0.0,
        math.pi / 8,
        math.pi / 4,
        3 * math.pi / 8,
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
        "ray_left_wide",
        "ray_left_forward",
        "ray_left_near",
        "ray_forward",
        "ray_right_near",
        "ray_right_forward",
        "ray_right_wide",
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
        random_start_curriculum: bool = False,
        lap_target: int = 1,
    ):
        if not 0.0 < fixed_dt <= 0.1:
            raise ValueError("fixed_dt must be in the (0, 0.1] interval")
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(random_start_curriculum, bool):
            raise ValueError("random_start_curriculum must be a boolean")
        self._validate_lap_target(lap_target)
        self.circuit = get_circuit(circuit) if isinstance(circuit, str) else circuit
        self.vehicle = Vehicle(build)
        self.fixed_dt = fixed_dt
        self.max_steps = max_steps
        self.random = random.Random(seed)
        self.seed = seed
        self.random_start_curriculum = random_start_curriculum
        self._lap_target = lap_target
        self._curriculum_unlocked = False
        self._spawn_mode = "start_line"
        self._spawn_progress = 0.0
        self._lap_origin_progress = 0.0
        self._episode_lap_progress = 0.0
        self._max_episode_lap_progress = 0.0
        self._episode_target_progress = 0.0
        self._max_episode_target_progress = 0.0
        self._last_curriculum_lap_completed = False
        self._last_lap_target_completed = False
        self.steps = 0
        self.laps = 0
        self.collisions = 0
        self.current_lap_time = 0.0
        self.last_lap_time: float | None = None
        self._episode_lap_times: list[float] = []
        self._episode_lap_time_bonus_total = 0.0
        self._last_lap_time_bonus = 0.0
        self._last_lap_time_bonus_valid = False
        self._best_laps: dict[str, LapRecord] = {}
        self._current_lap_trajectory: list[LapPose] = []
        self._record_interval = fixed_dt
        self._next_record_time = fixed_dt
        self._next_lap_checkpoint = 0
        self._lap_candidate_armed = True
        self._collision_contact = False
        self._lap_progress = 0.0
        self.previous_progress = 0.0
        self._stagnation_steps = 0
        self._wall_contact_active = False
        self._wall_contact_steps = 0
        self._collision_entry_steps: deque[int] = deque()
        self._recent_collision_entries = 0
        self._steps_since_collision = self.COLLISION_LOOP_WINDOW_STEPS
        self._collision_recovery_active = False
        self._collision_recovery_steps = 0
        self._collision_recovery_clean_steps = 0
        self._collision_recoveries = 0
        self._collision_pressure = 0.0
        self._collision_looped = False
        self._last_truncation_reason: str | None = None
        self._usable_clearance = 0.0
        self._previous_usable_clearance = 0.0
        self._clearance_delta = 0.0
        self._green_ray_fraction = 0.0
        self._wall_closing = False
        self._clearance_motion_ratio = 0.0
        self.last_projection: TrackProjection | None = None
        self.last_reward_terms: dict[str, float] = {}
        # Observation, telemetry, and rendering often request the same rays in
        # one frame. Track projection is comparatively expensive, so retain
        # the immutable tuple for the current pose instead of resampling it.
        # A pose-derived key makes movement and circuit swaps self-invalidating.
        self._sensor_ray_cache_key: tuple[object, ...] | None = None
        self._sensor_ray_cache: tuple[SensorRay, ...] = ()
        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None) -> tuple[float, ...]:
        if seed is not None:
            self.seed = seed
            self.random.seed(seed)
        use_start_line = not self.random_start_curriculum or (
            self._curriculum_unlocked
            and self.random.random() < self.NORMAL_START_PROBABILITY
        )
        if use_start_line:
            position, heading = self.circuit.start_pose()
            self._spawn_mode = "start_line"
        else:
            sampled_progress = self.random.random()
            position, tangent = self.circuit.point_tangent_at(sampled_progress)
            heading = math.atan2(tangent.y, tangent.x)
            self._spawn_mode = "random_track"
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
        self._stagnation_steps = 0
        self._wall_contact_active = False
        self._wall_contact_steps = 0
        self._collision_entry_steps.clear()
        self._recent_collision_entries = 0
        self._steps_since_collision = self.COLLISION_LOOP_WINDOW_STEPS
        self._collision_recovery_active = False
        self._collision_recovery_steps = 0
        self._collision_recovery_clean_steps = 0
        self._collision_recoveries = 0
        self._collision_pressure = 0.0
        self._collision_looped = False
        self._last_truncation_reason = None
        self._spawn_progress = self.last_projection.progress
        self._lap_origin_progress = self.last_projection.progress
        self._episode_lap_progress = 0.0
        self._max_episode_lap_progress = 0.0
        self._episode_target_progress = 0.0
        self._max_episode_target_progress = 0.0
        self._last_curriculum_lap_completed = False
        self._last_lap_target_completed = False
        self._episode_lap_times = []
        self._episode_lap_time_bonus_total = 0.0
        self._last_lap_time_bonus = 0.0
        self._last_lap_time_bonus_valid = False
        self.last_reward_terms = {}
        self._reset_lap_recording()
        self._usable_clearance, self._green_ray_fraction = self._clearance_snapshot(
            self.sensor_rays()
        )
        self._previous_usable_clearance = self._usable_clearance
        self._clearance_delta = 0.0
        self._wall_closing = False
        self._clearance_motion_ratio = 0.0
        return self.observation()

    @property
    def curriculum_unlocked(self) -> bool:
        """Whether a curriculum car has already proved it can finish a loop."""

        return self._curriculum_unlocked

    @property
    def curriculum_ready(self) -> bool:
        """Compatibility alias for :attr:`curriculum_unlocked`."""

        return self._curriculum_unlocked

    @property
    def normal_start_probability(self) -> float:
        """Chance of a start-line spawn after unlocking the curriculum."""

        return self.NORMAL_START_PROBABILITY

    @classmethod
    def _validate_lap_target(cls, value: object) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= cls.MAX_LAP_TARGET
        ):
            raise ValueError(
                f"lap_target must be an integer in [1, {cls.MAX_LAP_TARGET}]"
            )
        return value

    @property
    def lap_target(self) -> int:
        """Number of valid loops required to finish a learning evaluation."""

        return self._lap_target

    def set_lap_target(self, value: int) -> None:
        """Set the target at an episode or generation barrier.

        The method deliberately does not reset the car so session owners can
        update the target and timeout atomically before their next ``reset``.
        """

        self._lap_target = self._validate_lap_target(value)

    @property
    def lap_time_reference(self) -> float:
        """Optimistic center-line time used to normalize the pace bonus."""

        return self.circuit.length / max(1.0, self.vehicle.build.max_speed)

    def lap_time_bonus(self, duration: float) -> tuple[float, bool]:
        """Return a bounded bonus for one physically plausible valid lap."""

        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0.0
        ):
            raise ValueError("lap duration must be finite and positive")
        duration = float(duration)
        reference = self.lap_time_reference
        if duration + 1e-12 < reference * self.LAP_TIME_MINIMUM_RATIO:
            return 0.0, False
        bonus = self.LAP_TIME_BONUS_MAX * clamp(reference / duration, 0.0, 1.0)
        return bonus, True

    @property
    def spawn_mode(self) -> str:
        return self._spawn_mode

    @property
    def spawn_progress(self) -> float:
        return self._spawn_progress

    @property
    def lap_origin_progress(self) -> float:
        return self._lap_origin_progress

    def curriculum_state(self) -> dict[str, object]:
        """Return the small persistent state needed by checkpoints/clones."""

        return {
            "unlocked": self._curriculum_unlocked,
            "lap_target": self._lap_target,
        }

    def load_curriculum_state(self, state: Mapping[str, object]) -> None:
        """Restore curriculum progress without changing the current episode.

        An empty mapping is accepted for checkpoints written before the
        curriculum existed.  Seeding and resetting deliberately do not clear
        this flag; callers can explicitly load ``{}`` when they need a fresh
        curriculum.
        """

        if not isinstance(state, Mapping):
            raise ValueError("curriculum state must be a mapping")
        value = state.get("unlocked", state.get("ready", False))
        if not isinstance(value, bool):
            raise ValueError("curriculum unlocked state must be a boolean")
        lap_target = self._validate_lap_target(
            state.get("lap_target", self._lap_target)
        )
        self._curriculum_unlocked = value
        self._lap_target = lap_target

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
        Here "start" means this episode's lap origin: normally the official
        line, or the sampled centerline pose in the learning curriculum.
        """

        self.current_lap_time = 0.0
        self._lap_progress = 0.0
        self._episode_lap_progress = 0.0
        self._next_lap_checkpoint = 0
        projection = self.circuit.project(self.vehicle.state.position)
        distance_from_origin = abs(projection.progress - self._lap_origin_progress)
        exactly_on_origin = min(
            distance_from_origin, 1.0 - distance_from_origin
        ) <= 1e-9
        self._lap_candidate_armed = not wait_for_start or exactly_on_origin
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
        """Return the nine policy sensor rays in observation-label order.

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
        state = self.vehicle.state
        heading = state.heading
        cache_key: tuple[object, ...] = (
            id(self.circuit),
            state.position.x,
            state.position.y,
            heading,
            maximum,
            float(self.SENSOR_SAMPLE_STEP),
            int(self.SENSOR_REFINEMENT_STEPS),
            float(self.circuit.collision_radius),
        )
        if cache_key == self._sensor_ray_cache_key:
            return self._sensor_ray_cache
        rays = self._sensor_ray_fan(
            tuple(
                heading + relative_angle
                for relative_angle in self.SENSOR_RELATIVE_ANGLES
            ),
            maximum,
        )
        self._sensor_ray_cache_key = cache_key
        self._sensor_ray_cache = rays
        return rays

    def _sensor_ray(self, angle: float, max_distance: float) -> SensorRay:
        return self._sensor_ray_fan((angle,), max_distance)[0]

    def _sensor_ray_fan(
        self, angles: tuple[float, ...], max_distance: float
    ) -> tuple[SensorRay, ...]:
        """Batch all coarse samples and contact refinements for one ray fan."""

        origin = self.vehicle.state.position
        directions = tuple(Vec2.from_angle(angle) for angle in angles)
        sample_distances = [
            self.SENSOR_SAMPLE_STEP * index
            for index in range(
                1, int(max_distance // self.SENSOR_SAMPLE_STEP) + 1
            )
        ]
        if not sample_distances or sample_distances[-1] < max_distance - 1e-12:
            sample_distances.append(max_distance)
        samples = tuple(
            origin + direction * distance
            for direction in directions
            for distance in sample_distances
        )
        centerline_distances = self.circuit.distances_to_centerline(samples)
        samples_per_ray = len(sample_distances)
        hit_intervals: dict[int, tuple[float, float]] = {}
        for ray_index in range(len(angles)):
            start = ray_index * samples_per_ray
            readings = centerline_distances[start : start + samples_per_ray]
            for sample_index, clearance in enumerate(readings):
                if clearance >= self.circuit.collision_radius:
                    low = (
                        0.0
                        if sample_index == 0
                        else sample_distances[sample_index - 1]
                    )
                    hit_intervals[ray_index] = (low, sample_distances[sample_index])
                    break

        for _ in range(self.SENSOR_REFINEMENT_STEPS):
            if not hit_intervals:
                break
            ray_indices = tuple(hit_intervals)
            midpoints = tuple(
                (hit_intervals[index][0] + hit_intervals[index][1]) * 0.5
                for index in ray_indices
            )
            refinement_samples = tuple(
                origin + directions[index] * midpoint
                for index, midpoint in zip(ray_indices, midpoints)
            )
            refinements = self.circuit.distances_to_centerline(refinement_samples)
            for index, midpoint, clearance in zip(
                ray_indices, midpoints, refinements
            ):
                low, high = hit_intervals[index]
                if clearance >= self.circuit.collision_radius:
                    high = midpoint
                else:
                    low = midpoint
                hit_intervals[index] = (low, high)

        rays: list[SensorRay] = []
        for index, (angle, direction) in enumerate(zip(angles, directions)):
            interval = hit_intervals.get(index)
            hit = interval is not None
            distance = interval[1] if interval is not None else max_distance
            rays.append(
                SensorRay(
                    angle=angle,
                    max_distance=max_distance,
                    distance=distance,
                    normalized_distance=distance / max_distance,
                    origin=origin,
                    endpoint=origin + direction * distance,
                    hit=hit,
                )
            )
        return tuple(rays)

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

    @classmethod
    def _clearance_snapshot(
        cls, rays: tuple[SensorRay, ...]
    ) -> tuple[float, float]:
        """Compress the full fan into usable clearance and green-ray density.

        Clearance below ``CLEARANCE_USABLE_FLOOR`` is not useful escape room.
        The front three rays dominate the objective, while every side ray
        still contributes to choosing a direction around an approaching wall.
        """

        if len(rays) != len(cls.CLEARANCE_RAY_WEIGHTS):
            raise RuntimeError("clearance weights must match the policy ray fan")
        normalized = tuple(
            clamp(ray.normalized_distance, 0.0, 1.0) for ray in rays
        )
        usable = tuple(
            clamp(
                (reading - cls.CLEARANCE_USABLE_FLOOR)
                / (1.0 - cls.CLEARANCE_USABLE_FLOOR),
                0.0,
                1.0,
            )
            for reading in normalized
        )
        weighted_clearance = sum(
            weight * reading
            for weight, reading in zip(cls.CLEARANCE_RAY_WEIGHTS, usable)
        ) / sum(cls.CLEARANCE_RAY_WEIGHTS)
        middle = len(usable) // 2
        forward_clearance = min(usable[middle - 1 : middle + 2])
        combined = (
            cls.CLEARANCE_FRONT_SHARE * forward_clearance
            + (1.0 - cls.CLEARANCE_FRONT_SHARE) * weighted_clearance
        )
        green_fraction = sum(
            reading >= cls.CLEARANCE_GREEN_THRESHOLD for reading in normalized
        ) / len(normalized)
        return clamp(combined, 0.0, 1.0), green_fraction

    @classmethod
    def _clearance_objective_config(
        cls,
    ) -> dict[str, float | int | tuple[float, ...]]:
        """Expose the exact educational reward and cutoff constants."""

        return {
            "usable_floor": cls.CLEARANCE_USABLE_FLOOR,
            "green_threshold": cls.CLEARANCE_GREEN_THRESHOLD,
            "front_share": cls.CLEARANCE_FRONT_SHARE,
            "ray_weights": cls.CLEARANCE_RAY_WEIGHTS,
            "gain_scale": cls.CLEARANCE_GAIN_REWARD_SCALE,
            "green_density_scale": cls.GREEN_DENSITY_REWARD_SCALE,
            "closing_penalty_scale": cls.CLEARANCE_CLOSING_PENALTY_SCALE,
            "hazard_threshold": cls.CLEARANCE_HAZARD_THRESHOLD,
            "hazard_penalty_scale": cls.CLEARANCE_HAZARD_PENALTY_SCALE,
            "wall_contact_penalty": cls.WALL_CONTACT_PENALTY,
            "collision_entry_penalty": cls.COLLISION_ENTRY_PENALTY,
            "collision_impact_scale": cls.COLLISION_IMPACT_PENALTY_SCALE,
            "collision_impact_cap": cls.COLLISION_IMPACT_PENALTY_CAP,
            "wall_contact_limit": cls.COLLISION_RECOVERY_TIMEOUT_STEPS,
            "collision_entry_limit": cls.COLLISION_LOOP_ENTRY_LIMIT,
            "collision_window_steps": cls.COLLISION_LOOP_WINDOW_STEPS,
            "recovery_confirm_steps": cls.COLLISION_RECOVERY_CONFIRM_STEPS,
            "recovery_timeout_steps": cls.COLLISION_RECOVERY_TIMEOUT_STEPS,
        }

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
        self._wall_contact_active = collided
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

        # Collision-entry latching prevents one scrape from being counted as
        # many impacts.  Recovery pressure is intentionally different: only a
        # tick that physically penetrated the barrier contributes contact cost.
        if collided:
            self._wall_contact_steps += 1
            self._collision_recovery_active = True
            self._collision_recovery_clean_steps = 0
        if self._collision_recovery_active:
            self._collision_recovery_steps += 1

        # Keep exact entry timestamps rather than a reset-on-idle counter. The
        # active set is therefore a true sliding window: an entry expires as
        # soon as it is 180 simulation ticks old, independent of later hits.
        collision_step = self.steps
        while (
            self._collision_entry_steps
            and collision_step - self._collision_entry_steps[0]
            >= self.COLLISION_LOOP_WINDOW_STEPS
        ):
            self._collision_entry_steps.popleft()
        if collision_started:
            self._collision_entry_steps.append(collision_step)
        self._recent_collision_entries = len(self._collision_entry_steps)
        if self._collision_entry_steps:
            self._steps_since_collision = (
                collision_step - self._collision_entry_steps[-1]
            )
        else:
            self._steps_since_collision = self.COLLISION_LOOP_WINDOW_STEPS

        raw_delta_progress = after.progress - self.previous_progress
        delta_progress = raw_delta_progress
        if delta_progress < -0.5:
            delta_progress += 1.0
        elif delta_progress > 0.5:
            delta_progress -= 1.0

        valid_forward_progress = 0.0 < delta_progress <= self.MAX_LAP_PROGRESS_STEP
        ordered_checkpoint_advanced = False
        relative_previous = (
            self.previous_progress - self._lap_origin_progress
        ) % 1.0
        relative_after = (after.progress - self._lap_origin_progress) % 1.0
        forward_lap_origin_crossed = (
            valid_forward_progress
            and relative_after + 1e-12 < relative_previous
        )
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
            if valid_forward_progress and not forward_lap_origin_crossed:
                checkpoint = (
                    self.LAP_CHECKPOINTS[self._next_lap_checkpoint]
                    if self._next_lap_checkpoint < len(self.LAP_CHECKPOINTS)
                    else None
                )
                if (
                    checkpoint is not None
                    and relative_previous < checkpoint <= relative_after
                ):
                    self._next_lap_checkpoint += 1
                    ordered_checkpoint_advanced = True

        lap_completed = (
            forward_lap_origin_crossed
            and self._lap_candidate_armed
            and self._next_lap_checkpoint == len(self.LAP_CHECKPOINTS)
            and self._lap_progress >= 1.0 - 1e-9
        )
        lap_time_bonus = 0.0
        lap_time_bonus_valid = False
        if lap_completed:
            self._record_lap_pose(force=True)
            self.laps += 1
            self.last_lap_time = self.current_lap_time
            self._episode_lap_times.append(self.last_lap_time)
            lap_time_bonus, lap_time_bonus_valid = self.lap_time_bonus(
                self.last_lap_time
            )
            self._episode_lap_time_bonus_total += lap_time_bonus
            self._last_lap_time_bonus = lap_time_bonus
            self._last_lap_time_bonus_valid = lap_time_bonus_valid
            previous_best = self.best_lap_record
            if self._spawn_mode == "start_line" and (
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
        elif forward_lap_origin_crossed:
            # Crossing this episode's origin without all ordered gates rejects
            # shortcuts and starts a clean candidate for the following loop.
            self.current_lap_time = 0.0
            self._lap_progress = 0.0
            self._next_lap_checkpoint = 0
            self._lap_candidate_armed = True
            self._reset_lap_recording()
        else:
            if self._lap_candidate_armed:
                self._record_lap_pose()

        curriculum_lap_completed = self.random_start_curriculum and lap_completed
        lap_target_completed = (
            curriculum_lap_completed and self.laps >= self._lap_target
        )
        episode_lap_progress = 1.0 if lap_completed else self._lap_progress
        self._episode_lap_progress = episode_lap_progress
        self._max_episode_lap_progress = max(
            self._max_episode_lap_progress,
            episode_lap_progress,
        )
        self._episode_target_progress = clamp(
            (
                self.laps
                + (0.0 if lap_completed else self._lap_progress)
            )
            / self._lap_target,
            0.0,
            1.0,
        )
        self._max_episode_target_progress = max(
            self._max_episode_target_progress,
            self._episode_target_progress,
        )
        if curriculum_lap_completed:
            self._curriculum_unlocked = True
        self._last_curriculum_lap_completed = curriculum_lap_completed
        self._last_lap_target_completed = lap_target_completed

        reward_progress = (
            delta_progress if abs(delta_progress) <= self.MAX_LAP_PROGRESS_STEP else 0.0
        )
        forward_distance = reward_progress * self.circuit.length
        if forward_distance >= self.STAGNATION_PROGRESS_DISTANCE:
            self._stagnation_steps = 0
        else:
            self._stagnation_steps += 1

        on_road = after.distance <= self.circuit.track_width * 0.5
        meaningful_forward_recovery = (
            self._collision_recovery_active
            and not collided
            and on_road
            and forward_distance >= self.STAGNATION_PROGRESS_DISTANCE
        )
        if meaningful_forward_recovery:
            self._collision_recovery_clean_steps += 1
        elif self._collision_recovery_active:
            self._collision_recovery_clean_steps = 0

        if (
            self._collision_recovery_active
            and self._collision_recovery_clean_steps
            >= self.COLLISION_RECOVERY_CONFIRM_STEPS
        ):
            # A stable return to forward road motion ends the incident without
            # touching ordered lap gates, timers, or accumulated lap progress.
            self._collision_recoveries += 1
            self._collision_recovery_active = False
            self._collision_recovery_steps = 0
            self._collision_recovery_clean_steps = 0
            self._wall_contact_steps = 0
            self._collision_entry_steps.clear()
            self._recent_collision_entries = 0
            self._steps_since_collision = self.COLLISION_LOOP_WINDOW_STEPS
            self._collision_contact = False

        if self._collision_recovery_active:
            contact_pressure = (
                self._wall_contact_steps / self.COLLISION_RECOVERY_TIMEOUT_STEPS
            )
            entry_pressure = (
                self._recent_collision_entries / self.COLLISION_LOOP_ENTRY_LIMIT
            )
            self._collision_pressure = clamp(
                max(contact_pressure, entry_pressure), 0.0, 1.0
            )
        else:
            self._collision_pressure = 0.0
        self._collision_looped = (
            self._collision_recovery_active and self._collision_pressure >= 1.0
        )
        max_speed = max(1.0, self.vehicle.build.max_speed)
        forward_speed_ratio = clamp(
            max(0.0, telemetry.longitudinal_speed) / max_speed, 0.0, 1.0
        )
        reverse_speed_ratio = clamp(
            max(0.0, -telemetry.longitudinal_speed) / max_speed, 0.0, 1.0
        )
        track_heading = math.atan2(after.tangent.y, after.tangent.x)
        heading_alignment = math.cos(
            wrap_angle(self.vehicle.state.heading - track_heading)
        )
        forward_alignment = max(0.0, heading_alignment)
        road_half_width = max(1.0, self.circuit.track_width * 0.5)
        offset_ratio = clamp(after.distance / road_half_width, 0.0, 2.0)
        center_factor = clamp(1.0 - offset_ratio, 0.0, 1.0)

        rays = self.sensor_rays()
        middle = len(rays) // 2
        forward_clearance = min(
            ray.normalized_distance for ray in rays[middle - 1 : middle + 2]
        )
        clearance_hazard = clamp(
            (self.CLEARANCE_HAZARD_THRESHOLD - forward_clearance)
            / self.CLEARANCE_HAZARD_THRESHOLD,
            0.0,
            1.0,
        )
        previous_usable_clearance = self._usable_clearance
        usable_clearance, green_ray_fraction = self._clearance_snapshot(rays)
        clearance_delta = usable_clearance - previous_usable_clearance
        motion_ratio = clamp(telemetry.speed / max_speed, 0.0, 1.0)
        clearance_gain = (
            max(0.0, clearance_delta)
            * self.CLEARANCE_GAIN_REWARD_SCALE
            * motion_ratio
        )
        closing_amount = max(0.0, -clearance_delta)
        wall_closing_penalty = -(
            closing_amount
            * self.CLEARANCE_CLOSING_PENALTY_SCALE
            * (0.35 + 0.65 * motion_ratio)
        )
        self._previous_usable_clearance = previous_usable_clearance
        self._usable_clearance = usable_clearance
        self._clearance_delta = clearance_delta
        self._green_ray_fraction = green_ray_fraction
        self._wall_closing = closing_amount > 1e-9
        self._clearance_motion_ratio = motion_ratio
        productive_speed = (
            forward_speed_ratio
            * forward_alignment
            * center_factor
            * forward_clearance
        )
        # This small state reward distinguishes a broadly open green fan from
        # the same front corridor with blocked side escape routes. Forward
        # motion, alignment, and centering gate it, so a parked car earns zero;
        # its 0.025 ceiling also leaves signed progress as the dominant signal.
        green_clearance_reward = (
            self.GREEN_DENSITY_REWARD_SCALE
            * green_ray_fraction
            * forward_speed_ratio
            * forward_alignment
            * center_factor
        )
        stagnation_excess = max(
            0, self._stagnation_steps - self.STAGNATION_GRACE_STEPS
        )
        reward_terms = {
            # Normalized center-line progress is the dominant dense signal and
            # is symmetric: reversing removes exactly as much fitness as the
            # same forward displacement earns.
            "progress": reward_progress * 300.0,
            # Speed is useful only while aligned, centered, and looking into
            # clear track. Standing still can never farm this term.
            "pace": 0.045 * productive_speed,
            "green_clearance": green_clearance_reward,
            "road": 0.0 if on_road else -0.12,
            "alignment": -0.05
            * forward_speed_ratio
            * (1.0 - forward_alignment),
            "track_offset": -0.04 * forward_speed_ratio * offset_ratio**2,
            "slip": -0.04 * abs(telemetry.lateral_speed) / max_speed,
            "clearance": -self.CLEARANCE_HAZARD_PENALTY_SCALE
            * forward_speed_ratio
            * clearance_hazard**2,
            "clearance_gain": clearance_gain,
            "wall_closing": wall_closing_penalty,
            "reverse": -0.16 * reverse_speed_ratio,
            "barrier_contact": (
                -self.WALL_CONTACT_PENALTY
                if collided
                else 0.0
            ),
            "collision": (
                -self.COLLISION_ENTRY_PENALTY
                - min(
                    self.COLLISION_IMPACT_PENALTY_CAP,
                    impact_speed * self.COLLISION_IMPACT_PENALTY_SCALE,
                )
                if collision_started
                else 0.0
            ),
            "stagnation": -min(0.12, stagnation_excess * 0.002),
            # Ordered gates are monotonically consumed within one lap
            # candidate, so crossing back and forth cannot farm this signal.
            "checkpoint": (
                self.ORDERED_CHECKPOINT_REWARD
                if ordered_checkpoint_advanced
                else 0.0
            ),
            "lap": self.LAP_COMPLETION_REWARD if lap_completed else 0.0,
            "lap_time": lap_time_bonus if lap_completed else 0.0,
        }
        reward = sum(reward_terms.values())
        self.last_reward_terms = reward_terms
        self.steps += 1
        self.previous_progress = after.progress
        self.last_projection = after
        stagnated = self._stagnation_steps >= self.STAGNATION_LIMIT_STEPS
        truncated = (
            self.steps >= self.max_steps
            or stagnated
            or self._collision_looped
        )
        if self._collision_looped:
            truncation_reason = "collision_loop"
        elif stagnated:
            truncation_reason = "stagnation"
        elif self.steps >= self.max_steps:
            truncation_reason = "step_limit"
        else:
            truncation_reason = None
        self._last_truncation_reason = truncation_reason
        info: dict[str, object] = {
            "circuit": self.circuit.slug,
            "terrain": active_terrain.kind.value,
            "on_road": on_road,
            "progress": after.progress,
            "episode_lap_progress": episode_lap_progress,
            "max_episode_lap_progress": self._max_episode_lap_progress,
            "episode_target_progress": self._episode_target_progress,
            "max_episode_target_progress": self._max_episode_target_progress,
            "laps": self.laps,
            "lap_target": self._lap_target,
            "laps_remaining": max(0, self._lap_target - self.laps),
            "lap_completed": lap_completed,
            "lap_target_completed": lap_target_completed,
            "checkpoint_advanced": ordered_checkpoint_advanced,
            "next_lap_checkpoint": self._next_lap_checkpoint,
            "current_lap_time": self.current_lap_time,
            "last_lap_time": self.last_lap_time,
            "best_lap_time": self.best_lap_time,
            "episode_best_lap_time": (
                min(self._episode_lap_times) if self._episode_lap_times else None
            ),
            "episode_mean_lap_time": (
                sum(self._episode_lap_times) / len(self._episode_lap_times)
                if self._episode_lap_times
                else None
            ),
            "lap_time_reference": self.lap_time_reference,
            "lap_time_bonus": lap_time_bonus if lap_completed else 0.0,
            "episode_lap_time_bonus_total": (
                self._episode_lap_time_bonus_total
            ),
            "lap_time_bonus_valid": (
                lap_time_bonus_valid if lap_completed else False
            ),
            "lap_candidate_valid": self._lap_candidate_armed,
            "lap_origin_progress": self._lap_origin_progress,
            "random_start_curriculum": self.random_start_curriculum,
            "curriculum_unlocked": self._curriculum_unlocked,
            "curriculum_ready": self._curriculum_unlocked,
            "curriculum_lap_completed": curriculum_lap_completed,
            "normal_start_probability": self.NORMAL_START_PROBABILITY,
            "spawn_mode": self._spawn_mode,
            "spawn_progress": self._spawn_progress,
            "collided": collided,
            "collision_started": collision_started,
            "impact_speed": impact_speed,
            "heading_alignment": heading_alignment,
            "forward_clearance": forward_clearance,
            "usable_clearance": self._usable_clearance,
            "previous_usable_clearance": self._previous_usable_clearance,
            "clearance_delta": self._clearance_delta,
            "green_ray_fraction": self._green_ray_fraction,
            "wall_closing": self._wall_closing,
            "clearance_motion_ratio": self._clearance_motion_ratio,
            "clearance_green_threshold": self.CLEARANCE_GREEN_THRESHOLD,
            "wall_contact_active": self._wall_contact_active,
            "wall_contact_steps": self._wall_contact_steps,
            "wall_contact_limit": self.COLLISION_RECOVERY_TIMEOUT_STEPS,
            "recent_collision_entries": self._recent_collision_entries,
            "collision_entry_limit": self.COLLISION_LOOP_ENTRY_LIMIT,
            "collision_loop_window_steps": self.COLLISION_LOOP_WINDOW_STEPS,
            "steps_since_collision": self._steps_since_collision,
            "collision_looped": self._collision_looped,
            "collision_recovery_active": self._collision_recovery_active,
            "collision_recovery_steps": self._collision_recovery_steps,
            "collision_recovery_clean_steps": (
                self._collision_recovery_clean_steps
            ),
            "collision_recovery_confirm_steps": (
                self.COLLISION_RECOVERY_CONFIRM_STEPS
            ),
            "collision_recovery_timeout_steps": (
                self.COLLISION_RECOVERY_TIMEOUT_STEPS
            ),
            "collision_recoveries": self._collision_recoveries,
            "collision_pressure": self._collision_pressure,
            "clearance_objective": self._clearance_objective_config(),
            "stagnation_steps": self._stagnation_steps,
            "stagnated": stagnated,
            "truncation_reason": truncation_reason,
            "reward_terms": reward_terms.copy(),
            "telemetry": telemetry,
        }
        return StepResult(
            self.observation(), reward, lap_target_completed, truncated, info
        )

    def change_circuit(self, name: str) -> tuple[float, ...]:
        self.circuit = get_circuit(name)
        self._curriculum_unlocked = False
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
            "lap_target": self._lap_target,
            "laps_remaining": max(0, self._lap_target - self.laps),
            "current_lap_time": self.current_lap_time,
            "last_lap_time": self.last_lap_time,
            "best_lap_time": self.best_lap_time,
            "episode_best_lap_time": (
                min(self._episode_lap_times) if self._episode_lap_times else None
            ),
            "episode_mean_lap_time": (
                sum(self._episode_lap_times) / len(self._episode_lap_times)
                if self._episode_lap_times
                else None
            ),
            "lap_time_reference": self.lap_time_reference,
            "lap_time_bonus": self._last_lap_time_bonus,
            "episode_lap_time_bonus_total": (
                self._episode_lap_time_bonus_total
            ),
            "lap_time_bonus_valid": self._last_lap_time_bonus_valid,
            "lap_candidate_valid": self._lap_candidate_armed,
            "episode_lap_progress": self._episode_lap_progress,
            "max_episode_lap_progress": self._max_episode_lap_progress,
            "episode_target_progress": self._episode_target_progress,
            "max_episode_target_progress": self._max_episode_target_progress,
            "lap_origin_progress": self._lap_origin_progress,
            "random_start_curriculum": self.random_start_curriculum,
            "curriculum_unlocked": self._curriculum_unlocked,
            "curriculum_ready": self._curriculum_unlocked,
            "curriculum_lap_completed": self._last_curriculum_lap_completed,
            "lap_target_completed": self._last_lap_target_completed,
            "normal_start_probability": self.NORMAL_START_PROBABILITY,
            "spawn_mode": self._spawn_mode,
            "spawn_progress": self._spawn_progress,
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
            "forward_clearance": min(
                ray.normalized_distance
                for ray in self.sensor_rays()[
                    len(self.SENSOR_RELATIVE_ANGLES) // 2
                    - 1 : len(self.SENSOR_RELATIVE_ANGLES) // 2
                    + 2
                ]
            ),
            "usable_clearance": self._usable_clearance,
            "previous_usable_clearance": self._previous_usable_clearance,
            "clearance_delta": self._clearance_delta,
            "green_ray_fraction": self._green_ray_fraction,
            "wall_closing": self._wall_closing,
            "clearance_motion_ratio": self._clearance_motion_ratio,
            "clearance_green_threshold": self.CLEARANCE_GREEN_THRESHOLD,
            "wall_contact_active": self._wall_contact_active,
            "wall_contact_steps": self._wall_contact_steps,
            "wall_contact_limit": self.COLLISION_RECOVERY_TIMEOUT_STEPS,
            "recent_collision_entries": self._recent_collision_entries,
            "collision_entry_limit": self.COLLISION_LOOP_ENTRY_LIMIT,
            "collision_loop_window_steps": self.COLLISION_LOOP_WINDOW_STEPS,
            "steps_since_collision": self._steps_since_collision,
            "collision_looped": self._collision_looped,
            "collision_recovery_active": self._collision_recovery_active,
            "collision_recovery_steps": self._collision_recovery_steps,
            "collision_recovery_clean_steps": (
                self._collision_recovery_clean_steps
            ),
            "collision_recovery_confirm_steps": (
                self.COLLISION_RECOVERY_CONFIRM_STEPS
            ),
            "collision_recovery_timeout_steps": (
                self.COLLISION_RECOVERY_TIMEOUT_STEPS
            ),
            "collision_recoveries": self._collision_recoveries,
            "collision_pressure": self._collision_pressure,
            "truncation_reason": self._last_truncation_reason,
            "clearance_objective": self._clearance_objective_config(),
            "stagnation_steps": self._stagnation_steps,
            "stagnation_limit_steps": self.STAGNATION_LIMIT_STEPS,
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
