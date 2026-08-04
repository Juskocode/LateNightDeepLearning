"""Deterministic learning-only safety prior for the nine-ray driving policy.

The prior is intentionally a transparent action filter rather than a reward
source. It reads only features already available to the learner, leaves clear
road decisions unchanged, and reports both the neural policy's proposal and
the action that actually reached the environment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from .environment import DrivingAction, DrivingEnv


@dataclass(frozen=True, slots=True)
class SensorClearanceDecision:
    """One immutable proposed-to-executed action decision."""

    proposed_action: int
    executed_action: int
    intervened: bool
    dangerous: bool
    reason: str
    speed_ratio: float
    forward_clearance: float
    danger_threshold: float
    critical_clearance: float
    boundary_threshold: float
    projected_offset: float
    left_open_space: float
    right_open_space: float
    left_utility: float
    right_utility: float
    ray_clearances: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed_action": self.proposed_action,
            "executed_action": self.executed_action,
            "intervened": self.intervened,
            "dangerous": self.dangerous,
            "reason": self.reason,
            "speed_ratio": self.speed_ratio,
            "forward_clearance": self.forward_clearance,
            "danger_threshold": self.danger_threshold,
            "critical_clearance": self.critical_clearance,
            "boundary_threshold": self.boundary_threshold,
            "projected_offset": self.projected_offset,
            "left_open_space": self.left_open_space,
            "right_open_space": self.right_open_space,
            "left_utility": self.left_utility,
            "right_utility": self.right_utility,
            "ray_clearances": list(self.ray_clearances),
        }


class SensorClearancePolicy:
    """Look ahead with speed and steer toward the safest green corridor."""

    BASE_FEATURE_COUNT = 7
    RAY_COUNT = 9
    DANGER_CLEARANCE = 0.44
    SPEED_LOOKAHEAD_GAIN = 0.12
    CRITICAL_CLEARANCE = 0.08
    BRAKE_SPEED_RATIO = 0.45
    RECOVERY_CLEARANCE = 0.10
    RECOVERY_SPEED_RATIO = 0.30
    RECOVERY_RELEASE_OFFSET = 0.48
    RECOVERY_INWARD_ALIGNMENT = -0.15
    GREEN_BONUS = 0.12
    LOOKAHEAD_BASE = 0.45
    LOOKAHEAD_SPEED_GAIN = 0.55
    LATERAL_PROJECTION_WEIGHT = 0.25
    CENTERING_UTILITY_WEIGHT = 0.70
    BOUNDARY_THRESHOLD = 0.58
    BOUNDARY_SPEED_TIGHTENING = 0.08
    SCORE_EPSILON = 1e-9
    # Outer-to-inner weights. Near-forward visibility matters most, while all
    # four rays on each side still contribute to the chosen escape corridor.
    SIDE_WEIGHTS = (0.10, 0.18, 0.28, 0.44)

    def __init__(self) -> None:
        expected = self.BASE_FEATURE_COUNT + self.RAY_COUNT
        if len(DrivingEnv.OBSERVATION_LABELS) != expected:
            raise RuntimeError(
                "SensorClearancePolicy requires the exact 7-feature + 9-ray "
                "driving observation contract"
            )
        if len(DrivingEnv.SENSOR_RELATIVE_ANGLES) != self.RAY_COUNT:
            raise RuntimeError(
                "SensorClearancePolicy requires the exact nine-ray fan"
            )

    def decide(
        self,
        observation: Sequence[float],
        proposed_action: int | DrivingAction,
    ) -> SensorClearanceDecision:
        """Return the deterministic action filter result for one observation."""

        values = self._validated_observation(observation)
        if isinstance(proposed_action, bool):
            raise ValueError(
                f"Invalid proposed driving action: {proposed_action!r}"
            )
        try:
            proposed = DrivingAction(proposed_action)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid proposed driving action: {proposed_action!r}"
            ) from error

        speed_ratio = max(0.0, values[0])
        lateral_speed = values[2]
        heading_error = values[3]
        track_offset = values[4]
        rays = values[-self.RAY_COUNT :]
        middle = self.RAY_COUNT // 2
        forward_clearance = min(rays[middle - 1 : middle + 2])
        left_open_space = sum(
            clearance * weight
            for clearance, weight in zip(rays[:middle], self.SIDE_WEIGHTS)
        )
        right_open_space = sum(
            clearance * weight
            for clearance, weight in zip(
                rays[middle + 1 :], reversed(self.SIDE_WEIGHTS)
            )
        )
        lookahead_speed = min(speed_ratio, 1.0)
        left_green_bonus = self.GREEN_BONUS * sum(
            weight
            for clearance, weight in zip(rays[:middle], self.SIDE_WEIGHTS)
            if clearance >= DrivingEnv.CLEARANCE_GREEN_THRESHOLD
        )
        right_green_bonus = self.GREEN_BONUS * sum(
            weight
            for clearance, weight in zip(
                rays[middle + 1 :], reversed(self.SIDE_WEIGHTS)
            )
            if clearance >= DrivingEnv.CLEARANCE_GREEN_THRESHOLD
        )
        lookahead_horizon = (
            self.LOOKAHEAD_BASE + self.LOOKAHEAD_SPEED_GAIN * lookahead_speed
        )
        projected_offset = (
            track_offset
            + heading_error * lookahead_horizon
            + self.LATERAL_PROJECTION_WEIGHT * lateral_speed
        )
        left_utility = (
            left_open_space
            + left_green_bonus
            + self.CENTERING_UTILITY_WEIGHT * max(0.0, projected_offset)
        )
        right_utility = (
            right_open_space
            + right_green_bonus
            + self.CENTERING_UTILITY_WEIGHT * max(0.0, -projected_offset)
        )
        danger_threshold = (
            self.DANGER_CLEARANCE + self.SPEED_LOOKAHEAD_GAIN * lookahead_speed
        )
        boundary_threshold = (
            self.BOUNDARY_THRESHOLD
            - self.BOUNDARY_SPEED_TIGHTENING * lookahead_speed
        )
        dangerous = (
            forward_clearance <= danger_threshold
            or abs(projected_offset) >= boundary_threshold
        )
        front_fan_max = max(rays[middle - 1 : middle + 2])
        critical = (
            front_fan_max <= self.CRITICAL_CLEARANCE
            and speed_ratio >= self.BRAKE_SPEED_RATIO
        )
        if abs(track_offset) <= self.SCORE_EPSILON:
            inward_alignment = 1.0
        else:
            inward_direction = -math.copysign(math.pi / 2.0, track_offset)
            inward_alignment = math.cos(
                heading_error * math.pi - inward_direction
            )
        nose_faces_outward = (
            abs(track_offset) >= self.RECOVERY_RELEASE_OFFSET
            and inward_alignment <= self.RECOVERY_INWARD_ALIGNMENT
        )
        reverse_recovery = (
            (front_fan_max <= self.RECOVERY_CLEARANCE or nose_faces_outward)
            and speed_ratio <= self.RECOVERY_SPEED_RATIO
        )
        if not dangerous:
            executed = proposed
            reason = "clear_road"
        elif critical:
            executed = DrivingAction.BRAKE
            reason = "critical_brake"
        elif reverse_recovery:
            # BRAKE is an arcade brake-then-reverse action. At a nearly stopped,
            # fully blocked nose it creates the backwards motion needed for
            # steering to become effective instead of feeding throttle into the
            # same barrier until the recovery budget expires.
            executed = DrivingAction.BRAKE
            reason = "blocked_reverse_recovery"
        else:
            executed, reason = self._open_side_action(
                proposed,
                left_utility,
                right_utility,
                rays[middle - 1],
                rays[middle + 1],
            )

        return SensorClearanceDecision(
            proposed_action=int(proposed),
            executed_action=int(executed),
            intervened=executed != proposed,
            dangerous=dangerous,
            reason=reason,
            speed_ratio=speed_ratio,
            forward_clearance=forward_clearance,
            danger_threshold=danger_threshold,
            critical_clearance=self.CRITICAL_CLEARANCE,
            boundary_threshold=boundary_threshold,
            projected_offset=projected_offset,
            left_open_space=left_open_space,
            right_open_space=right_open_space,
            left_utility=left_utility,
            right_utility=right_utility,
            ray_clearances=rays,
        )

    __call__ = decide

    def _open_side_action(
        self,
        proposed: DrivingAction,
        left_score: float,
        right_score: float,
        left_near: float,
        right_near: float,
    ) -> tuple[DrivingAction, str]:
        if left_score > right_score + self.SCORE_EPSILON:
            return DrivingAction.STEER_LEFT, "danger_steer_left"
        if right_score > left_score + self.SCORE_EPSILON:
            return DrivingAction.STEER_RIGHT, "danger_steer_right"
        if left_near > right_near + self.SCORE_EPSILON:
            return DrivingAction.STEER_LEFT, "danger_steer_left_tiebreak"
        if right_near > left_near + self.SCORE_EPSILON:
            return DrivingAction.STEER_RIGHT, "danger_steer_right_tiebreak"
        if proposed in (DrivingAction.STEER_LEFT, DrivingAction.STEER_RIGHT):
            return proposed, "danger_equal_space_keep_steer"
        # A fully symmetric blocked fan has no observable preferred side. A
        # fixed tie break remains deterministic across threads and processes.
        return DrivingAction.STEER_LEFT, "danger_equal_space_left_tiebreak"

    @classmethod
    def _validated_observation(
        cls, observation: Sequence[float]
    ) -> tuple[float, ...]:
        if isinstance(observation, (str, bytes)):
            raise ValueError("observation must be a numeric sequence")
        try:
            values = tuple(float(value) for value in observation)
        except (TypeError, ValueError) as error:
            raise ValueError("observation must be a numeric sequence") from error
        expected = cls.BASE_FEATURE_COUNT + cls.RAY_COUNT
        if len(values) != expected:
            raise ValueError(
                f"observation must contain exactly {expected} values "
                f"({cls.BASE_FEATURE_COUNT} base features and {cls.RAY_COUNT} rays)"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("observation values must be finite")
        rays = values[-cls.RAY_COUNT :]
        if any(clearance < 0.0 or clearance > 1.0 for clearance in rays):
            raise ValueError("ray clearances must be normalized to [0, 1]")
        return values


@dataclass(slots=True)
class SensorClearanceStats:
    """Ephemeral observability for a stateless, checkpoint-free safety prior."""

    decisions: int = 0
    interventions: int = 0
    last: SensorClearanceDecision | None = None

    def observe(self, decision: SensorClearanceDecision) -> None:
        if not isinstance(decision, SensorClearanceDecision):
            raise TypeError("decision must be a SensorClearanceDecision")
        self.decisions += 1
        self.interventions += int(decision.intervened)
        self.last = decision

    def snapshot(self) -> dict[str, Any]:
        last: Mapping[str, Any]
        if self.last is None:
            last = {
                "proposed_action": None,
                "executed_action": None,
                "intervened": False,
                "dangerous": False,
                "reason": "not_evaluated",
                "speed_ratio": 0.0,
                "forward_clearance": 1.0,
                "danger_threshold": SensorClearancePolicy.DANGER_CLEARANCE,
                "critical_clearance": SensorClearancePolicy.CRITICAL_CLEARANCE,
                "boundary_threshold": SensorClearancePolicy.BOUNDARY_THRESHOLD,
                "projected_offset": 0.0,
                "left_open_space": 1.0,
                "right_open_space": 1.0,
                "left_utility": 1.0,
                "right_utility": 1.0,
                "ray_clearances": [],
            }
        else:
            last = self.last.to_dict()
        rate = self.interventions / self.decisions if self.decisions else 0.0
        return {
            "enabled": True,
            **last,
            "decisions": self.decisions,
            "interventions": self.interventions,
            "intervention_rate": rate,
            "danger_clearance": SensorClearancePolicy.DANGER_CLEARANCE,
            "speed_lookahead_gain": SensorClearancePolicy.SPEED_LOOKAHEAD_GAIN,
            "critical_clearance_base": SensorClearancePolicy.CRITICAL_CLEARANCE,
            "green_bonus": SensorClearancePolicy.GREEN_BONUS,
            "boundary_threshold_base": SensorClearancePolicy.BOUNDARY_THRESHOLD,
            "boundary_speed_tightening": (
                SensorClearancePolicy.BOUNDARY_SPEED_TIGHTENING
            ),
            "brake_speed_ratio": SensorClearancePolicy.BRAKE_SPEED_RATIO,
            "recovery_clearance": SensorClearancePolicy.RECOVERY_CLEARANCE,
            "recovery_speed_ratio": SensorClearancePolicy.RECOVERY_SPEED_RATIO,
            "recovery_release_offset": (
                SensorClearancePolicy.RECOVERY_RELEASE_OFFSET
            ),
            "recovery_inward_alignment": (
                SensorClearancePolicy.RECOVERY_INWARD_ALIGNMENT
            ),
        }


__all__ = (
    "SensorClearanceDecision",
    "SensorClearancePolicy",
    "SensorClearanceStats",
)
