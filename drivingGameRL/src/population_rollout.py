"""Read-only, parallel policy rollouts for the driving learning dashboard.

This module deliberately sits outside the training loop.  Every displayed
driver owns a cloned policy and an independent :class:`DrivingEnv`, so drawing
or advancing the comparison cannot add replay transitions, run gradients, or
change the population's fitness and selection state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .environment import DrivingEnv
from .ml import DrivingDQNAgent

if TYPE_CHECKING:
    from .learning_runtime import DrivingLearningSession


@dataclass(slots=True)
class _PolicyRollout:
    """Presentation-owned state for one member of the visible generation."""

    index: int
    member_id: int
    agent: DrivingDQNAgent
    env: DrivingEnv
    observation: tuple[float, ...]
    action: int
    q_values: tuple[float, ...]
    episodes: int = 0


class PopulationRolloutManager:
    """Run bounded, greedy population comparisons without touching training.

    The manager refreshes automatically when ``session.current_generation``
    changes.  All cars in a refresh use the same environment seed and start
    pose, which makes differences on track attributable to their policies.
    """

    HARD_MAX_CARS = 12

    def __init__(
        self,
        session: DrivingLearningSession,
        max_cars: int = HARD_MAX_CARS,
    ) -> None:
        if isinstance(max_cars, bool) or not isinstance(max_cars, int):
            raise ValueError("max_cars must be an integer")
        if max_cars <= 0:
            raise ValueError("max_cars must be positive")
        self.session = session
        self.max_cars = min(max_cars, self.HARD_MAX_CARS)
        self._generation = -1
        self._rollouts: list[_PolicyRollout] = []
        self.refresh(force=True)

    @property
    def generation(self) -> int:
        """Generation whose cloned policies are currently on track."""

        return self._generation

    @property
    def count(self) -> int:
        return len(self._rollouts)

    @property
    def environments(self) -> tuple[DrivingEnv, ...]:
        """Presentation environments, exposed read-only as a bounded tuple."""

        return tuple(rollout.env for rollout in self._rollouts)

    def refresh(self, *, force: bool = False) -> bool:
        """Clone the current generation when it changes.

        Returns ``True`` when the displayed population was replaced.  A forced
        refresh is useful after loading a checkpoint that retains its generation
        number but replaces its weights.
        """

        generation = int(self.session.current_generation)
        if not force and generation == self._generation:
            return False

        policies = self.session.population_policy_clones(
            max_cars=self.max_cars,
            reusable_policy_clones=tuple(
                rollout.agent for rollout in self._rollouts
            ),
        )
        source_env = self.session.env
        # The training environment may use a generation/member-specific seed.
        # Sharing that value among these independent environments guarantees an
        # identical comparison without sharing any mutable simulation object.
        rollout_seed = (
            source_env.seed
            if source_env.seed is not None
            else int(getattr(self.session.config, "seed", 0))
        )
        curriculum_state = source_env.curriculum_state()
        rollouts: list[_PolicyRollout] = []
        for index, (member_id, agent) in enumerate(policies):
            env = DrivingEnv(
                source_env.circuit,
                build=source_env.vehicle.build,
                seed=rollout_seed,
                fixed_dt=source_env.fixed_dt,
                max_steps=source_env.max_steps,
                random_start_curriculum=source_env.random_start_curriculum,
            )
            # The constructor starts a fresh curriculum. Copy only its tiny
            # readiness latch, then replay the shared seed so every displayed
            # genome receives the same scenario as its peers.
            env.load_curriculum_state(curriculum_state)
            observation = env.reset(seed=rollout_seed)
            q_values = agent.q_values(observation)
            rollouts.append(
                _PolicyRollout(
                    index=index,
                    member_id=int(member_id),
                    agent=agent,
                    env=env,
                    observation=observation,
                    action=int(np.argmax(q_values)),
                    q_values=tuple(float(value) for value in q_values),
                )
            )

        self._rollouts = rollouts
        self._generation = generation
        return True

    def step(self, steps: int = 1) -> None:
        """Advance every presentation car by the same number of fixed frames."""

        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        self.refresh()
        for _ in range(steps):
            for rollout in self._rollouts:
                # ``action`` is always the greedy decision for the observation
                # currently stored on the rollout.  After moving, refresh the
                # decision so every telemetry field describes the same instant.
                action = rollout.action
                result = rollout.env.step(action)
                rollout.observation = result.observation
                if result.terminated or result.truncated:
                    # Restart only this car.  Other cars retain their position,
                    # lap clock, and episode state.
                    # Keep consuming the private seeded RNG sequence. Reseeding
                    # here would replay one identical spawn forever.
                    rollout.observation = rollout.env.reset()
                    rollout.episodes += 1
                q_values = rollout.agent.q_values(rollout.observation)
                rollout.action = int(np.argmax(q_values))
                rollout.q_values = tuple(float(value) for value in q_values)

    def telemetry(self, *, include_rays: bool = True) -> list[dict[str, Any]]:
        """Return JSON-friendly poses, policy decisions, and real sensor rays."""

        self.refresh()
        snapshots: list[dict[str, Any]] = []
        for rollout in self._rollouts:
            env_snapshot = rollout.env.telemetry()
            rays = self._sensor_rays(rollout) if include_rays else []
            item: dict[str, Any] = {
                "index": rollout.index,
                "member": rollout.member_id,
                "member_id": rollout.member_id,
                "generation": self._generation,
                "position": tuple(float(value) for value in env_snapshot["position"]),
                "heading": float(rollout.env.vehicle.state.heading),
                "heading_degrees": float(env_snapshot["heading_degrees"]),
                "speed": float(env_snapshot["speed"]),
                "progress": float(env_snapshot["progress"]),
                "episode_lap_progress": float(
                    env_snapshot["episode_lap_progress"]
                ),
                "laps": int(env_snapshot["laps"]),
                "steps": int(env_snapshot["steps"]),
                "episodes": rollout.episodes,
                "random_start_curriculum": bool(
                    env_snapshot["random_start_curriculum"]
                ),
                "curriculum_unlocked": bool(
                    env_snapshot["curriculum_unlocked"]
                ),
                "spawn_mode": str(env_snapshot["spawn_mode"]),
                "spawn_progress": float(env_snapshot["spawn_progress"]),
                "lap_origin_progress": float(
                    env_snapshot["lap_origin_progress"]
                ),
                "action": rollout.action,
                "q_values": list(rollout.q_values),
                "observation": list(rollout.observation),
                # ``rays`` is the compact visual contract; ``sensor_rays`` is
                # the explicit educational name retained for API clarity.
                "rays": rays,
                "sensor_rays": rays,
            }
            snapshots.append(item)
        return snapshots

    @staticmethod
    def _sensor_rays(rollout: _PolicyRollout) -> list[dict[str, Any]]:
        """Serialize the five distances used by this policy's observation."""

        sensor_rays = rollout.env.sensor_rays()
        labels = DrivingEnv.OBSERVATION_LABELS[-len(sensor_rays) :]
        rays: list[dict[str, Any]] = []
        for index, (label, relative_angle, ray) in enumerate(
            zip(labels, DrivingEnv.SENSOR_RELATIVE_ANGLES, sensor_rays)
        ):
            rays.append(
                {
                    "index": index,
                    "label": label,
                    "relative_angle": relative_angle,
                    "angle": ray.angle,
                    "normalized_distance": ray.normalized_distance,
                    "distance": ray.distance,
                    "max_distance": ray.max_distance,
                    "origin": (float(ray.origin.x), float(ray.origin.y)),
                    "endpoint": (float(ray.endpoint.x), float(ray.endpoint.y)),
                    "hit": ray.hit,
                }
            )
        return rays
