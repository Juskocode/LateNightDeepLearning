"""Live population views for the driving learning dashboard.

Population learning renders the real, concurrently scored environments after
their synchronization barrier. Standalone DQN keeps the original isolated
policy clone so its optional comparison view cannot alter replay or gradients.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    """Expose a bounded live comparison without advancing training twice.

    Genetic modes proxy the real scored cars, whose environments already move
    together in :class:`PopulationTrainer`. Standalone modes still use one
    presentation-only clone. The manager refreshes automatically when the
    generation changes.
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
    def uses_scored_population(self) -> bool:
        """Whether telemetry comes from training-owned population environments."""

        return bool(self.session.is_population)

    @property
    def generation(self) -> int:
        """Generation whose scored cars or standalone clone are on track."""

        return self._generation

    @property
    def count(self) -> int:
        if self.uses_scored_population:
            trainer = self.session._population_trainer
            return min(len(trainer.member_environments), self.max_cars)
        return len(self._rollouts)

    @property
    def environments(self) -> tuple[DrivingEnv, ...]:
        """Currently displayed environments, exposed as a bounded tuple."""

        if self.uses_scored_population:
            trainer = self.session._population_trainer
            return tuple(trainer.member_environments[: self.max_cars])
        return tuple(rollout.env for rollout in self._rollouts)

    def refresh(self, *, force: bool = False) -> bool:
        """Refresh the scored-generation proxy or standalone comparison clone.

        Returns ``True`` when the displayed population was replaced.  A forced
        refresh is useful after loading a checkpoint that retains its generation
        number but replaces its weights.
        """

        generation = int(self.session.current_generation)
        if not force and generation == self._generation:
            return False

        if self.uses_scored_population:
            # The trainer creates the generation's scored environments. Never
            # clone or reset them here: the synchronous training barrier is the
            # sole owner of their advancement.
            self._rollouts = []
            self._generation = generation
            return True

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
        if self.uses_scored_population:
            # ``DrivingLearningSession.step`` already advanced every scored car
            # once. This compatibility no-op lets the existing game/capture
            # loops call manager.step() without double stepping the population.
            return
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
        if self.uses_scored_population:
            return scored_population_telemetry(
                self.session,
                max_cars=self.max_cars,
                include_rays=include_rays,
            )
        snapshots: list[dict[str, Any]] = []
        for rollout in self._rollouts:
            env_snapshot = rollout.env.telemetry()
            rays = _sensor_rays(rollout.env) if include_rays else []
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
        """Serialize the nine distances used by this policy's observation."""

        return _sensor_rays(rollout.env)


def _sensor_rays(env: DrivingEnv) -> list[dict[str, Any]]:
    """Serialize the exact nine ray readings for one environment pose."""

    sensor_rays = env.sensor_rays()
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


def scored_population_telemetry(
    session: DrivingLearningSession,
    *,
    max_cars: int | None = None,
    include_rays: bool = True,
) -> list[dict[str, Any]]:
    """Snapshot the real scored population after its synchronous step barrier."""

    if not session.is_population:
        return []
    if max_cars is not None and (
        isinstance(max_cars, bool) or not isinstance(max_cars, int) or max_cars <= 0
    ):
        raise ValueError("max_cars must be a positive integer or None")

    trainer = session._population_trainer
    environments = tuple(trainer.member_environments)
    observations = tuple(trainer.member_observations)
    members = tuple(trainer.population)
    count = len(members) if max_cars is None else min(len(members), max_cars)
    if not (len(environments) == len(observations) == len(members)):
        raise RuntimeError("population runtime state is not aligned by member index")

    rows = tuple(trainer.member_runtime)
    active_indices = set(int(index) for index in trainer.active_member_indices)
    snapshots: list[dict[str, Any]] = []
    for index, (member, env, observation) in enumerate(
        zip(members[:count], environments[:count], observations[:count])
    ):
        row = rows[index] if index < len(rows) else {}
        env_snapshot = env.telemetry()
        q_values = tuple(float(value) for value in member.agent.q_values(observation))
        action_value = row.get("action")
        action = int(np.argmax(q_values) if action_value is None else action_value)
        raw_action_value = row.get("raw_action")
        raw_action = int(action if raw_action_value is None else raw_action_value)
        safety_value = row.get("safety")
        safety = dict(safety_value) if isinstance(safety_value, Mapping) else {}
        rays = _sensor_rays(env) if include_rays else []
        status = row.get("status")
        if status == "active":
            status = "evaluating"
        if status is None:
            status = (
                "evaluating"
                if index in active_indices
                else "evaluated"
                if member.evaluated
                else "queued"
            )
        item: dict[str, Any] = {
            "index": index,
            "member": int(member.member_id),
            "member_id": int(member.member_id),
            "generation": int(trainer.generation),
            "position": tuple(float(value) for value in env_snapshot["position"]),
            "heading": float(env.vehicle.state.heading),
            "heading_degrees": float(env_snapshot["heading_degrees"]),
            "speed": float(env_snapshot["speed"]),
            "progress": float(env_snapshot["progress"]),
            "episode_lap_progress": float(env_snapshot["episode_lap_progress"]),
            "laps": int(env_snapshot["laps"]),
            "collisions": int(env_snapshot["collisions"]),
            "steps": int(row.get("evaluation_step", env_snapshot["steps"])),
            "episodes": 0,
            "status": str(status),
            "fitness": (
                None
                if row.get("result") is None
                else float(row["result"].fitness)
            ),
            "evaluation_return": float(row.get("evaluation_return", 0.0)),
            "random_start_curriculum": bool(
                env_snapshot["random_start_curriculum"]
            ),
            "curriculum_unlocked": bool(env_snapshot["curriculum_unlocked"]),
            "spawn_mode": str(env_snapshot["spawn_mode"]),
            "spawn_progress": float(env_snapshot["spawn_progress"]),
            "lap_origin_progress": float(env_snapshot["lap_origin_progress"]),
            "action": action,
            "raw_action": raw_action,
            "proposed_action": raw_action,
            "executed_action": action,
            "safety_intervened": bool(row.get("safety_intervened", False)),
            "safety": safety,
            "usable_clearance": float(env_snapshot["usable_clearance"]),
            "previous_usable_clearance": float(
                env_snapshot["previous_usable_clearance"]
            ),
            "clearance_delta": float(env_snapshot["clearance_delta"]),
            "green_ray_fraction": float(env_snapshot["green_ray_fraction"]),
            "wall_closing": bool(env_snapshot["wall_closing"]),
            "wall_contact_active": bool(env_snapshot["wall_contact_active"]),
            "wall_contact_steps": int(env_snapshot["wall_contact_steps"]),
            "wall_contact_limit": int(env_snapshot["wall_contact_limit"]),
            "recent_collision_entries": int(
                env_snapshot["recent_collision_entries"]
            ),
            "collision_entry_limit": int(env_snapshot["collision_entry_limit"]),
            "collision_looped": bool(env_snapshot["collision_looped"]),
            "truncation_reason": env_snapshot["truncation_reason"],
            "reward_terms": dict(env_snapshot["reward_terms"]),
            "q_values": list(q_values),
            "observation": [float(value) for value in observation],
            "rays": rays,
            "sensor_rays": rays,
            "scored": True,
            "source": "training",
        }
        snapshots.append(item)
    return snapshots
