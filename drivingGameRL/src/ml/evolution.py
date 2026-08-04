"""Deterministic population learning for the top-down Driving Lab.

The population owns ordinary :class:`DrivingDQNAgent` instances.  In pure
``genetic`` mode their neural-network weights are the chromosome and fitness
is the return earned by a greedy policy.  ``genetic_dqn`` evaluates the same
chromosomes with epsilon-greedy actions and applies replay-based TD updates
before selection, combining within-lifetime and across-generation learning.

Every member owns an isolated ``DrivingEnv`` and evaluation context.  Active
members advance one lockstep tick through a bounded thread pool; results are
merged by population index on the coordinator thread.  Seeded runs therefore
remain reproducible even when worker completion order changes.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Literal, NoReturn

import numpy as np
import torch

from ..environment import DrivingAction, DrivingEnv, StepResult
from ..sensor_clearance import (
    SensorClearanceDecision,
    SensorClearancePolicy,
    SensorClearanceStats,
)
from .config import DQNConfig, default_population_dqn_config
from .dqn import DrivingDQNAgent


EvolutionAlgorithm = Literal["genetic", "genetic_dqn"]
CrossoverMode = Literal["uniform", "blend"]


def normalize_evolution_algorithm(value: str) -> EvolutionAlgorithm:
    """Normalize user-facing spelling without silently accepting other RL modes."""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"ga", "genetic", "evolution", "evolutionary"}:
        return "genetic"
    if normalized in {
        "hybrid",
        "genetic_dqn",
        "genetic_q",
        "evolutionary_dqn",
    }:
        return "genetic_dqn"
    raise ValueError("algorithm must be 'genetic' or 'genetic_dqn'")


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """Validated genetic-search and evaluation controls."""

    algorithm: EvolutionAlgorithm = "genetic_dqn"
    population_size: int = 12
    elite_count: int = 2
    tournament_size: int = 3
    crossover: CrossoverMode = "uniform"
    crossover_rate: float = 0.90
    blend_alpha: float = 0.20
    mutation_rate: float = 0.06
    mutation_std: float = 0.035
    evaluation_steps: int = 1_800
    history_capacity: int = 256
    seed: int = 7

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "algorithm", normalize_evolution_algorithm(self.algorithm)
        )
        crossover = str(self.crossover).strip().lower().replace("-", "_")
        if crossover not in ("uniform", "blend"):
            raise ValueError("crossover must be 'uniform' or 'blend'")
        object.__setattr__(self, "crossover", crossover)

        self._positive_int("population_size", self.population_size)
        if self.population_size < 2:
            raise ValueError("population_size must be at least two")
        self._positive_int("elite_count", self.elite_count)
        if self.elite_count >= self.population_size:
            raise ValueError("elite_count must be smaller than population_size")
        self._positive_int("tournament_size", self.tournament_size)
        if self.tournament_size > self.population_size:
            raise ValueError("tournament_size cannot exceed population_size")
        self._positive_int("evaluation_steps", self.evaluation_steps)
        self._positive_int("history_capacity", self.history_capacity)
        self._probability("crossover_rate", self.crossover_rate)
        self._probability("mutation_rate", self.mutation_rate)
        self._non_negative_finite("blend_alpha", self.blend_alpha)
        self._non_negative_finite("mutation_std", self.mutation_std)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in the [0, 2**63) interval")

    @property
    def crossover_mode(self) -> CrossoverMode:
        """Readable alias used by visual controls."""

        return self.crossover

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "EvolutionConfig":
        return cls(**dict(values))

    @staticmethod
    def _positive_int(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _probability(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be between zero and one")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{name} must be between zero and one")

    @staticmethod
    def _non_negative_finite(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be finite and non-negative")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Complete, immutable scorecard for one population member."""

    generation: int
    member_id: int
    fitness: float
    total_reward: float
    steps: int
    laps: int
    progress: float
    collisions: int
    terminated: bool
    truncated: bool
    mean_loss: float = 0.0
    training_updates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "EvaluationResult":
        return cls(**dict(values))


# Both terms are useful in educational material.  Keep them as one concrete
# type so callers never need conversion glue.
MemberResult = EvaluationResult


@dataclass(slots=True)
class PopulationMember:
    """One independently owned policy and its lineage metadata."""

    member_id: int
    agent: DrivingDQNAgent = field(repr=False)
    birth_generation: int = 0
    parent_ids: tuple[int, ...] = ()
    result: EvaluationResult | None = None

    @property
    def fitness(self) -> float:
        return -math.inf if self.result is None else self.result.fitness

    @property
    def evaluated(self) -> bool:
        return self.result is not None

    def summary(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "birth_generation": self.birth_generation,
            "parent_ids": list(self.parent_ids),
            "evaluated": self.evaluated,
            "fitness": None if self.result is None else self.result.fitness,
            "result": None if self.result is None else self.result.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """Bounded history row for a completely evaluated generation."""

    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    worst_fitness: float
    fitness_std: float
    champion_id: int
    elite_ids: tuple[int, ...]
    population_size: int
    genome_diversity: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["elite_ids"] = list(self.elite_ids)
        return data

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GenerationRecord":
        data = dict(values)
        data["elite_ids"] = tuple(data["elite_ids"])
        return cls(**data)


GenerationStats = GenerationRecord


@dataclass(frozen=True, slots=True)
class ChampionSnapshot:
    """Read-only champion identity; use ``champion_agent`` for a policy clone."""

    generation: int
    member_id: int
    fitness: float
    result: EvaluationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "member_id": self.member_id,
            "fitness": self.fitness,
            "result": self.result.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ChampionSnapshot":
        data = dict(values)
        result = data["result"]
        if not isinstance(result, EvaluationResult):
            result = EvaluationResult.from_dict(result)
        return cls(
            generation=int(data["generation"]),
            member_id=int(data["member_id"]),
            fitness=float(data["fitness"]),
            result=result,
        )


@dataclass(frozen=True, slots=True)
class PopulationStep:
    """What happened during one lockstep population tick.

    The original scalar fields describe a deterministic representative member
    (the focal member, or the lowest-index completion when another car ends
    first). ``member_results`` reports every member completed by the same tick.
    """

    generation: int
    member_id: int
    member_index: int
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    terminated: bool
    truncated: bool
    member_completed: bool
    generation_completed: bool
    evolved: bool
    result: EvaluationResult | None
    generation_record: GenerationRecord | None
    info: dict[str, object]
    member_results: tuple[EvaluationResult, ...] = ()
    active_member_indices: tuple[int, ...] = ()
    proposed_action: int | None = None
    executed_action: int | None = None
    safety_intervened: bool = False
    safety_reason: str = "not_evaluated"


@dataclass(slots=True)
class _EvaluationRuntime:
    """Mutable state owned by exactly one population member."""

    env: DrivingEnv
    observation: np.ndarray
    total_reward: float = 0.0
    steps: int = 0
    losses: list[float] = field(default_factory=list)
    last_reward: float = 0.0
    last_info: dict[str, object] = field(default_factory=dict)
    safety: SensorClearanceStats = field(default_factory=SensorClearanceStats)


@dataclass(frozen=True, slots=True)
class _MemberAdvance:
    """Worker-produced transition merged later by the coordinator thread."""

    index: int
    state: np.ndarray
    safety_decision: SensorClearanceDecision
    env_result: StepResult
    next_state: np.ndarray
    budget_reached: bool
    done: bool
    loss: float | None


class PopulationTrainer:
    """Lockstep, multithreaded genetic population session over ``DrivingEnv``."""

    CHECKPOINT_VERSION = 1
    GENOME_SAMPLE_LIMIT = 2_048
    MAX_WORKER_CHUNK_TICKS = 8
    _SEED_LIMIT = 2**63

    def __init__(
        self,
        evolution_config: EvolutionConfig | None = None,
        dqn_config: DQNConfig | None = None,
        *,
        circuit: str = "harbor_loop",
        env: DrivingEnv | None = None,
        env_factory: Callable[[int], DrivingEnv] | None = None,
        auto_evolve: bool = True,
        parallel_workers: int | None = None,
    ) -> None:
        if env is not None and env_factory is not None:
            raise ValueError("provide env or env_factory, not both")
        self.config = evolution_config or EvolutionConfig()
        self.dqn_config = dqn_config or default_population_dqn_config(
            evaluation_steps=self.config.evaluation_steps,
            seed=self.config.seed,
        )
        # Stateless and read-only: one instance is safe to share across member
        # workers. Per-member intervention statistics remain coordinator-owned.
        self.clearance_policy = SensorClearancePolicy()
        if self.dqn_config.action_size != len(DrivingAction):
            raise ValueError(
                f"DrivingEnv has {len(DrivingAction)} actions, but DQNConfig has "
                f"{self.dqn_config.action_size}"
            )
        self.auto_evolve = bool(auto_evolve)
        if parallel_workers is not None and (
            isinstance(parallel_workers, bool)
            or not isinstance(parallel_workers, int)
            or parallel_workers <= 0
        ):
            raise ValueError("parallel_workers must be a positive integer or None")
        cpu_workers = max(1, os.cpu_count() or 1)
        requested_workers = (
            cpu_workers if parallel_workers is None else parallel_workers
        )
        self.requested_parallel_workers = parallel_workers
        self.parallel_workers = min(
            self.config.population_size,
            requested_workers,
            32,
        )
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        self._worker_failure: BaseException | None = None
        self._environment_decisions = 0
        self._last_tick_member_count = 0
        self._last_batch_ticks = 0
        self._last_batch_decisions = 0
        self._last_batch_ms = 0.0
        self._decision_throughput = 0.0
        self._tick_throughput = 0.0
        self._safety_stats = SensorClearanceStats()
        self._last_safety_decision: SensorClearanceDecision | None = None
        self._rng = np.random.default_rng(self.config.seed)
        self._next_member_id = 0
        self.generation = 0
        self.history: deque[GenerationRecord] = deque(
            maxlen=self.config.history_capacity
        )

        initial_seed = self._generation_evaluation_seed(0)
        if env is not None:
            if type(env) is not DrivingEnv:
                raise TypeError(
                    "env must be an exact DrivingEnv; use env_factory for custom "
                    "DrivingEnv subclasses so every member gets identical semantics"
                )
            first_env = env
        elif env_factory is not None:
            first_env = env_factory(initial_seed)
        else:
            first_env = DrivingEnv(
                circuit,
                seed=initial_seed,
                max_steps=max(self.config.evaluation_steps, 1),
                random_start_curriculum=True,
            )
        if not isinstance(first_env, DrivingEnv):
            raise TypeError("env_factory must return DrivingEnv instances")

        member_envs = [first_env]
        for _ in range(1, self.config.population_size):
            if env_factory is not None:
                member_env = env_factory(initial_seed)
                if not isinstance(member_env, DrivingEnv):
                    raise TypeError("env_factory must return DrivingEnv instances")
            else:
                member_env = self._clone_environment(first_env, seed=initial_seed)
            member_envs.append(member_env)
        if any(type(member_env) is not type(first_env) for member_env in member_envs):
            raise TypeError("env_factory must return one consistent DrivingEnv type")
        if len({id(member_env) for member_env in member_envs}) != len(member_envs):
            raise ValueError(
                "each population member requires an independent environment"
            )
        self._member_envs = member_envs
        self.env = first_env

        # Constructors already establish one valid observation. Avoid an extra
        # reset for member zero: custom factories must see identical reset counts
        # across every member before the shared generation runtime begins.
        initial_observation = self.env.observation()
        if len(initial_observation) != self.dqn_config.observation_size:
            raise ValueError(
                "DrivingEnv observation size does not match DQNConfig: "
                f"{len(initial_observation)} != {self.dqn_config.observation_size}"
            )
        expected_safety_observation_size = len(DrivingEnv.OBSERVATION_LABELS)
        if len(initial_observation) != expected_safety_observation_size:
            raise ValueError(
                "sensor-clearance policy requires the exact "
                f"{expected_safety_observation_size}-value driving observation; "
                f"received {len(initial_observation)}"
            )

        self.population = [
            self._new_random_member() for _ in range(self.config.population_size)
        ]
        self._current_index = 0
        self._member_runtimes: list[_EvaluationRuntime] = []
        self._observation = np.asarray(initial_observation, dtype=np.float32)
        self._evaluation_return = 0.0
        self._evaluation_steps = 0
        self._evaluation_losses: list[float] = []
        self._last_reward = 0.0
        self._last_info: dict[str, object] = {}
        # Every member in one generation receives the same seeded spawn.  A
        # successful random-origin lap is latched until the generation ends,
        # preventing later members from receiving an easier 80%-start-line
        # distribution than the policies evaluated before them.
        self._generation_curriculum_ready = bool(self.env.curriculum_ready)
        self._pending_curriculum_unlock = False

        self._current_champion: ChampionSnapshot | None = None
        self._current_champion_agent: DrivingDQNAgent | None = None
        self._best_champion: ChampionSnapshot | None = None
        self._best_champion_agent: DrivingDQNAgent | None = None
        # Remains available between generations, so an interactive race never
        # loses the most recently established generation champion.
        self._race_champion: ChampionSnapshot | None = None
        self._race_champion_agent: DrivingDQNAgent | None = None
        self._start_generation_runtime()

    @property
    def algorithm(self) -> EvolutionAlgorithm:
        return self.config.algorithm

    @property
    def current_member_index(self) -> int | None:
        return (
            self._current_index
            if 0 <= self._current_index < len(self.population)
            else None
        )

    @property
    def current_member(self) -> PopulationMember | None:
        index = self.current_member_index
        return None if index is None else self.population[index]

    @property
    def current_agent(self) -> DrivingDQNAgent | None:
        member = self.current_member
        return None if member is None else member.agent

    @property
    def observation(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._observation)

    @property
    def member_environments(self) -> tuple[DrivingEnv, ...]:
        """Training-owned environments in stable population order.

        Callers may inspect these environments for rendering, but must not
        step or reset them; the trainer is their sole mutation owner.
        """

        return tuple(self._member_envs)

    @property
    def member_observations(self) -> tuple[tuple[float, ...], ...]:
        """Latest policy observation for every member in population order."""

        return tuple(
            tuple(float(value) for value in runtime.observation)
            for runtime in self._member_runtimes
        )

    @property
    def active_member_indices(self) -> tuple[int, ...]:
        """Unevaluated population indices advanced by the next lockstep tick."""

        return tuple(
            index
            for index, member in enumerate(self.population)
            if not member.evaluated
        )

    @property
    def environment_decisions(self) -> int:
        """Total scored member transitions across every generation."""

        return self._environment_decisions

    @property
    def member_runtime(self) -> tuple[dict[str, object], ...]:
        """Cheap aligned status rows for renderers and runtime facades."""

        active = set(self.active_member_indices)
        rows: list[dict[str, object]] = []
        for index, (member, runtime) in enumerate(
            zip(self.population, self._member_runtimes)
        ):
            safety = runtime.safety.snapshot()
            rows.append(
                {
                    "index": index,
                    "member_id": member.member_id,
                    "status": "active" if index in active else "evaluated",
                    "evaluation_step": runtime.steps,
                    "evaluation_return": runtime.total_reward,
                    "last_reward": runtime.last_reward,
                    "action": safety["executed_action"],
                    "raw_action": safety["proposed_action"],
                    "executed_action": safety["executed_action"],
                    "safety_intervened": safety["intervened"],
                    "safety": safety,
                    "result": member.result,
                }
            )
        return tuple(rows)

    @property
    def generation_complete(self) -> bool:
        return all(member.evaluated for member in self.population)

    @property
    def current_champion_snapshot(self) -> ChampionSnapshot | None:
        return self._current_champion

    @property
    def best_champion_snapshot(self) -> ChampionSnapshot | None:
        return self._best_champion

    @property
    def champion_snapshot(self) -> ChampionSnapshot | None:
        """Champion suitable for a live race, with between-generation fallback."""

        return self._current_champion or self._race_champion or self._best_champion

    @property
    def ranked_members(self) -> tuple[PopulationMember, ...]:
        """Evaluated members first, ordered deterministically on fitness ties."""

        return tuple(
            sorted(
                self.population,
                key=lambda member: (
                    member.result is None,
                    -member.fitness,
                    member.member_id,
                ),
            )
        )

    def champion_agent(self, *, best_ever: bool = False) -> DrivingDQNAgent:
        """Return an isolated clone safe for the human-versus-champion race."""

        if best_ever:
            source = self._best_champion_agent
        else:
            source = (
                self._current_champion_agent
                or self._race_champion_agent
                or self._best_champion_agent
            )
        if source is None:
            # Before the first evaluation, member zero is the only meaningful
            # current policy.  It is still returned as an isolated clone.
            source = self.population[0].agent
        return source.clone(seed=source.config.seed, include_optimizer=False)

    def reset(
        self, *, seed: int | None = None, restart_generation: bool = False
    ) -> tuple[float, ...]:
        """Restart every unfinished context on one shared scenario seed."""

        self._require_usable()
        if seed is not None:
            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed < self._SEED_LIMIT
            ):
                raise ValueError("seed must be an integer in the [0, 2**63) interval")
        if restart_generation:
            for member in self.population:
                member.result = None
            self._current_champion = None
            self._current_champion_agent = None
            self._pending_curriculum_unlock = False
        reset_seed = (
            self._generation_evaluation_seed(self.generation) if seed is None else seed
        )
        self._start_generation_runtime(seed=reset_seed, keep_evaluated=True)
        return self.observation

    def step(self) -> PopulationStep:
        """Advance every active member by one concurrent, lockstep physics frame."""

        return self.step_many(1)[0]

    def step_many(
        self,
        max_ticks: int,
        *,
        stop_after_generation: bool = False,
    ) -> tuple[PopulationStep, ...]:
        """Advance bounded population ticks with one submit per car and chunk.

        A worker owns one member for the duration of a short chunk, which avoids
        submitting and joining ``population_size`` futures on every physics
        frame.  Worker output is replayed by tick and merged in stable member
        order, so this is bit-equivalent to repeated :meth:`step` calls for both
        ``workers=1`` and a parallel pool.  A chunk never evaluates a member past
        termination or its generation budget.

        ``stop_after_generation`` is useful to an interactive loop with an exact
        generation limit: it returns immediately after the first evolution even
        when ``max_ticks`` had spare capacity.
        """

        self._require_usable()
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int):
            raise ValueError("max_ticks must be a positive integer")
        if max_ticks <= 0:
            raise ValueError("max_ticks must be a positive integer")

        started_at = perf_counter()
        starting_decisions = self._environment_decisions
        steps: list[PopulationStep] = []
        while len(steps) < max_ticks:
            active_indices = self.active_member_indices
            if not active_indices:
                if steps:
                    break
                raise RuntimeError(
                    "generation is complete; call evolve() before stepping"
                )
            generation = self.generation
            # Very long per-member tasks let Python-heavy ray casting monopolize
            # a worker and reduce fairness. Eight ticks amortizes submission
            # overhead while returning to the stable coordinator barrier often
            # enough for responsive telemetry and balanced workers.
            requested_ticks = min(
                max_ticks - len(steps),
                self.MAX_WORKER_CHUNK_TICKS,
            )
            batches = self._advance_active_member_batches(
                active_indices,
                requested_ticks,
            )
            batch_ticks = max((len(batch) for batch in batches), default=0)
            if batch_ticks <= 0:
                raise RuntimeError("population workers produced no transitions")

            for offset in range(batch_ticks):
                advances = tuple(
                    batch[offset] for batch in batches if offset < len(batch)
                )
                tick_active_indices = tuple(advance.index for advance in advances)
                population_step = self._merge_population_tick(
                    tick_active_indices,
                    advances,
                    generation=generation,
                )
                steps.append(population_step)
                if population_step.generation_completed:
                    break

            generation_ended = bool(steps[-1].generation_completed)
            if generation_ended and (stop_after_generation or not self.auto_evolve):
                break

        elapsed = max(perf_counter() - started_at, 1e-12)
        decisions = self._environment_decisions - starting_decisions
        self._last_batch_ticks = len(steps)
        self._last_batch_decisions = decisions
        self._last_batch_ms = elapsed * 1_000.0
        decision_sample = decisions / elapsed
        tick_sample = len(steps) / elapsed
        smoothing = 0.20
        self._decision_throughput = (
            decision_sample
            if self._decision_throughput <= 0.0
            else self._decision_throughput * (1.0 - smoothing)
            + decision_sample * smoothing
        )
        self._tick_throughput = (
            tick_sample
            if self._tick_throughput <= 0.0
            else self._tick_throughput * (1.0 - smoothing) + tick_sample * smoothing
        )
        return tuple(steps)

    def _merge_population_tick(
        self,
        active_indices: tuple[int, ...],
        advances: tuple[_MemberAdvance, ...],
        *,
        generation: int,
    ) -> PopulationStep:
        """Merge one worker-produced lockstep tick on the coordinator thread."""

        member_index = (
            self._current_index
            if self._current_index in active_indices
            else active_indices[0]
        )
        self._last_tick_member_count = len(advances)
        self._environment_decisions += len(advances)
        advances_by_index = {advance.index: advance for advance in advances}
        completed_results: list[EvaluationResult] = []
        focal_result: EvaluationResult | None = None
        for advance in advances:
            runtime = self._member_runtimes[advance.index]
            runtime.safety.observe(advance.safety_decision)
            self._safety_stats.observe(advance.safety_decision)
            runtime.observation = advance.next_state
            runtime.steps += 1
            runtime.total_reward += float(advance.env_result.reward)
            runtime.last_reward = float(advance.env_result.reward)
            runtime.last_info = dict(advance.env_result.info)
            if advance.loss is not None and math.isfinite(advance.loss):
                runtime.losses.append(advance.loss)
            if bool(advance.env_result.info.get("curriculum_lap_completed", False)):
                self._pending_curriculum_unlock = True
            if advance.done:
                result = self._finish_member(advance.index, advance)
                completed_results.append(result)
                if advance.index == member_index:
                    focal_result = result

        generation_record: GenerationRecord | None = None
        generation_completed = self.generation_complete
        evolved = False
        report_index = member_index
        report_result = focal_result
        if report_result is None and completed_results:
            completed_ids = {result.member_id for result in completed_results}
            report_index = next(
                index
                for index in active_indices
                if self.population[index].member_id in completed_ids
            )
            report_result = self.population[report_index].result
        report_member = self.population[report_index]
        report_advance = advances_by_index[report_index]
        self._last_safety_decision = report_advance.safety_decision
        if generation_completed and self.auto_evolve:
            generation_record = self.evolve()
            evolved = True
        elif generation_completed:
            self._sync_focal_aliases(member_index)
            self._current_index = len(self.population)
        else:
            self._sync_focal_aliases(self.active_member_indices[0])

        return PopulationStep(
            generation=generation,
            member_id=report_member.member_id,
            member_index=report_index,
            observation=tuple(float(value) for value in report_advance.state),
            action=report_advance.safety_decision.executed_action,
            reward=float(report_advance.env_result.reward),
            next_observation=tuple(float(value) for value in report_advance.next_state),
            terminated=bool(report_advance.env_result.terminated),
            truncated=bool(
                report_advance.env_result.truncated or report_advance.budget_reached
            ),
            member_completed=report_result is not None,
            generation_completed=generation_completed,
            evolved=evolved,
            result=report_result,
            generation_record=generation_record,
            info=dict(report_advance.env_result.info),
            member_results=tuple(completed_results),
            active_member_indices=active_indices,
            proposed_action=report_advance.safety_decision.proposed_action,
            executed_action=report_advance.safety_decision.executed_action,
            safety_intervened=report_advance.safety_decision.intervened,
            safety_reason=report_advance.safety_decision.reason,
        )

    def tournament_select(
        self, *, exclude_member_id: int | None = None
    ) -> PopulationMember:
        """Select the fittest member of a seeded, without-replacement sample."""

        candidates = [
            member
            for member in self.population
            if member.evaluated and member.member_id != exclude_member_id
        ]
        if not candidates:
            raise RuntimeError("tournament selection requires evaluated members")
        sample_size = min(self.config.tournament_size, len(candidates))
        indices = self._rng.choice(len(candidates), size=sample_size, replace=False)
        sampled = [candidates[int(index)] for index in np.atleast_1d(indices)]
        return min(sampled, key=lambda item: (-item.fitness, item.member_id))

    def crossover_agents(
        self,
        first: DrivingDQNAgent,
        second: DrivingDQNAgent,
        *,
        seed: int | None = None,
    ) -> DrivingDQNAgent:
        """Create one independent child using uniform or BLX-alpha crossover."""

        self._compatible_agents(first, second)
        child_seed = first.config.seed if seed is None else seed
        child = first.clone(seed=child_seed, include_optimizer=False)
        if self._rng.random() >= self.config.crossover_rate:
            return child
        with torch.no_grad():
            for child_parameter, first_parameter, second_parameter in zip(
                child.online_network.parameters(),
                first.online_network.parameters(),
                second.online_network.parameters(),
            ):
                first_values = first_parameter.detach().cpu().numpy()
                second_values = second_parameter.detach().cpu().numpy()
                if self.config.crossover == "uniform":
                    mask = self._rng.random(first_values.shape) < 0.5
                    values = np.where(mask, first_values, second_values)
                else:
                    distance = np.abs(first_values - second_values)
                    low = (
                        np.minimum(first_values, second_values)
                        - self.config.blend_alpha * distance
                    )
                    high = (
                        np.maximum(first_values, second_values)
                        + self.config.blend_alpha * distance
                    )
                    values = self._rng.uniform(low, high)
                child_parameter.copy_(
                    torch.as_tensor(
                        values,
                        dtype=child_parameter.dtype,
                        device=child_parameter.device,
                    )
                )
        child.sync_target()
        return child

    # Concise aliases make the operators convenient in lessons and notebooks.
    crossover = crossover_agents

    def mutate_agent(self, agent: DrivingDQNAgent) -> int:
        """Apply masked Gaussian mutation and return the number of changed genes."""

        changed = 0
        if self.config.mutation_rate == 0.0 or self.config.mutation_std == 0.0:
            return changed
        with torch.no_grad():
            for parameter in agent.online_network.parameters():
                shape = tuple(parameter.shape)
                mask = self._rng.random(shape) < self.config.mutation_rate
                count = int(np.count_nonzero(mask))
                if count == 0:
                    continue
                noise = self._rng.normal(0.0, self.config.mutation_std, size=shape)
                delta = np.where(mask, noise, 0.0)
                parameter.add_(
                    torch.as_tensor(
                        delta,
                        dtype=parameter.dtype,
                        device=parameter.device,
                    )
                )
                changed += count
        if changed:
            agent.sync_target()
        return changed

    mutate = mutate_agent

    def evolve(self) -> GenerationRecord:
        """Retain exact elites, create mutated children, and begin the next generation."""

        self._require_usable()
        if not self.generation_complete:
            raise RuntimeError(
                "every population member must be evaluated before evolve()"
            )
        self._refresh_champions()
        ranked = list(self.ranked_members)
        elites = ranked[: self.config.elite_count]
        fitness_values = np.asarray([member.fitness for member in ranked], dtype=float)
        record = GenerationRecord(
            generation=self.generation,
            best_fitness=float(fitness_values.max()),
            mean_fitness=float(fitness_values.mean()),
            median_fitness=float(np.median(fitness_values)),
            worst_fitness=float(fitness_values.min()),
            fitness_std=float(fitness_values.std()),
            champion_id=ranked[0].member_id,
            elite_ids=tuple(member.member_id for member in elites),
            population_size=len(ranked),
            genome_diversity=self._genome_diversity(self.population),
        )
        self.history.append(record)

        next_generation = self.generation + 1
        next_population: list[PopulationMember] = []
        for elite in elites:
            # No crossover and no mutation: this is strict elitism at the
            # genotype level.  Replay is intentionally not copied.
            elite_agent = elite.agent.clone(
                seed=self._member_seed(elite.member_id, next_generation),
                include_optimizer=False,
            )
            next_population.append(
                PopulationMember(
                    member_id=elite.member_id,
                    agent=elite_agent,
                    birth_generation=elite.birth_generation,
                    parent_ids=(elite.member_id,),
                )
            )

        while len(next_population) < self.config.population_size:
            first = self.tournament_select()
            second = self.tournament_select(exclude_member_id=first.member_id)
            member_id = self._take_member_id()
            child = self.crossover_agents(
                first.agent,
                second.agent,
                seed=self._member_seed(member_id, next_generation),
            )
            self.mutate_agent(child)
            next_population.append(
                PopulationMember(
                    member_id=member_id,
                    agent=child,
                    birth_generation=next_generation,
                    parent_ids=(first.member_id, second.member_id),
                )
            )

        self.population = next_population
        self.generation = next_generation
        self._current_index = 0
        self._current_champion = None
        self._current_champion_agent = None
        self._generation_curriculum_ready = (
            self._generation_curriculum_ready or self._pending_curriculum_unlock
        )
        self._pending_curriculum_unlock = False
        self._start_generation_runtime()
        return record

    def network_snapshot(self, *, champion: bool = False) -> dict[str, Any]:
        """Expose real weights and activations for a population visualizer."""

        agent = self.champion_agent() if champion else self.current_agent
        if agent is None:
            agent = self.champion_agent()
        return agent.network_snapshot(self._observation)

    def telemetry(self) -> dict[str, Any]:
        """Return bounded, serialization-friendly live population metrics."""

        evaluated = [member for member in self.population if member.evaluated]
        fitnesses = np.asarray([member.fitness for member in evaluated], dtype=float)
        member = self.current_member
        learning = None if member is None else member.agent.telemetry(self._observation)
        replay_size = sum(len(item.agent.replay) for item in self.population)
        replay_capacity = sum(item.agent.replay.capacity for item in self.population)
        population = []
        active_indices = self.active_member_indices
        active_set = set(active_indices)
        safety_decisions = 0
        safety_interventions = 0
        for index, (item, runtime) in enumerate(
            zip(self.population, self._member_runtimes)
        ):
            summary = item.summary()
            safety = runtime.safety.snapshot()
            safety_decisions += runtime.safety.decisions
            safety_interventions += runtime.safety.interventions
            summary.update(
                {
                    "index": index,
                    "status": "active" if index in active_set else "evaluated",
                    "evaluation_step": runtime.steps,
                    "evaluation_return": runtime.total_reward,
                    "last_reward": runtime.last_reward,
                    "action": safety["executed_action"],
                    "raw_action": safety["proposed_action"],
                    "executed_action": safety["executed_action"],
                    "safety_intervened": safety["intervened"],
                    "safety": safety,
                    "observation": [float(value) for value in runtime.observation],
                    "curriculum_qualified": runtime.env.curriculum_ready,
                    "curriculum_generation_ready": (self._generation_curriculum_ready),
                }
            )
            population.append(summary)
        aggregate_safety = self._safety_stats.snapshot()
        current_index = self.current_member_index
        if current_index is not None:
            # The coordinator's last merged decision can belong to a member
            # that completed on this tick. Keep aggregate counters, but source
            # the visible decision from the member now selected everywhere
            # else in telemetry so action, observation, and network agree.
            current_safety = self._member_runtimes[current_index].safety.snapshot()
            for key in (
                "proposed_action",
                "executed_action",
                "intervened",
                "dangerous",
                "reason",
                "speed_ratio",
                "forward_clearance",
                "danger_threshold",
                "critical_clearance",
                "boundary_threshold",
                "projected_offset",
                "left_open_space",
                "right_open_space",
                "left_utility",
                "right_utility",
                "ray_clearances",
            ):
                aggregate_safety[key] = current_safety[key]
        elif self._last_safety_decision is not None:
            aggregate_safety.update(self._last_safety_decision.to_dict())
        safety_snapshot = {
            **aggregate_safety,
            "population_decisions": safety_decisions,
            "population_interventions": safety_interventions,
            "population_intervention_rate": (
                safety_interventions / safety_decisions
                if safety_decisions
                else 0.0
            ),
        }
        return {
            "algorithm": self.config.algorithm,
            "dqn_algorithm": self.dqn_config.algorithm,
            "generation": self.generation,
            "population_size": len(self.population),
            "parallel_workers": self.parallel_workers,
            "requested_parallel_workers": self.requested_parallel_workers,
            "worker_failed": self._worker_failure is not None,
            "worker_failure_type": (
                None
                if self._worker_failure is None
                else type(self._worker_failure).__name__
            ),
            "environment_decisions": self._environment_decisions,
            "last_tick_member_count": self._last_tick_member_count,
            "last_batch_ticks": self._last_batch_ticks,
            "last_batch_decisions": self._last_batch_decisions,
            "last_batch_ms": self._last_batch_ms,
            "decision_throughput": self._decision_throughput,
            "tick_throughput": self._tick_throughput,
            "active_member_indices": list(active_indices),
            "evaluated_members": len(evaluated),
            "current_member_index": self.current_member_index,
            "current_member_id": None if member is None else member.member_id,
            "evaluation_step": self._evaluation_steps,
            "evaluation_steps": self.config.evaluation_steps,
            "evaluation_progress": self._evaluation_steps
            / self.config.evaluation_steps,
            "evaluation_return": self._evaluation_return,
            "last_reward": self._last_reward,
            "proposed_action": safety_snapshot["proposed_action"],
            "executed_action": safety_snapshot["executed_action"],
            "safety_intervened": safety_snapshot["intervened"],
            "safety_prior": safety_snapshot,
            "population": population,
            "fitness": {
                "best": None if not len(fitnesses) else float(fitnesses.max()),
                "mean": None if not len(fitnesses) else float(fitnesses.mean()),
                "median": None if not len(fitnesses) else float(np.median(fitnesses)),
                "worst": None if not len(fitnesses) else float(fitnesses.min()),
                "std": None if not len(fitnesses) else float(fitnesses.std()),
            },
            "genetics": {
                "elite_count": self.config.elite_count,
                "tournament_size": self.config.tournament_size,
                "crossover": self.config.crossover,
                "crossover_rate": self.config.crossover_rate,
                "blend_alpha": self.config.blend_alpha,
                "mutation_rate": self.config.mutation_rate,
                "mutation_std": self.config.mutation_std,
                "sampled_genome_diversity": self._genome_diversity(self.population),
            },
            "champion": (
                None
                if self.champion_snapshot is None
                else self.champion_snapshot.to_dict()
            ),
            "best_champion": (
                None if self._best_champion is None else self._best_champion.to_dict()
            ),
            "history": [record.to_dict() for record in self.history],
            "learning": learning,
            "memory": {
                "transitions": replay_size,
                "capacity": replay_capacity,
                "fill_ratio": replay_size / replay_capacity if replay_capacity else 0.0,
            },
            "curriculum": {
                "generation_ready": self._generation_curriculum_ready,
                "pending_unlock": self._pending_curriculum_unlock,
            },
            "environment": self.env.telemetry(),
        }

    def state_dict(self) -> dict[str, Any]:
        """Compact checkpoint state; replay transitions are intentionally excluded."""

        self._require_usable()
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "evolution_config": self.config.to_dict(),
            "dqn_config": self.dqn_config.to_dict(),
            # The curriculum is environment state rather than policy state.
            # Persist it explicitly so a resumed population does not forget
            # that it already demonstrated a complete lap from a random pose.
            "environment_curriculum": {
                "unlocked": self._generation_curriculum_ready,
                "generation_ready": self._generation_curriculum_ready,
                "pending_unlock": self._pending_curriculum_unlock,
            },
            "generation": self.generation,
            "next_member_id": self._next_member_id,
            "environment_decisions": self._environment_decisions,
            "rng_state": deepcopy(self._rng.bit_generator.state),
            "population": [
                {
                    "member_id": member.member_id,
                    "birth_generation": member.birth_generation,
                    "parent_ids": list(member.parent_ids),
                    "result": (
                        None if member.result is None else member.result.to_dict()
                    ),
                    "agent": member.agent.state_dict(),
                }
                for member in self.population
            ],
            "history": [record.to_dict() for record in self.history],
            "best_champion": self._champion_state(
                self._best_champion, self._best_champion_agent
            ),
            "race_champion": self._champion_state(
                self._race_champion, self._race_champion_agent
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a population and restart its first unfinished evaluation."""

        self._require_usable()
        if int(state.get("checkpoint_version", -1)) != self.CHECKPOINT_VERSION:
            raise ValueError("unsupported population checkpoint version")
        saved_evolution = EvolutionConfig.from_dict(state["evolution_config"])
        saved_dqn = DQNConfig.from_dict(state["dqn_config"])
        if saved_evolution != self.config:
            raise ValueError("population checkpoint EvolutionConfig is incompatible")
        # The agent owns the one supported architecture bridge (the legacy
        # five-ray input expands into the denser current fan). Keep the outer
        # population envelope strict for every other mismatch while allowing
        # each member payload to pass through that same validated migration.
        if not DrivingDQNAgent.checkpoint_config_compatible(
            self.dqn_config,
            saved_dqn,
        ):
            raise ValueError("population checkpoint DQN architecture is incompatible")

        population_payload = list(state["population"])
        if len(population_payload) != self.config.population_size:
            raise ValueError("population checkpoint has an unexpected member count")
        population: list[PopulationMember] = []
        for item in population_payload:
            member_id = int(item["member_id"])
            agent = DrivingDQNAgent(
                replace(
                    self.dqn_config,
                    seed=self._member_seed(member_id, int(state["generation"])),
                )
            )
            agent.load_state_dict(item["agent"])
            result_payload = item.get("result")
            population.append(
                PopulationMember(
                    member_id=member_id,
                    agent=agent,
                    birth_generation=int(item["birth_generation"]),
                    parent_ids=tuple(int(value) for value in item["parent_ids"]),
                    result=(
                        None
                        if result_payload is None
                        else EvaluationResult.from_dict(result_payload)
                    ),
                )
            )

        self.population = population
        self.generation = int(state["generation"])
        self._next_member_id = int(state["next_member_id"])
        self._environment_decisions = int(
            state.get(
                "environment_decisions",
                sum(
                    member.result.steps
                    for member in population
                    if member.result is not None
                ),
            )
        )
        self._last_tick_member_count = 0
        self._last_batch_ticks = 0
        self._last_batch_decisions = 0
        self._last_batch_ms = 0.0
        self._decision_throughput = 0.0
        self._tick_throughput = 0.0
        # Safety telemetry is intentionally ephemeral and absent from the
        # checkpoint schema; resumed policies start a fresh observability window.
        self._safety_stats = SensorClearanceStats()
        self._last_safety_decision = None
        self._rng.bit_generator.state = deepcopy(state["rng_state"])
        self.history = deque(
            (GenerationRecord.from_dict(item) for item in state.get("history", ())),
            maxlen=self.config.history_capacity,
        )
        self._best_champion, self._best_champion_agent = self._restore_champion(
            state.get("best_champion")
        )
        self._race_champion, self._race_champion_agent = self._restore_champion(
            state.get("race_champion")
        )
        self._current_champion = None
        self._current_champion_agent = None
        self._refresh_champions()
        curriculum = dict(state.get("environment_curriculum", {}))
        self._generation_curriculum_ready = bool(
            curriculum.get(
                "generation_ready",
                curriculum.get("unlocked", curriculum.get("ready", False)),
            )
        )
        self._pending_curriculum_unlock = bool(curriculum.get("pending_unlock", False))
        # Version-one checkpoints never stored partial environment contexts.
        # Keep that portable contract: retain completed results and restart all
        # unfinished members from the generation's one common scenario seed.
        self._start_generation_runtime(keep_evaluated=True)

    def save(self, path: str | Path) -> Path:
        self._require_usable()
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(self.state_dict(), temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    def load(self, path: str | Path) -> None:
        self._require_usable()
        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"population checkpoint not found: {checkpoint}")
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older Torch
            state = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state)

    def close(self) -> None:
        """Release persistent evaluation threads; safe to call repeatedly."""

        if self._closed:
            return
        self._closed = True
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "PopulationTrainer":
        self._require_usable()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive interpreter cleanup
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _new_random_member(self) -> PopulationMember:
        member_id = self._take_member_id()
        agent = DrivingDQNAgent(
            replace(
                self.dqn_config,
                seed=self._member_seed(member_id, self.generation),
            )
        )
        return PopulationMember(member_id, agent, self.generation)

    def _take_member_id(self) -> int:
        member_id = self._next_member_id
        self._next_member_id += 1
        return member_id

    def _member_seed(self, member_id: int, generation: int) -> int:
        return int(
            (self.config.seed + 1_000_003 * (member_id + 1) + 97_409 * (generation + 1))
            % self._SEED_LIMIT
        )

    def _evaluation_seed(self, generation: int, member_index: int) -> int:
        return int(
            (
                self.config.seed
                + 4_294_967_291 * (generation + 1)
                + 65_537 * (member_index + 1)
            )
            % self._SEED_LIMIT
        )

    def _generation_evaluation_seed(self, generation: int) -> int:
        """One common scenario seed keeps fitness comparable within a generation."""

        return self._evaluation_seed(generation, 0)

    @staticmethod
    def _clone_environment(source: DrivingEnv, *, seed: int) -> DrivingEnv:
        """Create one isolated simulation with the source environment's setup."""

        clone = DrivingEnv(
            source.circuit,
            build=source.vehicle.build,
            seed=seed,
            fixed_dt=source.fixed_dt,
            max_steps=source.max_steps,
            random_start_curriculum=source.random_start_curriculum,
        )
        clone.load_curriculum_state(source.curriculum_state())
        return clone

    def _start_generation_runtime(
        self,
        *,
        seed: int | None = None,
        keep_evaluated: bool = False,
    ) -> None:
        """Reset independent contexts to one reproducible generation scenario."""

        if not keep_evaluated:
            for member in self.population:
                member.result = None
        scenario_seed = (
            self._generation_evaluation_seed(self.generation) if seed is None else seed
        )
        runtimes: list[_EvaluationRuntime] = []
        for member_env in self._member_envs:
            member_env.load_curriculum_state(
                {"unlocked": self._generation_curriculum_ready}
            )
            observation = member_env.reset(seed=scenario_seed)
            if len(observation) != self.dqn_config.observation_size:
                raise ValueError(
                    "DrivingEnv observation size does not match DQNConfig: "
                    f"{len(observation)} != {self.dqn_config.observation_size}"
                )
            runtimes.append(
                _EvaluationRuntime(
                    env=member_env,
                    observation=np.asarray(observation, dtype=np.float32),
                )
            )
        self._member_runtimes = runtimes
        active = self.active_member_indices
        focal_index = active[0] if active else 0
        self._sync_focal_aliases(focal_index)
        if not active:
            self._current_index = len(self.population)

    def _sync_focal_aliases(self, index: int) -> None:
        """Keep legacy scalar accessors aligned with a deterministic member."""

        runtime = self._member_runtimes[index]
        self._current_index = index
        self.env = runtime.env
        self._observation = runtime.observation
        self._evaluation_return = runtime.total_reward
        self._evaluation_steps = runtime.steps
        self._evaluation_losses = runtime.losses
        self._last_reward = runtime.last_reward
        self._last_info = runtime.last_info

    def _advance_active_member_batches(
        self,
        active_indices: tuple[int, ...],
        max_ticks: int,
    ) -> tuple[tuple[_MemberAdvance, ...], ...]:
        """Run isolated member chunks concurrently and collect in stable order."""

        if self.parallel_workers == 1 or len(active_indices) == 1:
            batches: list[tuple[_MemberAdvance, ...]] = []
            try:
                for index in active_indices:
                    batches.append(self._advance_member_many(index, max_ticks))
            except BaseException as error:
                self._fail_after_worker_error(error)
            return tuple(batches)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.parallel_workers,
                thread_name_prefix="driving-population",
            )
        try:
            futures = {
                index: self._executor.submit(
                    self._advance_member_many,
                    index,
                    max_ticks,
                )
                for index in active_indices
            }
        except BaseException as error:
            self._fail_after_worker_error(error)
        batches = []
        first_error: BaseException | None = None
        for index in active_indices:
            try:
                batches.append(futures[index].result())
            except BaseException as error:  # wait for every in-flight context
                if first_error is None:
                    first_error = error
        if first_error is not None:
            self._fail_after_worker_error(first_error)
        return tuple(batches)

    def _fail_after_worker_error(self, error: BaseException) -> NoReturn:
        """Permanently stop after a partially executed population tick.

        A successful sibling may already have mutated its private environment
        or learner. Because the coordinator has not committed any runtime rows,
        continuing or saving that mixed state would be dishonest. Waiting for
        every submitted task and making the trainer fail-stop preserves the
        all-or-nothing tick contract.
        """

        self._worker_failure = error
        self._closed = True
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        if not isinstance(error, Exception):
            # Preserve process-level interrupts such as Ctrl-C after making the
            # partially executed tick impossible to resume.
            raise error
        raise RuntimeError(
            "population member evaluation failed; trainer is closed"
        ) from error

    def _require_usable(self) -> None:
        if self._worker_failure is not None:
            raise RuntimeError(
                "population trainer cannot continue after a worker failure"
            ) from self._worker_failure
        if self._closed:
            raise RuntimeError("population trainer is closed")

    def _advance_member_many(
        self,
        index: int,
        max_ticks: int,
    ) -> tuple[_MemberAdvance, ...]:
        """Advance one member chunk without touching another member context."""

        member = self.population[index]
        runtime = self._member_runtimes[index]
        state = runtime.observation.copy()
        completed_steps = runtime.steps
        explore = self.config.algorithm == "genetic_dqn"
        advances: list[_MemberAdvance] = []
        for _ in range(max_ticks):
            proposed_action = member.agent.select_action(state, explore=explore)
            safety_decision = self.clearance_policy.decide(
                state,
                proposed_action,
            )
            executed_action = safety_decision.executed_action
            env_result = runtime.env.step(executed_action)
            next_state = np.asarray(env_result.observation, dtype=np.float32)
            completed_steps += 1
            budget_reached = completed_steps >= self.config.evaluation_steps
            done = bool(
                env_result.terminated or env_result.truncated or budget_reached
            )
            loss: float | None = None
            if self.config.algorithm == "genetic_dqn":
                observed_loss = member.agent.observe(
                    state,
                    executed_action,
                    env_result.reward,
                    next_state,
                    done,
                )
                if observed_loss is not None and math.isfinite(float(observed_loss)):
                    loss = float(observed_loss)
            advances.append(
                _MemberAdvance(
                    index=index,
                    state=state,
                    safety_decision=safety_decision,
                    env_result=env_result,
                    next_state=next_state,
                    budget_reached=budget_reached,
                    done=done,
                    loss=loss,
                )
            )
            state = next_state
            if done:
                break
        return tuple(advances)

    def _finish_member(
        self,
        index: int,
        advance: _MemberAdvance,
    ) -> EvaluationResult:
        member = self.population[index]
        runtime = self._member_runtimes[index]
        env_result = advance.env_result
        info = env_result.info
        result = EvaluationResult(
            generation=self.generation,
            member_id=member.member_id,
            # The environment reward already combines progress, road holding,
            # speed, collisions, and lap completion without double counting.
            fitness=float(runtime.total_reward),
            total_reward=float(runtime.total_reward),
            steps=runtime.steps,
            laps=int(info.get("laps", runtime.env.laps)),
            # Random-origin episodes report absolute circuit position as
            # ``progress`` and distance travelled from their own origin as
            # ``episode_lap_progress``. Fitness summaries must use the latter so a
            # late-track spawn is not mistaken for a nearly completed loop.
            progress=float(info.get("episode_lap_progress", info.get("progress", 0.0))),
            collisions=int(runtime.env.collisions),
            terminated=bool(env_result.terminated),
            truncated=bool(env_result.truncated or advance.budget_reached),
            mean_loss=(float(np.mean(runtime.losses)) if runtime.losses else 0.0),
            training_updates=len(runtime.losses),
        )
        member.result = result
        self._consider_champion(member)
        return result

    def _consider_champion(self, member: PopulationMember) -> None:
        if member.result is None:
            return
        snapshot = ChampionSnapshot(
            generation=self.generation,
            member_id=member.member_id,
            fitness=member.result.fitness,
            result=member.result,
        )
        if self._better(snapshot, self._current_champion):
            self._current_champion = snapshot
            self._current_champion_agent = member.agent.clone(
                seed=member.agent.config.seed,
                include_optimizer=False,
            )
            self._race_champion = snapshot
            self._race_champion_agent = self._current_champion_agent.clone(
                seed=self._current_champion_agent.config.seed,
                include_optimizer=False,
            )
        if self._better(snapshot, self._best_champion):
            self._best_champion = snapshot
            self._best_champion_agent = member.agent.clone(
                seed=member.agent.config.seed,
                include_optimizer=False,
            )

    @staticmethod
    def _better(candidate: ChampionSnapshot, current: ChampionSnapshot | None) -> bool:
        return current is None or (candidate.fitness, -candidate.member_id) > (
            current.fitness,
            -current.member_id,
        )

    def _refresh_champions(self) -> None:
        for member in self.population:
            self._consider_champion(member)

    @staticmethod
    def _compatible_agents(first: DrivingDQNAgent, second: DrivingDQNAgent) -> None:
        first_signature = (
            first.config.observation_size,
            first.config.action_size,
            first.config.hidden_sizes,
        )
        second_signature = (
            second.config.observation_size,
            second.config.action_size,
            second.config.hidden_sizes,
        )
        if first_signature != second_signature:
            raise ValueError("parent network architectures must match")

    def _genome_sample(self, agent: DrivingDQNAgent) -> np.ndarray:
        remaining = self.GENOME_SAMPLE_LIMIT
        chunks: list[np.ndarray] = []
        for parameter in agent.online_network.parameters():
            values = parameter.detach().cpu().numpy().reshape(-1)
            take = min(remaining, len(values))
            if take:
                chunks.append(values[:take].astype(np.float64, copy=False))
                remaining -= take
            if remaining == 0:
                break
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)

    def _genome_diversity(self, members: list[PopulationMember]) -> float:
        if len(members) < 2:
            return 0.0
        samples = np.stack([self._genome_sample(member.agent) for member in members])
        return float(np.mean(np.std(samples, axis=0)))

    @staticmethod
    def _champion_state(
        snapshot: ChampionSnapshot | None,
        agent: DrivingDQNAgent | None,
    ) -> dict[str, Any] | None:
        if snapshot is None or agent is None:
            return None
        return {"snapshot": snapshot.to_dict(), "agent": agent.state_dict()}

    def _restore_champion(
        self, payload: Mapping[str, Any] | None
    ) -> tuple[ChampionSnapshot | None, DrivingDQNAgent | None]:
        if payload is None:
            return None, None
        snapshot = ChampionSnapshot.from_dict(payload["snapshot"])
        agent = DrivingDQNAgent(
            replace(
                self.dqn_config,
                seed=self._member_seed(snapshot.member_id, snapshot.generation),
            )
        )
        agent.load_state_dict(payload["agent"], load_optimizer=False)
        return snapshot, agent


# ``PopulationSession`` reads naturally in the playable view while
# ``PopulationTrainer`` remains precise in learning-oriented code.
PopulationSession = PopulationTrainer


__all__ = (
    "ChampionSnapshot",
    "CrossoverMode",
    "EvaluationResult",
    "EvolutionAlgorithm",
    "EvolutionConfig",
    "GenerationRecord",
    "GenerationStats",
    "MemberResult",
    "PopulationMember",
    "PopulationSession",
    "PopulationStep",
    "PopulationTrainer",
    "normalize_evolution_algorithm",
)
