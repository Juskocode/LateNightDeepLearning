"""Training orchestration and a side-effect-free human/champion race.

The learning algorithms intentionally live below the Pygame presentation layer so
headless experiments, tests, and the interactive dashboard all execute the exact
same fixed-step simulation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Sequence

import numpy as np
import torch

from .environment import DrivingAction, DrivingEnv, StepResult
from .learning_health import build_learning_health
from .ml import (
    DQNConfig,
    DrivingDQNAgent,
    default_population_dqn_config,
)
from .sensor_clearance import SensorClearancePolicy, SensorClearanceStats
from .vehicle import CarBuild, DriverControls


LearningAlgorithm = Literal["dqn", "double_dqn", "genetic", "genetic_dqn"]

ACTION_LABELS = tuple(action.name.replace("_", " ").title() for action in DrivingAction)


@dataclass(frozen=True, slots=True)
class LearningRuntimeConfig:
    """Small, validated contract shared by the CLI and visual trainer."""

    algorithm: LearningAlgorithm = "genetic_dqn"
    circuit: str = "harbor_loop"
    seed: int = 7
    evaluation_steps: int = 900
    population_size: int = 8
    elite_count: int = 2
    tournament_size: int = 2
    crossover: Literal["uniform", "blend"] = "uniform"
    crossover_rate: float = 0.65
    blend_alpha: float = 0.20
    mutation_rate: float = 0.08
    mutation_std: float = 0.055
    parallel_workers: int | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in ("dqn", "double_dqn", "genetic", "genetic_dqn"):
            raise ValueError(
                f"Unsupported driving learning algorithm: {self.algorithm}"
            )
        integer_fields = (
            "seed",
            "evaluation_steps",
            "population_size",
            "elite_count",
            "tournament_size",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in the [0, 2**63) interval")
        if self.evaluation_steps <= 0:
            raise ValueError("evaluation_steps must be positive")
        if self.population_size < 2:
            raise ValueError("population_size must be at least two")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be in [1, population_size)")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be in [1, population_size]")
        if self.crossover not in ("uniform", "blend"):
            raise ValueError("crossover must be 'uniform' or 'blend'")
        if not isinstance(self.circuit, str) or not self.circuit.strip():
            raise ValueError("circuit must be a non-empty string")
        for name in ("crossover_rate", "mutation_rate"):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"{name} must be finite and in [0, 1]")
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("blend_alpha", "mutation_std"):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"{name} must be finite and non-negative")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.parallel_workers is not None and (
            isinstance(self.parallel_workers, bool)
            or not isinstance(self.parallel_workers, int)
            or self.parallel_workers <= 0
        ):
            raise ValueError("parallel_workers must be a positive integer or None")


class DrivingLearningSession:
    """Uniform facade over episode DQN and population neuroevolution.

    A generation means one evaluated episode for the standalone DQN modes, and
    one full population sweep for genetic modes.  This keeps charts and headless
    stopping conditions meaningful across every algorithm.
    """

    def __init__(
        self,
        config: LearningRuntimeConfig | None = None,
        *,
        build: CarBuild | None = None,
        dqn_config: DQNConfig | None = None,
    ):
        self.config = config or LearningRuntimeConfig()
        self.build = build or CarBuild()
        self._last_event = "session_started"
        self._checkpoint_path: Path | None = None
        self._population_trainer: Any | None = None
        self._uses_population_default_dqn = False
        self._loss_history: deque[float] = deque(maxlen=300)
        self._epsilon_history: deque[float] = deque(maxlen=300)
        self._environment_decisions = 0
        self._health_decision_origin = 0
        self._health_update_origin = 0
        self._health_clip_origin = 0
        self._health_nonfinite_origin = 0
        self._wall_contact_decisions = 0
        self._collision_loop_terminations = 0
        self.clearance_policy = SensorClearancePolicy()
        self._safety_stats = SensorClearanceStats()

        if self.config.algorithm in ("genetic", "genetic_dqn"):
            self._init_population(dqn_config)
            return

        algorithm = self.config.algorithm
        if dqn_config is None:
            dqn_config = DQNConfig(algorithm=algorithm, seed=self.config.seed)
        elif dqn_config.algorithm != algorithm:
            dqn_config = replace(dqn_config, algorithm=algorithm)
        self.agent = DrivingDQNAgent(dqn_config)
        self.env = DrivingEnv(
            self.config.circuit,
            build=self.build,
            seed=self.config.seed,
            max_steps=self.config.evaluation_steps,
            random_start_curriculum=True,
        )
        self.observation = self.env.observation()
        self.generation = 1
        self.episode_return = 0.0
        self.best_fitness = -math.inf
        self._champion = self.agent.clone(seed=self.config.seed + 1)
        self.generation_history: list[dict[str, float | int]] = []
        self._episode_fitness: list[float] = []

    def _init_population(self, dqn_config: DQNConfig | None) -> None:
        # Imported lazily so the standalone DQN remains usable in minimal
        # educational examples that do not need population evolution.
        from .ml.evolution import EvolutionConfig, PopulationTrainer

        base_algorithm = "double_dqn"
        if dqn_config is None:
            # Population members live for a bounded 900-step episode by
            # default. The generic 512-step warmup spends most of that lifetime
            # collecting without learning, then trains every frame. Start
            # replay early and spread updates across four transitions instead:
            # better early feedback with substantially lower interactive CPU
            # cost. Population exploration also needs its own lifetime scale:
            # the generic 40k decay would make each freshly cloned 900-step
            # member almost entirely random. Explicit caller configurations
            # remain untouched.
            dqn_config = default_population_dqn_config(
                algorithm=base_algorithm,
                evaluation_steps=self.config.evaluation_steps,
                seed=self.config.seed,
            )
            self._uses_population_default_dqn = True
        evolution = EvolutionConfig(
            algorithm=self.config.algorithm,
            population_size=self.config.population_size,
            elite_count=self.config.elite_count,
            tournament_size=self.config.tournament_size,
            evaluation_steps=self.config.evaluation_steps,
            crossover=self.config.crossover,
            crossover_rate=self.config.crossover_rate,
            blend_alpha=self.config.blend_alpha,
            mutation_rate=self.config.mutation_rate,
            mutation_std=self.config.mutation_std,
            seed=self.config.seed,
        )
        population_env = DrivingEnv(
            self.config.circuit,
            build=self.build,
            seed=self.config.seed,
            max_steps=self.config.evaluation_steps,
            random_start_curriculum=True,
        )
        self._population_trainer = PopulationTrainer(
            evolution,
            dqn_config=dqn_config,
            env=population_env,
            parallel_workers=self.config.parallel_workers,
        )
        self.env = self._population_trainer.env
        self.agent = self._population_trainer.current_agent
        self.observation = self.env.observation()

    @property
    def is_population(self) -> bool:
        return self._population_trainer is not None

    @property
    def completed_generations(self) -> int:
        if self.is_population:
            return len(self._population_trainer.history)
        return len(self.generation_history)

    @property
    def current_generation(self) -> int:
        """Generation represented by the policies currently being evaluated."""

        if self.is_population:
            return int(self._population_trainer.generation)
        return int(self.generation)

    @property
    def environment_decisions(self) -> int:
        """Total scored environment transitions consumed by this session."""

        if self.is_population:
            return int(self._population_trainer.environment_decisions)
        return self._environment_decisions

    def population_policy_clones(
        self,
        *,
        max_cars: int = 12,
        reusable_policy_clones: Sequence[DrivingDQNAgent] = (),
    ) -> tuple[tuple[int, DrivingDQNAgent], ...]:
        """Return bounded, isolated policies for presentation-only rollouts.

        No caller receives a training-owned agent, which prevents a renderer or
        comparison view from changing replay, optimizer, fitness, or selection.
        Existing presentation-owned clones can be supplied when a generation
        changes. Compatible networks are updated in place, avoiding repeated
        optimizer and replay-buffer allocation while retaining independent
        parameter storage.
        """

        if isinstance(max_cars, bool) or not isinstance(max_cars, int):
            raise ValueError("max_cars must be an integer")
        if max_cars <= 0:
            raise ValueError("max_cars must be positive")
        reusable = tuple(reusable_policy_clones)
        if any(not isinstance(agent, DrivingDQNAgent) for agent in reusable):
            raise TypeError(
                "reusable_policy_clones must contain DrivingDQNAgent instances"
            )
        if self.is_population:
            training_agents = tuple(
                member.agent for member in self._population_trainer.population
            )
            sources = tuple(
                (member.member_id, member.agent)
                for member in self._population_trainer.population[:max_cars]
            )
        else:
            training_agents = (self.agent,)
            sources = ((0, self.agent),)
        training_agent_ids = {id(agent) for agent in training_agents}
        if any(id(agent) in training_agent_ids for agent in reusable):
            raise ValueError(
                "reusable_policy_clones cannot contain training-owned agents"
            )

        clones: list[tuple[int, DrivingDQNAgent]] = []
        for index, (member_id, source) in enumerate(sources):
            clone = reusable[index] if index < len(reusable) else None
            compatible = (
                clone is not None
                and clone.online_network.architecture
                == source.online_network.architecture
            )
            if compatible:
                # ``load_state_dict`` copies tensor values into the already
                # independent storage; no tensor or optimizer is shared with
                # the population trainer.
                source_weights = source.online_network.state_dict()
                clone.online_network.load_state_dict(source_weights)
                clone.target_network.load_state_dict(source_weights)
                clone.target_network.eval()
            else:
                clone = source.clone(
                    seed=source.config.seed,
                    include_optimizer=False,
                )
            clones.append((int(member_id), clone))
        return tuple(clones)

    def step(self) -> Any:
        """Advance exactly one deterministic environment step."""

        if self.is_population:
            result = self._population_trainer.step()
            self.env = self._population_trainer.env
            self.agent = self._population_trainer.current_agent
            self.observation = self._population_trainer.observation
            if result.evolved:
                self._last_event = "generation_evolved"
            elif result.generation_completed:
                self._last_event = "generation_complete"
            elif result.member_completed or result.member_results:
                self._last_event = "member_complete"
            else:
                self._last_event = "population_step"
            self._record_learning_trace(result)
            return result

        state = self.observation
        proposed_action = self.agent.select_action(state, explore=True)
        safety_decision = self.clearance_policy.decide(state, proposed_action)
        self._safety_stats.observe(safety_decision)
        executed_action = safety_decision.executed_action
        result = self.env.step(executed_action)
        self._environment_decisions += 1
        self._wall_contact_decisions += int(
            bool(result.info.get("wall_contact_active", False))
        )
        done = result.terminated or result.truncated
        self._collision_loop_terminations += int(
            done and bool(result.info.get("collision_looped", False))
        )
        # Replay receives the action that caused the transition. Crediting an
        # overridden unsafe proposal with the safety prior's reward would teach
        # the wrong Q-value and invite reliance on the filter.
        self.agent.observe(
            state,
            executed_action,
            result.reward,
            result.observation,
            done,
        )
        self.episode_return += result.reward
        self.observation = result.observation
        self._last_event = "training_step"
        if done:
            self._finish_dqn_episode()
        self._record_learning_trace()
        return result

    def step_many(
        self,
        max_steps: int,
        *,
        stop_after_generation: bool = False,
    ) -> tuple[Any, ...]:
        """Advance a bounded training chunk with one population pool barrier.

        Population modes amortize worker scheduling across the requested ticks.
        Standalone DQN keeps the same public contract by executing ordinary
        deterministic steps. The returned tuple contains one result per
        simulation tick, making generation and step limits exact.
        """

        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise ValueError("max_steps must be a positive integer")
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not self.is_population:
            starting_generation = self.current_generation
            results: list[Any] = []
            for _ in range(max_steps):
                results.append(self.step())
                if (
                    stop_after_generation
                    and self.current_generation != starting_generation
                ):
                    break
            return tuple(results)

        results = self._population_trainer.step_many(
            max_steps,
            stop_after_generation=stop_after_generation,
        )
        self.env = self._population_trainer.env
        self.agent = self._population_trainer.current_agent
        self.observation = self._population_trainer.observation
        for result in results:
            if result.evolved:
                self._last_event = "generation_evolved"
            elif result.generation_completed:
                self._last_event = "generation_complete"
            elif result.member_completed or result.member_results:
                self._last_event = "member_complete"
            else:
                self._last_event = "population_step"
        # The trainer may have crossed an evolution boundary during this
        # chunk. Record one honest snapshot from the final live agent instead
        # of attributing that agent's epsilon/loss to every earlier tick.
        self._record_learning_trace()
        return tuple(results)

    def _record_learning_trace(self, population_step: Any | None = None) -> None:
        if self.agent is None:
            return
        loss = float(self.agent.last_loss)
        if population_step is not None and population_step.result is not None:
            loss = float(population_step.result.mean_loss)
        if math.isfinite(loss):
            self._loss_history.append(loss)
        epsilon = (
            0.0 if self.config.algorithm == "genetic" else float(self.agent.epsilon)
        )
        if math.isfinite(epsilon):
            self._epsilon_history.append(epsilon)

    def _finish_dqn_episode(self) -> None:
        fitness = float(self.episode_return)
        self._episode_fitness.append(fitness)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self._champion = self.agent.clone(seed=self.config.seed + self.generation)
            self._last_event = "new_champion"
        else:
            self._last_event = "episode_complete"
        self.generation_history.append(
            {
                "generation": self.generation,
                "best": fitness,
                "mean": fitness,
                "worst": fitness,
            }
        )
        self.generation += 1
        self.episode_return = 0.0
        self.observation = self.env.reset()

    def champion_agent(self) -> DrivingDQNAgent:
        """Return an isolated frozen policy safe for a parallel race."""

        if self.is_population:
            champion = self._population_trainer.champion_agent()
            return champion.clone(seed=self.config.seed + 99_991)
        return self._champion.clone(seed=self.config.seed + 99_991)

    def scored_population_telemetry(
        self,
        *,
        include_rays: bool = True,
        max_cars: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return real scored car states after the population step barrier.

        Genetic modes expose every training environment in stable population
        order. Standalone DQN has no concurrent population and returns an empty
        list. The optional bound is presentation-only; all members continue to
        train regardless of how many are rendered.
        """

        if not self.is_population:
            return []
        from .population_rollout import scored_population_telemetry

        return scored_population_telemetry(
            self,
            max_cars=max_cars,
            include_rays=include_rays,
        )

    def _epsilon_schedule_telemetry(self) -> dict[str, Any]:
        """Describe the effective proposal-exploration schedule analytically."""

        config = self.agent.config
        lifetime = max(1, int(self.config.evaluation_steps))
        decay = max(1, int(config.epsilon_decay_steps))
        schedule_step = max(0, int(self.agent.environment_steps))
        # Forecast the next evaluation-sized horizon from the learner's actual
        # schedule position. Sum clamped linear progress without allocating a
        # per-frame list on every dashboard refresh.
        linear_start = min(schedule_step, decay)
        linear_stop = min(schedule_step + lifetime, decay)
        linear_count = max(0, linear_stop - linear_start)
        progress_sum = linear_count * (linear_start + linear_stop - 1) / (2.0 * decay)
        progress_sum += lifetime - linear_count
        mean_epsilon = config.epsilon_start + (
            config.epsilon_end - config.epsilon_start
        ) * (progress_sum / lifetime)
        exploration_enabled = self.config.algorithm != "genetic"
        expected_exploration = mean_epsilon if exploration_enabled else 0.0
        current_epsilon = float(self.agent.epsilon) if exploration_enabled else 0.0
        return {
            "enabled": exploration_enabled,
            "population_default": bool(self._uses_population_default_dqn),
            "start": float(config.epsilon_start),
            "end": float(config.epsilon_end),
            "decay_steps": decay,
            "schedule_step": schedule_step,
            "evaluation_lifetime_steps": lifetime,
            "decay_scaled_to_lifetime": decay == lifetime,
            "current": current_epsilon,
            "expected_exploration_fraction": expected_exploration,
            "expected_greedy_fraction": 1.0 - expected_exploration,
        }

    def telemetry(self) -> dict[str, Any]:
        """Merge environment, learning, replay, and real-network state."""

        if self.is_population:
            raw = dict(self._population_trainer.telemetry())
            self.env = self._population_trainer.env
            self.agent = self._population_trainer.current_agent
            observation = self._population_trainer.observation
            learning = raw.get("learning") or self.agent.telemetry(observation)
            fitness = raw.get("fitness") or {}
            population = []
            current_index = raw.get("current_member_index")
            active_indices = {
                int(index) for index in raw.get("active_member_indices", ())
            }
            for index, member in enumerate(raw.get("population", ())):
                result = member.get("result") or {}
                member_fitness = member.get("fitness")
                runtime_status = member.get("status")
                if runtime_status == "active":
                    runtime_status = "evaluating"
                population.append(
                    {
                        "index": index,
                        "member_id": member.get("member_id", index),
                        "fitness": member_fitness,
                        "status": (
                            runtime_status
                            or (
                                "evaluating"
                                if index in active_indices
                                else (
                                    "evaluated" if member.get("evaluated") else "queued"
                                )
                            )
                        ),
                        "elite": False,
                        "laps": result.get("laps", 0),
                        "collisions": result.get("collisions", 0),
                        "parents": member.get("parent_ids", ()),
                        "evaluation_step": member.get("evaluation_step", 0),
                        "evaluation_return": member.get("evaluation_return", 0.0),
                        "action": member.get("action"),
                        "raw_action": member.get("raw_action"),
                        "executed_action": member.get("executed_action"),
                        "safety_intervened": bool(
                            member.get("safety_intervened", False)
                        ),
                        "safety": member.get("safety", {}),
                    }
                )
            history = [
                {
                    "generation": row["generation"],
                    "best": row["best_fitness"],
                    "mean": row["mean_fitness"],
                    "worst": row["worst_fitness"],
                    "diversity": row.get("genome_diversity", 0.0),
                }
                for row in raw.get("history", ())
            ]
            if history:
                elite_ids = set(self._population_trainer.history[-1].elite_ids)
                for item in population:
                    item["elite"] = item["member_id"] in elite_ids
            genetics = raw.get("genetics") or {}
            champion = raw.get("champion") or raw.get("best_champion") or {}
            current_best = fitness.get("best")
            if current_best is None:
                current_best = champion.get("fitness")
            current_mean = fitness.get("mean")
            if current_mean is None and history:
                current_mean = history[-1]["mean"]
            snapshot = {
                **raw,
                "completed_generations": len(history),
                "member_index": current_index,
                "episode_step": raw.get("evaluation_step", 0),
                "current_fitness": raw.get("evaluation_return", 0.0),
                "best_fitness": current_best,
                "mean_fitness": current_mean,
                "champion_member": champion.get("member_id", 0),
                "population": population,
                "generation_history": history,
                "event": self._last_event,
                "epsilon": learning.get("epsilon", 0.0),
                "loss": learning.get("last_loss", 0.0),
                "td_error": learning.get("mean_absolute_td_error", 0.0),
                "gradient_steps": learning.get("gradient_steps", 0),
                "mutation_rate": genetics.get("mutation_rate", 0.0),
                "mutation_std": genetics.get("mutation_std", 0.0),
                "crossover_rate": genetics.get("crossover_rate", 0.0),
                "elite_count": genetics.get("elite_count", 0),
            }
        else:
            observation = self.observation
            learning = self.agent.telemetry(observation)
            population = [
                {
                    "index": 0,
                    "fitness": self.episode_return,
                    "status": "evaluating",
                    "elite": self.best_fitness > -math.inf,
                }
            ]
            snapshot = {
                "algorithm": self.config.algorithm,
                "generation": self.generation,
                "completed_generations": len(self.generation_history),
                "member_index": 0,
                "population_size": 1,
                "episode_step": self.env.steps,
                "evaluation_steps": self.config.evaluation_steps,
                "current_fitness": self.episode_return,
                "best_fitness": (
                    None if self.best_fitness == -math.inf else self.best_fitness
                ),
                "mean_fitness": (
                    sum(self._episode_fitness) / len(self._episode_fitness)
                    if self._episode_fitness
                    else self.episode_return
                ),
                "population": population,
                "generation_history": list(self.generation_history),
                "event": self._last_event,
                **learning,
            }

        if self.agent is None:
            # Auto-evolution normally installs generation zero's successor in
            # the same step. This fallback protects custom non-auto trainers.
            self.agent = self._population_trainer.champion_agent()
        agent_learning = (
            dict(learning)
            if self.is_population and isinstance(learning, Mapping)
            else self.agent.telemetry(observation)
        )
        # The visualizer receives the exact network. It may down-sample nodes and
        # edges for legibility, but never invents values.
        network = self.agent.network_snapshot(observation)
        replay = agent_learning.get("replay", {})
        memory_samples = [
            {
                "action": item.action,
                "reward": item.reward,
                "done": item.done,
            }
            for item in list(self.agent.replay)[-12:]
        ]
        safety_value = snapshot.get("safety_prior")
        safety = (
            dict(safety_value)
            if isinstance(safety_value, Mapping)
            else self._safety_stats.snapshot()
        )
        proposed_action = safety.get(
            "proposed_action",
            agent_learning.get("last_action"),
        )
        executed_action = safety.get("executed_action", proposed_action)
        epsilon_schedule = self._epsilon_schedule_telemetry()
        environment_value = snapshot.get("environment")
        environment_snapshot = (
            dict(environment_value)
            if isinstance(environment_value, Mapping)
            else self.env.telemetry()
        )
        snapshot.update(
            {
                "mode": "population" if self.is_population else "episode",
                "algorithm": self.config.algorithm,
                "observation": list(observation),
                "observation_labels": list(DrivingEnv.OBSERVATION_LABELS),
                "action_labels": list(ACTION_LABELS),
                "q_values": agent_learning["q_values"],
                # The network view highlights its raw proposal; last_action is
                # the command that actually produced the visible transition.
                "selected_action": proposed_action,
                "last_action": executed_action,
                "proposed_action": proposed_action,
                "executed_action": executed_action,
                "safety_intervened": bool(safety.get("intervened", False)),
                "safety_prior": safety,
                "epsilon": (
                    0.0
                    if self.config.algorithm == "genetic"
                    else agent_learning["epsilon"]
                ),
                "epsilon_start": epsilon_schedule["start"],
                "epsilon_end": epsilon_schedule["end"],
                "epsilon_decay_steps": epsilon_schedule["decay_steps"],
                "expected_exploration_fraction": epsilon_schedule[
                    "expected_exploration_fraction"
                ],
                "expected_greedy_fraction": epsilon_schedule[
                    "expected_greedy_fraction"
                ],
                "epsilon_schedule": epsilon_schedule,
                "loss": agent_learning["last_loss"],
                "td_error": agent_learning["mean_absolute_td_error"],
                "gradient_steps": agent_learning["gradient_steps"],
                "target_syncs": agent_learning["target_syncs"],
                "action_counts": agent_learning["action_counts"],
                "batch_size": self.agent.config.batch_size,
                "warmup_steps": self.agent.config.warmup_steps,
                "train_interval": self.agent.config.train_interval,
                "environment_decisions": self.environment_decisions,
                "replay_size": replay.get("size", 0),
                "replay_capacity": replay.get("capacity", 0),
                "replay": replay,
                "memory_samples": memory_samples,
                "loss_history": list(self._loss_history),
                "epsilon_history": list(self._epsilon_history),
                "fitness": snapshot.get("current_fitness", 0.0),
                "network": network,
                "environment": environment_snapshot,
            }
        )
        if self.is_population:
            raw_health = snapshot.get("health")
            health = dict(raw_health) if isinstance(raw_health, Mapping) else {}
        else:
            session_learning = dict(agent_learning)
            session_learning["nonfinite_update_rejections"] = max(
                0,
                self.agent.nonfinite_update_rejections
                - self._health_nonfinite_origin,
            )
            health = build_learning_health(
                learning=session_learning,
                replay=replay,
                safety=safety,
                environment=snapshot["environment"],
                throughput={"workers": 1},
                environment_decisions=max(
                    0, self.environment_decisions - self._health_decision_origin
                ),
                batch_size=self.agent.config.batch_size,
                warmup_steps=self.agent.config.warmup_steps,
                gradient_clip=self.agent.config.gradient_clip,
                optimization_updates=max(
                    0, self.agent.gradient_steps - self._health_update_origin
                ),
                gradient_clip_events=max(
                    0,
                    self.agent.gradient_clip_events - self._health_clip_origin,
                ),
                wall_contact_decisions=self._wall_contact_decisions,
                collision_loop_terminations=self._collision_loop_terminations,
            )
            agent_health = agent_learning.get("health")
            if isinstance(agent_health, Mapping) and not bool(
                agent_health.get("finite", True)
            ):
                health["finite"] = False
                health["status"] = "critical"
                health["alerts"] = list(
                    dict.fromkeys(
                        [
                            *health["alerts"],
                            *(str(value) for value in agent_health.get("alerts", ())),
                        ]
                    )
                )
        snapshot["health"] = health
        snapshot["learning_status"] = health.get("status", "critical")
        return snapshot

    def save(self, path: str | Path) -> Path:
        """Save the current learner; population trainers may include ancestry."""

        output = Path(path).expanduser().resolve()
        if self.is_population and hasattr(self._population_trainer, "save"):
            saved = self._population_trainer.save(output)
        else:
            # Keep the file compatible with ``DrivingDQNAgent.load`` by adding
            # session metadata to the ordinary agent payload.  The agent
            # intentionally ignores unknown top-level keys.
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = self.agent.state_dict()
            payload["environment_curriculum"] = self.env.curriculum_state()
            # Resuming must continue the deterministic spawn stream. Saving
            # only the unlock latch would make every process restart replay
            # the same first 80/20 draw and random origin.
            payload["environment_rng_state"] = self.env.random.getstate()
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                torch.save(payload, temporary)
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            saved = output
        self._checkpoint_path = Path(saved)
        self._last_event = "checkpoint_saved"
        return Path(saved)

    def load(self, path: str | Path) -> None:
        checkpoint = Path(path).expanduser().resolve()
        if self.is_population:
            self._population_trainer.load(checkpoint)
            self.env = self._population_trainer.env
            self.agent = self._population_trainer.current_agent
            self.observation = self._population_trainer.observation
        else:
            state = self.agent.read_checkpoint(checkpoint)
            curriculum_state = state.get("environment_curriculum", {})
            if not isinstance(curriculum_state, Mapping):
                raise ValueError("checkpoint environment_curriculum must be a mapping")
            curriculum_value = curriculum_state.get(
                "unlocked", curriculum_state.get("ready", False)
            )
            if not isinstance(curriculum_value, bool):
                raise ValueError(
                    "checkpoint curriculum unlocked state must be a boolean"
                )
            rng_state = state.get("environment_rng_state")
            if rng_state is not None:
                import random

                rng_probe = random.Random()
                try:
                    rng_probe.setstate(rng_state)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "checkpoint environment RNG state is malformed"
                    ) from error
            self.agent.load_state_dict(state)
            self._environment_decisions = int(self.agent.environment_steps)
            self._health_decision_origin = self._environment_decisions
            self._health_update_origin = int(self.agent.gradient_steps)
            self._health_clip_origin = int(self.agent.gradient_clip_events)
            self._health_nonfinite_origin = int(
                self.agent.nonfinite_update_rejections
            )
            self.env.load_curriculum_state(curriculum_state)
            if rng_state is None:
                # Backward-compatible checkpoints predate the environment RNG
                # payload, so retain their former deterministic seed behavior.
                self.observation = self.env.reset(seed=self.config.seed)
            else:
                self.env.random.setstate(rng_state)
                # The constructor created one stale episode before the state
                # was read. Consume the saved stream once to begin the exact
                # next episode an uninterrupted session would have seen.
                self.observation = self.env.reset()
            self.episode_return = 0.0
            # A standalone agent checkpoint has no episode scoreboard. Until a
            # new complete evaluation is available, its loaded policy is the
            # honest best-available opponent for the P race.
            self._champion = self.agent.clone(seed=self.config.seed + 1)
        # These session-window diagnostics reset only after the checkpoint and
        # its nested state have loaded successfully. A rejected load therefore
        # leaves both the learner and its observability window unchanged.
        self._safety_stats = SensorClearanceStats()
        self._wall_contact_decisions = 0
        self._collision_loop_terminations = 0
        self._checkpoint_path = checkpoint
        self._last_event = "checkpoint_loaded"

    def reset_current_evaluation(self) -> tuple[float, ...]:
        """Restart the visible evaluation without modifying learned weights."""

        if self.is_population:
            observation = self._population_trainer.reset()
            self.env = self._population_trainer.env
            self.agent = self._population_trainer.current_agent
        else:
            observation = self.env.reset()
            self.episode_return = 0.0
        self.observation = tuple(float(value) for value in observation)
        self._last_event = "evaluation_reset"
        return self.observation

    def close(self) -> None:
        """Release persistent population workers; standalone DQN owns none."""

        if self._population_trainer is not None:
            self._population_trainer.close()


class ChampionRace:
    """A one-lap, fixed-60-Hz race against a frozen champion clone.

    The two environments and the policy are private to the race. Consequently,
    entering, leaving, or replaying a race cannot alter training replay memory,
    generation fitness, optimizer state, or population selection.
    """

    def __init__(self, session: DrivingLearningSession):
        self.circuit = session.env.circuit.slug
        self.build = session.build
        self.seed = session.config.seed + 50_000 + session.completed_generations
        self.agent = session.champion_agent()
        self.human_env = DrivingEnv(self.circuit, build=self.build, seed=self.seed)
        self.champion_env = DrivingEnv(self.circuit, build=self.build, seed=self.seed)
        self.champion_observation = self.champion_env.observation()
        # The race reproduces the evaluated learning policy, including its
        # deterministic safety prior. Human controls remain completely direct.
        self.clearance_policy = SensorClearancePolicy()
        self._champion_safety_stats = SensorClearanceStats()
        self.steps = 0
        self.winner: str | None = None
        self.human_finish_time: float | None = None
        self.champion_finish_time: float | None = None
        self.last_champion_action = int(DrivingAction.COAST)
        self.last_champion_proposed_action = int(DrivingAction.COAST)

    @property
    def elapsed(self) -> float:
        return self.steps * self.human_env.fixed_dt

    @property
    def finished(self) -> bool:
        return self.winner is not None

    def step(self, human_controls: DriverControls) -> tuple[StepResult, StepResult]:
        if self.finished:
            raise RuntimeError("race is finished; create a rematch before stepping")
        human_result = self.human_env.step_controls(human_controls)
        q_values = self.agent.q_values(self.champion_observation)
        self.last_champion_proposed_action = int(np.argmax(q_values))
        safety_decision = self.clearance_policy.decide(
            self.champion_observation,
            self.last_champion_proposed_action,
        )
        self._champion_safety_stats.observe(safety_decision)
        self.last_champion_action = safety_decision.executed_action
        champion_result = self.champion_env.step(self.last_champion_action)
        self.champion_observation = champion_result.observation
        self.steps += 1

        if self.human_env.laps and self.human_finish_time is None:
            self.human_finish_time = float(self.human_env.last_lap_time or self.elapsed)
        if self.champion_env.laps and self.champion_finish_time is None:
            self.champion_finish_time = float(
                self.champion_env.last_lap_time or self.elapsed
            )
        if self.human_finish_time is not None or self.champion_finish_time is not None:
            if self.human_finish_time is None:
                self.winner = "champion"
            elif self.champion_finish_time is None:
                self.winner = "human"
            elif abs(self.human_finish_time - self.champion_finish_time) <= 1e-9:
                self.winner = "tie"
            elif self.human_finish_time < self.champion_finish_time:
                self.winner = "human"
            else:
                self.winner = "champion"
        elif human_result.truncated or champion_result.truncated:
            if human_result.truncated and not champion_result.truncated:
                self.winner = "champion"
            elif champion_result.truncated and not human_result.truncated:
                self.winner = "human"
            else:
                human_progress = self.human_env.laps + float(
                    human_result.info.get("progress", 0.0)
                )
                champion_progress = self.champion_env.laps + float(
                    champion_result.info.get("progress", 0.0)
                )
                if abs(human_progress - champion_progress) <= 1e-9:
                    self.winner = "tie"
                elif human_progress > champion_progress:
                    self.winner = "human"
                else:
                    self.winner = "champion"
        return human_result, champion_result

    def telemetry(self) -> dict[str, Any]:
        human = self.human_env.telemetry()
        champion = self.champion_env.telemetry()
        safety = self._champion_safety_stats.snapshot()
        return {
            "mode": "race",
            "elapsed": self.elapsed,
            "winner": self.winner,
            "human_finish_time": self.human_finish_time,
            "champion_finish_time": self.champion_finish_time,
            "human_progress": human["laps"] + human["progress"],
            "champion_progress": champion["laps"] + champion["progress"],
            "human_speed": human["speed"],
            "champion_speed": champion["speed"],
            "champion_proposed_action": self.last_champion_proposed_action,
            "champion_proposed_action_label": ACTION_LABELS[
                self.last_champion_proposed_action
            ],
            "champion_action": self.last_champion_action,
            "champion_action_label": ACTION_LABELS[self.last_champion_action],
            "champion_safety_intervened": bool(safety["intervened"]),
            "champion_safety": safety,
        }
