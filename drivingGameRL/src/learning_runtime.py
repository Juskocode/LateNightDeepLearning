"""Training orchestration and a side-effect-free human/champion race.

The learning algorithms intentionally live below the Pygame presentation layer so
headless experiments, tests, and the interactive dashboard all execute the exact
same fixed-step simulation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .environment import DrivingAction, DrivingEnv, StepResult
from .ml import DQNConfig, DrivingDQNAgent
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

    def __post_init__(self) -> None:
        if self.algorithm not in ("dqn", "double_dqn", "genetic", "genetic_dqn"):
            raise ValueError(f"Unsupported driving learning algorithm: {self.algorithm}")
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
        for name in ("crossover_rate", "mutation_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("blend_alpha", "mutation_std"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


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
        self._loss_history: deque[float] = deque(maxlen=300)
        self._epsilon_history: deque[float] = deque(maxlen=300)

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
            dqn_config = DQNConfig(algorithm=base_algorithm, seed=self.config.seed)
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
        )
        self._population_trainer = PopulationTrainer(
            evolution,
            dqn_config=dqn_config,
            env=population_env,
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
            elif result.member_completed:
                self._last_event = "member_complete"
            else:
                self._last_event = "population_step"
            self._record_learning_trace(result)
            return result

        state = self.observation
        action = self.agent.select_action(state, explore=True)
        result = self.env.step(action)
        done = result.terminated or result.truncated
        self.agent.observe(state, action, result.reward, result.observation, done)
        self.episode_return += result.reward
        self.observation = result.observation
        self._last_event = "training_step"
        if done:
            self._finish_dqn_episode()
        self._record_learning_trace()
        return result

    def _record_learning_trace(self, population_step: Any | None = None) -> None:
        if self.agent is None:
            return
        loss = float(self.agent.last_loss)
        if population_step is not None and population_step.result is not None:
            loss = float(population_step.result.mean_loss)
        if math.isfinite(loss):
            self._loss_history.append(loss)
        epsilon = (
            0.0
            if self.config.algorithm == "genetic"
            else float(self.agent.epsilon)
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
            for index, member in enumerate(raw.get("population", ())):
                result = member.get("result") or {}
                member_fitness = member.get("fitness")
                population.append(
                    {
                        "index": index,
                        "member_id": member.get("member_id", index),
                        "fitness": member_fitness,
                        "status": (
                            "evaluating"
                            if index == current_index
                            else "evaluated"
                            if member.get("evaluated")
                            else "queued"
                        ),
                        "elite": False,
                        "laps": result.get("laps", 0),
                        "collisions": result.get("collisions", 0),
                        "parents": member.get("parent_ids", ()),
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
        agent_learning = self.agent.telemetry(observation)
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
        snapshot.update(
            {
                "mode": "population" if self.is_population else "episode",
                "algorithm": self.config.algorithm,
                "observation": list(observation),
                "observation_labels": list(DrivingEnv.OBSERVATION_LABELS),
                "action_labels": list(ACTION_LABELS),
                "q_values": agent_learning["q_values"],
                "selected_action": agent_learning.get("last_action"),
                "last_action": agent_learning.get("last_action"),
                "epsilon": (
                    0.0
                    if self.config.algorithm == "genetic"
                    else agent_learning["epsilon"]
                ),
                "loss": agent_learning["last_loss"],
                "td_error": agent_learning["mean_absolute_td_error"],
                "gradient_steps": agent_learning["gradient_steps"],
                "target_syncs": agent_learning["target_syncs"],
                "action_counts": agent_learning["action_counts"],
                "batch_size": self.agent.config.batch_size,
                "replay_size": replay.get("size", 0),
                "replay_capacity": replay.get("capacity", 0),
                "replay": replay,
                "memory_samples": memory_samples,
                "loss_history": list(self._loss_history),
                "epsilon_history": list(self._epsilon_history),
                "fitness": snapshot.get("current_fitness", 0.0),
                "network": network,
                "environment": self.env.telemetry(),
            }
        )
        return snapshot

    def save(self, path: str | Path) -> Path:
        """Save the current learner; population trainers may include ancestry."""

        output = Path(path).expanduser().resolve()
        if self.is_population and hasattr(self._population_trainer, "save"):
            saved = self._population_trainer.save(output)
        else:
            saved = self.agent.save(output)
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
            self.agent.load(checkpoint)
            # A standalone agent checkpoint has no episode scoreboard. Until a
            # new complete evaluation is available, its loaded policy is the
            # honest best-available opponent for the P race.
            self._champion = self.agent.clone(seed=self.config.seed + 1)
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
        self.steps = 0
        self.winner: str | None = None
        self.human_finish_time: float | None = None
        self.champion_finish_time: float | None = None
        self.last_champion_action = int(DrivingAction.COAST)

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
        self.last_champion_action = int(np.argmax(q_values))
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
        elif human_result.truncated and champion_result.truncated:
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
            "champion_action": self.last_champion_action,
            "champion_action_label": ACTION_LABELS[self.last_champion_action],
        }
