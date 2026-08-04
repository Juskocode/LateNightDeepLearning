"""Validated configuration for the Driving Lab Q-learning agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Mapping


Algorithm = Literal["dqn", "double_dqn"]

POPULATION_EPSILON_START = 0.30
POPULATION_EPSILON_END = 0.05
POPULATION_REPLAY_WARMUP_STEPS = 96
POPULATION_TRAIN_INTERVAL = 4


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Hyperparameters for a deterministic CPU DQN experiment.

    The defaults match :class:`drivingGameRL.src.environment.DrivingEnv`:
    sixteen normalized observations and five discrete driving actions.
    """

    observation_size: int = 16
    action_size: int = 5
    hidden_sizes: tuple[int, ...] = (128, 128)
    algorithm: Algorithm = "double_dqn"
    replay_capacity: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 512
    train_interval: int = 1
    target_sync_interval: int = 500
    gamma: float = 0.99
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40_000
    seed: int = 0

    def __post_init__(self) -> None:
        hidden_sizes = tuple(self.hidden_sizes)
        object.__setattr__(self, "hidden_sizes", hidden_sizes)

        self._positive_integer("observation_size", self.observation_size)
        self._positive_integer("action_size", self.action_size)
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        for index, size in enumerate(hidden_sizes):
            self._positive_integer(f"hidden_sizes[{index}]", size)
        if self.algorithm not in ("dqn", "double_dqn"):
            raise ValueError("algorithm must be 'dqn' or 'double_dqn'")
        self._positive_integer("replay_capacity", self.replay_capacity)
        self._positive_integer("batch_size", self.batch_size)
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity")
        self._non_negative_integer("warmup_steps", self.warmup_steps)
        self._positive_integer("train_interval", self.train_interval)
        self._positive_integer("target_sync_interval", self.target_sync_interval)
        self._positive_integer("epsilon_decay_steps", self.epsilon_decay_steps)

        self._finite_in_range("gamma", self.gamma, 0.0, 1.0)
        self._positive_finite("learning_rate", self.learning_rate)
        self._non_negative_finite("weight_decay", self.weight_decay)
        self._positive_finite("gradient_clip", self.gradient_clip)
        self._finite_in_range("epsilon_start", self.epsilon_start, 0.0, 1.0)
        self._finite_in_range("epsilon_end", self.epsilon_end, 0.0, 1.0)
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end cannot exceed epsilon_start")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in the [0, 2**63) interval")

    @property
    def input_size(self) -> int:
        """Alias used by generic network visualizers."""

        return self.observation_size

    @property
    def output_size(self) -> int:
        """Alias used by generic network visualizers."""

        return self.action_size

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-friendly representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DQNConfig":
        """Rebuild and revalidate a serialized configuration."""

        return cls(**dict(values))

    @staticmethod
    def _positive_integer(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _non_negative_integer(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    @staticmethod
    def _positive_finite(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite positive number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")

    @staticmethod
    def _non_negative_finite(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite non-negative number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")

    @staticmethod
    def _finite_in_range(name: str, value: object, low: float, high: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be in the [{low}, {high}] interval")
        numeric = float(value)
        if not math.isfinite(numeric) or not low <= numeric <= high:
            raise ValueError(f"{name} must be in the [{low}, {high}] interval")


def default_population_dqn_config(
    *,
    evaluation_steps: int,
    seed: int,
    algorithm: Algorithm = "double_dqn",
) -> DQNConfig:
    """Return defaults scaled to one bounded population-member lifetime."""

    if (
        isinstance(evaluation_steps, bool)
        or not isinstance(evaluation_steps, int)
        or evaluation_steps <= 0
    ):
        raise ValueError("evaluation_steps must be a positive integer")
    return DQNConfig(
        algorithm=algorithm,
        warmup_steps=POPULATION_REPLAY_WARMUP_STEPS,
        train_interval=POPULATION_TRAIN_INTERVAL,
        epsilon_start=POPULATION_EPSILON_START,
        epsilon_end=POPULATION_EPSILON_END,
        epsilon_decay_steps=evaluation_steps,
        seed=seed,
    )


__all__ = (
    "Algorithm",
    "DQNConfig",
    "POPULATION_EPSILON_END",
    "POPULATION_EPSILON_START",
    "POPULATION_REPLAY_WARMUP_STEPS",
    "POPULATION_TRAIN_INTERVAL",
    "default_population_dqn_config",
)
