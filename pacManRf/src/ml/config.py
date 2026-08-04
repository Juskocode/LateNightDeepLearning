"""Configuration primitives for the Pacman DQN stack.

The learning code intentionally keeps game dimensions in configuration instead
of baking a particular observation encoder into the neural network.  The
defaults match :class:`PacmanObservationEncoder`, while callers can provide a
different observation vector without replacing any of the training machinery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Literal, Mapping


Algorithm = Literal["dqn", "double_dqn"]

# The default encoder emits 20 ray features, 4 direction features, 20 ghost
# features, 6 target features, and 10 global/phase features.
DEFAULT_OBSERVATION_SIZE = 60
DEFAULT_ACTION_LABELS = ("UP", "DOWN", "LEFT", "RIGHT")


def normalize_algorithm(value: str) -> Algorithm:
    """Return a canonical algorithm name or raise a helpful error."""

    if not isinstance(value, str):
        raise ValueError("algorithm must be 'dqn' or 'double_dqn'")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"ddqn", "doubleq", "double_q", "double_dqn"}:
        return "double_dqn"
    if normalized == "dqn":
        return "dqn"
    raise ValueError("algorithm must be 'dqn' or 'double_dqn'")


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Validated, serializable hyperparameters for a Pacman agent."""

    observation_size: int = DEFAULT_OBSERVATION_SIZE
    action_size: int = 4
    hidden_sizes: tuple[int, ...] = (256, 128)
    action_labels: tuple[str, ...] = DEFAULT_ACTION_LABELS
    algorithm: Algorithm = "double_dqn"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 100_000
    replay_warmup: int = 1_000
    train_frequency: int = 1
    target_update_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    gradient_clip: float = 10.0
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "algorithm", normalize_algorithm(self.algorithm))
        hidden_sizes = tuple(self.hidden_sizes)
        if not hidden_sizes or any(
            isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0
            for size in hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integer layer sizes")
        object.__setattr__(self, "hidden_sizes", tuple(int(size) for size in hidden_sizes))
        object.__setattr__(self, "action_labels", tuple(str(label) for label in self.action_labels))

        positive_ints = {
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "train_frequency": self.train_frequency,
            "target_update_interval": self.target_update_interval,
            "epsilon_decay_steps": self.epsilon_decay_steps,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.replay_warmup, bool)
            or not isinstance(self.replay_warmup, Integral)
        ):
            raise ValueError("replay_warmup must be an integer")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, Integral)
        ):
            raise ValueError("seed must be an integer or None")
        if len(self.action_labels) != self.action_size:
            raise ValueError("action_labels length must equal action_size")
        if any(not label.strip() for label in self.action_labels):
            raise ValueError("action_labels cannot contain empty labels")
        if len(set(self.action_labels)) != len(self.action_labels):
            raise ValueError("action_labels must be unique")

        finite_numbers = {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "gradient_clip": self.gradient_clip,
            "weight_decay": self.weight_decay,
        }
        for name, value in finite_numbers.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.replay_warmup < 0 or self.replay_warmup > self.replay_capacity:
            raise ValueError("replay_warmup must be between 0 and replay_capacity")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-friendly representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DQNConfig":
        data = dict(values)
        if "hidden_sizes" in data:
            data["hidden_sizes"] = tuple(data["hidden_sizes"])
        if "action_labels" in data:
            data["action_labels"] = tuple(data["action_labels"])
        return cls(**data)
