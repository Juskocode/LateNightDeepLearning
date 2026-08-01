"""Interchangeable learning algorithms for the Snake environment.

The small backend interface deliberately keeps policy selection separate from
the game.  Neural algorithms and educational tabular algorithms can therefore
share the same observation, replay buffer, telemetry, and evaluation harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from snakeGameQDlearning.src.config.settings import (
    GAMMA,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LEARNING_RATE,
    MODEL_DIR,
    OUTPUT_SIZE,
)

from .models import DuelingQNet, LinearQNet
from .replay import Experience
from .trainer import QTrainer


@dataclass(frozen=True)
class AlgorithmInfo:
    name: str
    family: str
    description: str


ALGORITHM_REGISTRY = {
    "dqn": AlgorithmInfo(
        "dqn", "deep", "DQN with a target network and max-target bootstrapping"
    ),
    "double_dqn": AlgorithmInfo(
        "double_dqn", "deep", "Double DQN with separate action selection and evaluation"
    ),
    "dueling_dqn": AlgorithmInfo(
        "dueling_dqn", "deep", "Dueling DQN with value and advantage streams"
    ),
    "dueling_double_dqn": AlgorithmInfo(
        "dueling_double_dqn",
        "deep",
        "Dueling Double DQN with value and advantage streams",
    ),
    "q_learning": AlgorithmInfo(
        "q_learning", "tabular", "Educational off-policy tabular Q-learning"
    ),
    "sarsa": AlgorithmInfo("sarsa", "tabular", "Educational on-policy Expected SARSA"),
}

ALGORITHM_ALIASES = {
    "tabular_q": "q_learning",
    "expected_sarsa": "sarsa",
    "dueling": "dueling_dqn",
    "dueling_double": "dueling_double_dqn",
}


def normalize_algorithm_name(name: str) -> str:
    normalized = ALGORITHM_ALIASES.get(name.lower(), name.lower())
    if normalized not in ALGORITHM_REGISTRY:
        choices = ", ".join(ALGORITHM_REGISTRY)
        raise ValueError(f"unknown algorithm {name!r}; choose one of: {choices}")
    return normalized


class LearningAlgorithm(ABC):
    """Minimal policy/learning/checkpoint interface used by :class:`Agent`."""

    def __init__(self, name: str):
        self.name = normalize_algorithm_name(name)
        self.info = ALGORITHM_REGISTRY[self.name]
        self.last_loss = 0.0
        self.last_target_mean = 0.0
        self.last_predicted_mean = 0.0
        self.last_gradient_norm = 0.0

    @abstractmethod
    def predict(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def target_predict(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, experiences: Sequence[Experience], epsilon: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def train_transition(self, experience: Experience, epsilon: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def save(self, filename: str, model_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, filename: str, model_dir: Path) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def structure_label(self) -> str:
        raise NotImplementedError

    @property
    def target_sync_progress(self) -> float:
        return 1.0

    @property
    def learned_states(self) -> int:
        return 0

    @property
    def supports_replay(self) -> bool:
        return True


class DeepQAlgorithm(LearningAlgorithm):
    def __init__(self, name: str):
        super().__init__(name)
        model_type = DuelingQNet if self.name.startswith("dueling_") else LinearQNet
        self.model = model_type(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        target_rule = {
            "dueling_dqn": "dqn",
            "dueling_double_dqn": "double_dqn",
        }.get(self.name, self.name)
        self.trainer = QTrainer(
            self.model,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            algorithm=target_rule,
        )

    @staticmethod
    def _predict(model, state: np.ndarray) -> np.ndarray:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            values = model(torch.as_tensor(state, dtype=torch.float32)).cpu().numpy()
        if was_training:
            model.train()
        return np.asarray(values, dtype=np.float32)

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self._predict(self.model, state)

    def target_predict(self, state: np.ndarray) -> np.ndarray:
        return self._predict(self.trainer.target_model, state)

    def train_step(self, experiences: Sequence[Experience], epsilon: float) -> float:
        if not experiences:
            return 0.0
        states, actions, rewards, next_states, dones = zip(*experiences)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        self._copy_metrics()
        return loss

    def train_transition(self, experience: Experience, epsilon: float) -> float:
        loss = self.trainer.train_step(*experience)
        self._copy_metrics()
        return loss

    def _copy_metrics(self) -> None:
        self.last_loss = self.trainer.last_loss
        self.last_target_mean = self.trainer.last_target_mean
        self.last_predicted_mean = self.trainer.last_predicted_mean
        self.last_gradient_norm = self.trainer.last_gradient_norm

    def save(self, filename: str, model_dir: Path) -> None:
        self.model.save(filename, str(model_dir))

    def load(self, filename: str, model_dir: Path) -> None:
        self.model.load(filename, str(model_dir))
        self.trainer.target_model.load_state_dict(self.model.state_dict())

    @property
    def structure_label(self) -> str:
        if self.name.startswith("dueling_"):
            return "DUELING  11 → 512 → 256 → VALUE + ADV → 3"
        return "ONLINE  11 → 512 → 256 → 3"

    @property
    def target_sync_progress(self) -> float:
        return (
            self.trainer.update_target_counter % self.trainer.target_update_freq
        ) / self.trainer.target_update_freq


class TabularAlgorithm(LearningAlgorithm):
    """Q table over the 2^11 binary observations (small enough to inspect)."""

    def __init__(self, name: str, *, learning_rate: float = 0.15, gamma: float = GAMMA):
        super().__init__(name)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.table: dict[int, np.ndarray] = {}
        # Compatibility attributes used by existing educational notebooks.
        self.model = self
        self.trainer = self
        self.target_model = self
        self.update_target_counter = 0
        self.target_update_freq = 1

    @staticmethod
    def encode_state(state: np.ndarray | Sequence[float]) -> int:
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        if values.shape != (INPUT_SIZE,):
            raise ValueError(f"state must contain {INPUT_SIZE} features")
        bits = values > 0.5
        return int(sum(int(bit) << index for index, bit in enumerate(bits)))

    def _row(self, state: np.ndarray | Sequence[float]) -> np.ndarray:
        key = self.encode_state(state)
        return self.table.setdefault(key, np.zeros(OUTPUT_SIZE, dtype=np.float32))

    def predict(self, state: np.ndarray) -> np.ndarray:
        key = self.encode_state(state)
        values = self.table.get(key)
        if values is None:
            return np.zeros(OUTPUT_SIZE, dtype=np.float32)
        return values.copy()

    def target_predict(self, state: np.ndarray) -> np.ndarray:
        return self.predict(state)

    def _bootstrap(self, next_values: np.ndarray, epsilon: float) -> float:
        if self.name == "q_learning":
            return float(np.max(next_values))
        probabilities = np.full(OUTPUT_SIZE, epsilon / OUTPUT_SIZE, dtype=np.float32)
        probabilities[int(np.argmax(next_values))] += 1.0 - epsilon
        return float(np.dot(probabilities, next_values))

    def train_step(self, experiences: Sequence[Experience], epsilon: float) -> float:
        if not experiences:
            return 0.0
        errors = []
        targets = []
        predictions = []
        for experience in experiences:
            action_values = np.asarray(experience.action)
            if (
                action_values.shape != (OUTPUT_SIZE,)
                or not np.isin(action_values, (0, 1)).all()
                or int(action_values.sum()) != 1
            ):
                raise ValueError("actions must be one-hot encoded")
            row = self._row(experience.state)
            action_index = int(action_values.argmax())
            prediction = float(row[action_index])
            bootstrap = (
                0.0
                if experience.done
                else self._bootstrap(self._row(experience.next_state), epsilon)
            )
            target = float(experience.reward) + self.gamma * bootstrap
            error = target - prediction
            row[action_index] += self.learning_rate * error
            errors.append(error)
            targets.append(target)
            predictions.append(prediction)
        self.last_loss = float(np.mean(np.square(errors)))
        self.last_target_mean = float(np.mean(targets))
        self.last_predicted_mean = float(np.mean(predictions))
        self.last_gradient_norm = 0.0
        return self.last_loss

    def train_transition(self, experience: Experience, epsilon: float) -> float:
        return self.train_step([experience], epsilon)

    def save(self, filename: str = "model.pth", model_dir: Path | None = None) -> None:
        directory = Path(model_dir) if model_dir is not None else MODEL_DIR
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": self.name,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "q_table": {
                key: torch.as_tensor(value, dtype=torch.float32)
                for key, value in self.table.items()
            },
        }
        torch.save(payload, directory / filename)
        print(f"Q table saved to {directory / filename}")

    def load(self, filename: str = "model.pth", model_dir: Path | None = None) -> None:
        directory = Path(model_dir) if model_dir is not None else MODEL_DIR
        path = directory / filename
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("algorithm") != self.name or "q_table" not in payload:
            raise ValueError(f"checkpoint is not compatible with {self.name}")
        self.table = {
            int(key): np.asarray(value, dtype=np.float32)
            for key, value in payload["q_table"].items()
        }
        print(f"Q table loaded from {path}")

    @property
    def structure_label(self) -> str:
        method = "EXPECTED SARSA" if self.name == "sarsa" else "Q LEARNING"
        return f"{method}  {len(self.table):,} / {2 ** INPUT_SIZE:,} STATES → 3"

    @property
    def learned_states(self) -> int:
        return len(self.table)

    @property
    def supports_replay(self) -> bool:
        # Expected SARSA is updated from the current behaviour policy only;
        # replaying transitions collected under old epsilon values would make
        # its on-policy interpretation misleading.
        return self.name != "sarsa"


def create_algorithm(name: str) -> LearningAlgorithm:
    """Create a learning backend from a stable CLI/configuration name."""

    normalized = normalize_algorithm_name(name)
    if ALGORITHM_REGISTRY[normalized].family == "deep":
        return DeepQAlgorithm(normalized)
    return TabularAlgorithm(normalized)


def available_algorithms() -> tuple[str, ...]:
    return tuple(ALGORITHM_REGISTRY)
