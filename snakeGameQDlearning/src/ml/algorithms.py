"""Interchangeable learning algorithms for the Snake environment.

The small backend interface deliberately keeps policy selection separate from
the game.  Neural algorithms and educational tabular algorithms can therefore
share the same observation, replay buffer, telemetry, and evaluation harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
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
from .replay import Experience, validated_experience
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
        self.last_q_abs_max = 0.0
        self.last_td_error_abs_mean = 0.0
        self.last_td_error_abs_max = 0.0
        self.update_count = 0
        self.attempted_update_count = 0
        self.rejected_update_count = 0
        self.last_batch_size = 0
        self.last_rejection: str | None = None

    @staticmethod
    def _validate_epsilon(epsilon: float) -> float:
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, Real)
            or not math.isfinite(float(epsilon))
            or not 0.0 <= epsilon <= 1.0
        ):
            raise ValueError("epsilon must be a finite number between 0 and 1")
        return float(epsilon)

    def health_metrics(self, decisions: int) -> dict:
        ratio = self.update_count / decisions if decisions > 0 else 0.0
        numeric = (
            self.last_loss,
            self.last_target_mean,
            self.last_predicted_mean,
            self.last_gradient_norm,
            self.last_q_abs_max,
            self.last_td_error_abs_mean,
            self.last_td_error_abs_max,
            ratio,
        )
        return {
            "finite": all(math.isfinite(float(value)) for value in numeric),
            "updates": self.update_count,
            "attempted_updates": self.attempted_update_count,
            "rejected_updates": self.rejected_update_count,
            "update_to_decision_ratio": ratio,
            "gradient_applicable": False,
            "gradient_norm": None,
            "clip_threshold": None,
            "clip_count": None,
            "clip_ratio": None,
            "last_batch_size": self.last_batch_size,
            "q_abs_max": self.last_q_abs_max,
            "td_error_abs_mean": self.last_td_error_abs_mean,
            "td_error_abs_max": self.last_td_error_abs_max,
            "last_rejection": self.last_rejection,
        }

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
        values_array = np.asarray(state, dtype=np.float32)
        if values_array.shape != (INPUT_SIZE,) or not np.isfinite(values_array).all():
            raise ValueError(f"state must be a finite {INPUT_SIZE}-feature vector")
        was_training = model.training
        model.eval()
        with torch.no_grad():
            values = model(torch.as_tensor(values_array, dtype=torch.float32)).cpu().numpy()
        if was_training:
            model.train()
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (OUTPUT_SIZE,) or not np.isfinite(result).all():
            raise FloatingPointError("network produced invalid Q values")
        return result

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self._predict(self.model, state)

    def target_predict(self, state: np.ndarray) -> np.ndarray:
        return self._predict(self.trainer.target_model, state)

    def train_step(self, experiences: Sequence[Experience], epsilon: float) -> float:
        if not experiences:
            return 0.0
        self._validate_epsilon(epsilon)
        validated = [validated_experience(experience) for experience in experiences]
        states, actions, rewards, next_states, dones = zip(*validated)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        self._copy_metrics()
        return loss

    def train_transition(self, experience: Experience, epsilon: float) -> float:
        self._validate_epsilon(epsilon)
        loss = self.trainer.train_step(*validated_experience(experience))
        self._copy_metrics()
        return loss

    def _copy_metrics(self) -> None:
        self.last_loss = self.trainer.last_loss
        self.last_target_mean = self.trainer.last_target_mean
        self.last_predicted_mean = self.trainer.last_predicted_mean
        self.last_gradient_norm = self.trainer.last_gradient_norm
        self.last_q_abs_max = self.trainer.last_q_abs_max
        self.last_td_error_abs_mean = self.trainer.last_td_error_abs_mean
        self.last_td_error_abs_max = self.trainer.last_td_error_abs_max
        self.update_count = self.trainer.update_target_counter
        self.attempted_update_count = self.trainer.attempted_updates
        self.rejected_update_count = self.trainer.rejected_updates
        self.last_batch_size = self.trainer.last_batch_size
        self.last_rejection = self.trainer.last_rejection

    def health_metrics(self, decisions: int) -> dict:
        metrics = self.trainer.health_metrics(decisions)
        metrics["gradient_applicable"] = True
        return metrics

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
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, Real)
            or not math.isfinite(float(learning_rate))
            or not 0.0 < learning_rate <= 1.0
        ):
            raise ValueError("learning_rate must be finite and in (0, 1]")
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, Real)
            or not math.isfinite(float(gamma))
            or not 0.0 <= gamma <= 1.0
        ):
            raise ValueError("gamma must be finite and between 0 and 1")
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
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
        if not np.isfinite(values).all():
            raise ValueError("state must contain only finite features")
        if not np.isin(values, (0.0, 1.0)).all():
            raise ValueError("tabular state features must be binary")
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
        epsilon = self._validate_epsilon(epsilon)
        self.attempted_update_count += 1
        try:
            validated = [validated_experience(experience) for experience in experiences]
        except (TypeError, ValueError) as error:
            self.rejected_update_count += 1
            self.last_rejection = str(error)
            raise
        table_before = {key: values.copy() for key, values in self.table.items()}
        errors = []
        targets = []
        predictions = []
        try:
            for experience in validated:
                action_values = np.asarray(experience.action)
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
                updated = prediction + self.learning_rate * error
                if not all(math.isfinite(value) for value in (prediction, target, error, updated)):
                    raise FloatingPointError("tabular update produced non-finite values")
                with np.errstate(over="ignore", invalid="ignore"):
                    stored_update = np.float32(updated)
                if not np.isfinite(stored_update):
                    raise FloatingPointError(
                        "tabular update is outside the learner's float32 range"
                    )
                row[action_index] = stored_update
                errors.append(error)
                targets.append(target)
                predictions.append(prediction)
        except (ValueError, FloatingPointError) as error:
            self.table = table_before
            self.rejected_update_count += 1
            self.last_rejection = str(error)
            raise
        derived = {
            "loss": float(np.mean(np.square(np.asarray(errors, dtype=np.float64)))),
            "target_mean": float(np.mean(targets)),
            "predicted_mean": float(np.mean(predictions)),
            "td_error_abs_mean": float(np.mean(np.abs(errors))),
            "td_error_abs_max": float(np.max(np.abs(errors))),
        }
        if not all(math.isfinite(value) for value in derived.values()):
            self.table = table_before
            self.rejected_update_count += 1
            self.last_rejection = "tabular diagnostics became non-finite"
            raise FloatingPointError(self.last_rejection)
        self.last_loss = derived["loss"]
        self.last_target_mean = derived["target_mean"]
        self.last_predicted_mean = derived["predicted_mean"]
        self.last_gradient_norm = 0.0
        self.last_q_abs_max = max(
            (float(np.max(np.abs(row))) for row in self.table.values()), default=0.0
        )
        self.last_td_error_abs_mean = derived["td_error_abs_mean"]
        self.last_td_error_abs_max = derived["td_error_abs_max"]
        self.update_count += 1
        self.last_batch_size = len(validated)
        self.last_rejection = None
        return self.last_loss

    def train_transition(self, experience: Experience, epsilon: float) -> float:
        return self.train_step([experience], epsilon)

    def save(self, filename: str = "model.pth", model_dir: Path | None = None) -> None:
        directory = Path(model_dir) if model_dir is not None else MODEL_DIR
        directory.mkdir(parents=True, exist_ok=True)
        if not all(
            isinstance(key, int)
            and not isinstance(key, bool)
            and 0 <= key < 2 ** INPUT_SIZE
            and np.asarray(value).shape == (OUTPUT_SIZE,)
            and np.isfinite(value).all()
            for key, value in self.table.items()
        ):
            raise ValueError("Q table contains malformed or non-finite rows")
        payload = {
            "algorithm": self.name,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "q_table": {
                key: torch.as_tensor(value, dtype=torch.float32)
                for key, value in self.table.items()
            },
        }
        destination = directory / filename
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(directory)
        )
        os.close(descriptor)
        try:
            torch.save(payload, temporary_name)
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        print(f"Q table saved to {destination}")

    def load(self, filename: str = "model.pth", model_dir: Path | None = None) -> None:
        directory = Path(model_dir) if model_dir is not None else MODEL_DIR
        path = directory / filename
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("algorithm") != self.name \
                or not isinstance(payload.get("q_table"), dict):
            raise ValueError(f"checkpoint is not compatible with {self.name}")
        learning_rate = payload.get("learning_rate", self.learning_rate)
        gamma = payload.get("gamma", self.gamma)
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, Real)
            or not math.isfinite(float(learning_rate))
            or not 0.0 < learning_rate <= 1.0
        ):
            raise ValueError("checkpoint learning_rate is invalid")
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, Real)
            or not math.isfinite(float(gamma))
            or not 0.0 <= gamma <= 1.0
        ):
            raise ValueError("checkpoint gamma is invalid")
        restored: dict[int, np.ndarray] = {}
        for raw_key, raw_value in payload["q_table"].items():
            if isinstance(raw_key, bool) or not isinstance(raw_key, int) \
                    or not 0 <= raw_key < 2 ** INPUT_SIZE:
                raise ValueError("checkpoint Q-table key is invalid")
            value = np.asarray(raw_value, dtype=np.float32)
            if value.shape != (OUTPUT_SIZE,) or not np.isfinite(value).all():
                raise ValueError("checkpoint Q-table row is malformed or non-finite")
            restored[raw_key] = value.copy()
        # Commit all validated state together; rejected loads leave the live
        # policy and its hyperparameters unchanged.
        self.table = restored
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
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
