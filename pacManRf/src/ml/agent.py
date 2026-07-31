"""Pacman DQN agent: policy, replay, telemetry, and checkpoint lifecycle."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from pacManRf.src.game.constants import Direction

from .config import DEFAULT_OBSERVATION_SIZE, DQNConfig
from .models import PacmanQNetwork
from .observations import ObservationFrame, PacmanObservationEncoder
from .replay import Experience, ReplayBuffer
from .trainer import DQNTrainer, TrainingMetrics


CHECKPOINT_VERSION = 1


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location=map_location)


class PacmanDQNAgent:
    """A configurable epsilon-greedy agent for four-direction Pacman control."""

    def __init__(
        self,
        config: DQNConfig | None = None,
        *,
        encoder: PacmanObservationEncoder | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        if encoder is None and self.config.observation_size == DEFAULT_OBSERVATION_SIZE:
            encoder = PacmanObservationEncoder()
        if encoder is not None and encoder.size != self.config.observation_size:
            raise ValueError("encoder size does not match configured observation_size")
        self.encoder = encoder

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        self.model = PacmanQNetwork(
            self.config.observation_size,
            self.config.action_size,
            self.config.hidden_sizes,
        )
        self.trainer = DQNTrainer(self.model, self.config)
        self.memory = ReplayBuffer(
            self.config.replay_capacity,
            observation_size=self.config.observation_size,
            action_size=self.config.action_size,
            seed=self.config.seed,
        )

        self.env_steps = 0
        self.episodes = 0
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.last_action_index = 0
        self.last_policy_mode = "explore"
        self.last_q_values = np.zeros(self.config.action_size, dtype=np.float32)
        self.last_target_q_values = np.zeros(self.config.action_size, dtype=np.float32)
        self.last_legal_action_mask = np.ones(self.config.action_size, dtype=np.bool_)
        self.last_observation_frame: ObservationFrame | None = None
        self.checkpoint_metadata: dict[str, Any] = {}

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.env_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    @property
    def n_games(self) -> int:
        """Compatibility name used by the Snake trainer and older examples."""

        return self.episodes

    @n_games.setter
    def n_games(self, value: int) -> None:
        self.episodes = int(value)

    def get_state(self, game: Any) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("get_state requires an encoder matching the configured observation size")
        self.last_observation_frame = self.encoder.observe(game)
        return self.last_observation_frame.values.copy()

    def _state(self, state: Any) -> np.ndarray:
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        if values.shape != (self.config.observation_size,):
            raise ValueError(f"state must have shape ({self.config.observation_size},)")
        if not np.isfinite(values).all():
            raise ValueError("state must contain finite values")
        return values

    def _action_index(self, action: int | Sequence[int]) -> int:
        values = np.asarray(action)
        if values.ndim == 0:
            index = int(values)
        elif values.shape == (self.config.action_size,):
            if not np.all((values == 0) | (values == 1)) or values.sum() != 1:
                raise ValueError("one-hot action must contain exactly one selected action")
            index = int(values.argmax())
        else:
            raise ValueError("action must be an index or one-hot vector")
        if not 0 <= index < self.config.action_size:
            raise ValueError("action is outside the configured action space")
        return index

    def _legal_mask(self, legal_action_mask: Sequence[bool] | None) -> np.ndarray:
        if legal_action_mask is None:
            return np.ones(self.config.action_size, dtype=np.bool_)
        mask = np.asarray(legal_action_mask, dtype=np.bool_)
        if mask.shape != (self.config.action_size,):
            raise ValueError(f"legal_action_mask must have shape ({self.config.action_size},)")
        if not mask.any():
            raise ValueError("at least one action must be legal")
        return mask

    def select_action(
        self,
        state: Any,
        *,
        explore: bool = True,
        legal_action_mask: Sequence[bool] | None = None,
        advance_schedule: bool = True,
    ) -> int:
        values = self._state(state)
        mask = self._legal_mask(legal_action_mask)
        self.last_q_values = np.asarray(self.trainer.predict(values), dtype=np.float32)
        self.last_target_q_values = np.asarray(self.trainer.predict(values, target=True), dtype=np.float32)
        current_epsilon = self.epsilon if explore else 0.0
        legal_indices = np.flatnonzero(mask)
        if explore and self.rng.random() < current_epsilon:
            action = int(self.rng.choice(legal_indices))
            self.last_policy_mode = "explore"
        else:
            legal_values = np.where(mask, self.last_q_values, -np.inf)
            action = int(np.argmax(legal_values))
            self.last_policy_mode = "exploit"
        self.last_action_index = action
        self.last_legal_action_mask = mask.copy()
        if advance_schedule:
            self.env_steps += 1
        return action

    def get_action(
        self,
        state: Any,
        *,
        explore: bool = True,
        legal_action_mask: Sequence[bool] | None = None,
    ) -> list[int]:
        index = self.select_action(state, explore=explore, legal_action_mask=legal_action_mask)
        action = [0] * self.config.action_size
        action[index] = 1
        return action

    def direction_for_action(self, action: int | Sequence[int]) -> Direction:
        if self.config.action_size != 4:
            raise ValueError("Direction conversion requires the four-action Pacman space")
        return (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)[self._action_index(action)]

    def remember(self, state: Any, action: int | Sequence[int], reward: float, next_state: Any, done: bool) -> None:
        state_values = self._state(state)
        next_values = self._state(next_state)
        action_index = self._action_index(action)
        self.memory.append(Experience(state_values, action_index, float(reward), next_values, bool(done)))
        self.last_reward = float(reward)

    def train_short_memory(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> float:
        return self.trainer.train_step(
            self._state(state), self._action_index(action), reward, self._state(next_state), done
        )

    def train_long_memory(self) -> float:
        batch = self.memory.sample_batch(self.config.batch_size)
        if batch is None:
            return 0.0
        return self.trainer.train_step(*batch)

    def learn_if_ready(self) -> TrainingMetrics | None:
        if len(self.memory) < max(self.config.replay_warmup, self.config.batch_size):
            return None
        if self.env_steps % self.config.train_frequency:
            return None
        batch = self.memory.sample_batch(self.config.batch_size)
        if batch is None:
            return None
        return self.trainer.update(*batch)

    def observe(
        self,
        state: Any,
        action: int | Sequence[int],
        reward: float,
        next_state: Any,
        done: bool,
    ) -> TrainingMetrics | None:
        self.remember(state, action, reward, next_state, done)
        self.episode_return += float(reward)
        metrics = self.learn_if_ready()
        if done:
            self.episodes += 1
        return metrics

    def reset_episode(self) -> None:
        self.episode_return = 0.0

    def telemetry(self, state: Any | None = None, *, max_neurons_per_layer: int | None = 16) -> dict[str, Any]:
        if state is not None:
            values = self._state(state)
            self.last_q_values = np.asarray(self.trainer.predict(values), dtype=np.float32)
            self.last_target_q_values = np.asarray(self.trainer.predict(values, target=True), dtype=np.float32)
        elif self.last_observation_frame is not None:
            values = self.last_observation_frame.values
        else:
            values = np.zeros(self.config.observation_size, dtype=np.float32)
        network = self.model.network_snapshot(values, max_neurons_per_layer=max_neurons_per_layer)
        recent = self.memory.tail(32)
        metrics = self.trainer.last_metrics.to_dict()
        activations = {
            layer["name"]: {
                "indices": layer["selected_indices"],
                "values": layer["activations"],
                "stats": layer["stats"],
            }
            for layer in network["layers"]
        }
        weights = {
            f"{connection['from']}->{connection['to']}": {
                "source_indices": connection["source_indices"],
                "target_indices": connection["target_indices"],
                "values": connection["weights"],
                "biases": connection["biases"],
                "stats": connection["stats"],
            }
            for connection in network["connections"]
        }
        return {
            "algorithm": self.config.algorithm,
            "state": values.astype(float).tolist(),
            "observation_labels": list(self.encoder.labels) if self.encoder is not None else [],
            "vision": self.last_observation_frame.to_dict() if self.last_observation_frame else None,
            "q_values": self.last_q_values.astype(float).tolist(),
            "target_q_values": self.last_target_q_values.astype(float).tolist(),
            "action_index": self.last_action_index,
            "action_label": self.config.action_labels[self.last_action_index],
            "legal_action_mask": self.last_legal_action_mask.tolist(),
            "policy_mode": self.last_policy_mode,
            "epsilon": self.epsilon,
            "reward": self.last_reward,
            "episode_return": self.episode_return,
            "episodes": self.episodes,
            "env_steps": self.env_steps,
            "train_steps": self.trainer.train_steps,
            "loss": metrics["loss"],
            "gradient_norm": metrics["gradient_norm"],
            "target_mean": metrics["target_q_mean"],
            "predicted_mean": metrics["predicted_q_mean"],
            "bootstrap_mean": metrics["bootstrap_mean"],
            "td_error_mean": metrics["td_error_mean"],
            "td_error_abs_mean": metrics["td_error_abs_mean"],
            "target_sync_progress": self.trainer.target_sync_progress,
            "memory": len(self.memory),
            "memory_capacity": self.memory.capacity,
            "memory_stats": self.memory.stats(),
            "recent_rewards": [item.reward for item in recent],
            "recent_actions": [item.action for item in recent],
            "recent_dones": [item.done for item in recent],
            "network": network,
            "network_activations": activations,
            "network_weights": weights,
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        include_replay: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "config": self.config.to_dict(),
            "trainer": self.trainer.state_dict(),
            "agent": {
                "env_steps": self.env_steps,
                "episodes": self.episodes,
                "episode_return": self.episode_return,
                "last_reward": self.last_reward,
                "last_action_index": self.last_action_index,
                "last_policy_mode": self.last_policy_mode,
                "numpy_rng_state": self.rng.bit_generator.state,
                "torch_rng_state": torch.random.get_rng_state(),
            },
            "metadata": dict(metadata or {}),
        }
        if include_replay:
            payload["replay"] = self.memory.state_dict()

        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
                temporary = handle.name
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        self.checkpoint_metadata = dict(metadata or {})
        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
        load_replay: bool = True,
    ) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint not found: {source}")
        payload = _safe_torch_load(source, self.trainer.device)
        if int(payload.get("checkpoint_version", 0)) > CHECKPOINT_VERSION:
            raise ValueError("checkpoint was created by a newer learning-core version")
        saved_config = DQNConfig.from_dict(payload["config"])
        architecture = (saved_config.observation_size, saved_config.action_size, saved_config.hidden_sizes)
        expected = (self.config.observation_size, self.config.action_size, self.config.hidden_sizes)
        if architecture != expected:
            raise ValueError("checkpoint network architecture does not match this agent")
        if saved_config.algorithm != self.config.algorithm:
            raise ValueError("checkpoint algorithm does not match this agent")
        self.trainer.load_state_dict(payload["trainer"], load_optimizer=load_optimizer)
        agent_state = payload.get("agent", {})
        self.env_steps = int(agent_state.get("env_steps", 0))
        self.episodes = int(agent_state.get("episodes", 0))
        self.episode_return = float(agent_state.get("episode_return", 0.0))
        self.last_reward = float(agent_state.get("last_reward", 0.0))
        self.last_action_index = int(agent_state.get("last_action_index", 0))
        self.last_policy_mode = str(agent_state.get("last_policy_mode", "explore"))
        if "numpy_rng_state" in agent_state:
            self.rng.bit_generator.state = agent_state["numpy_rng_state"]
        if "torch_rng_state" in agent_state:
            torch.random.set_rng_state(agent_state["torch_rng_state"].cpu())
        if load_replay and "replay" in payload:
            self.memory.load_state_dict(payload["replay"])
        self.checkpoint_metadata = dict(payload.get("metadata", {}))
        return self.checkpoint_metadata.copy()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        load_optimizer: bool = True,
        load_replay: bool = True,
    ) -> "PacmanDQNAgent":
        source = Path(path).expanduser().resolve()
        payload = _safe_torch_load(source, "cpu")
        config = DQNConfig.from_dict(payload["config"])
        if device is not None:
            config = replace(config, device=device)
        agent = cls(config)
        agent.load_checkpoint(source, load_optimizer=load_optimizer, load_replay=load_replay)
        return agent

    save = save_checkpoint
    load = load_checkpoint


Agent = PacmanDQNAgent

