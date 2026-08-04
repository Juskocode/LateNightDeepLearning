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
from .validation import (
    action_index,
    binary_flag,
    boolean_mask,
    finite_float,
    finite_vector,
    require_mapping,
    strict_int,
)


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
        self.transitions_observed = 0
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
        frame = self.encoder.observe(game)
        values = self._state(frame.values)
        self.last_observation_frame = frame
        return values.copy()

    def _state(self, state: Any) -> np.ndarray:
        return finite_vector(state, self.config.observation_size, name="state")

    def _action_index(self, action: int | Sequence[int]) -> int:
        values = np.asarray(action)
        if values.ndim == 0:
            index = action_index(values.item(), self.config.action_size)
        elif values.shape == (self.config.action_size,):
            mask = boolean_mask(values, self.config.action_size, name="one-hot action")
            if int(mask.sum()) != 1:
                raise ValueError("one-hot action must contain exactly one selected action")
            index = int(mask.argmax())
        else:
            raise ValueError("action must be an index or one-hot vector")
        return index

    def _legal_mask(self, legal_action_mask: Sequence[bool] | None) -> np.ndarray:
        if legal_action_mask is None:
            return np.ones(self.config.action_size, dtype=np.bool_)
        return boolean_mask(
            legal_action_mask,
            self.config.action_size,
            name="legal_action_mask",
        )

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

    def remember(
        self,
        state: Any,
        action: int | Sequence[int],
        reward: float,
        next_state: Any,
        done: bool,
        next_legal_action_mask: Sequence[bool] | None = None,
    ) -> None:
        state_values = self._state(state)
        next_values = self._state(next_state)
        action_index = self._action_index(action)
        reward_value = finite_float(reward, "reward")
        done_value = binary_flag(done, "done")
        mask = self._legal_mask(next_legal_action_mask)
        self.memory.append(
            Experience(state_values, action_index, reward_value, next_values, done_value, mask)
        )
        self.transitions_observed += 1
        self.last_reward = reward_value

    def train_short_memory(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        next_legal_action_mask: Sequence[bool] | None = None,
    ) -> float:
        return self.trainer.train_step(
            self._state(state),
            self._action_index(action),
            reward,
            self._state(next_state),
            done,
            self._legal_mask(next_legal_action_mask),
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
        next_legal_action_mask: Sequence[bool] | None = None,
    ) -> TrainingMetrics | None:
        reward_value = finite_float(reward, "reward")
        done_value = binary_flag(done, "done")
        next_return = self.episode_return + reward_value
        if not np.isfinite(next_return):
            raise FloatingPointError("episode return became non-finite; transition was rejected")
        self.remember(
            state,
            action,
            reward_value,
            next_state,
            done_value,
            next_legal_action_mask,
        )
        self.episode_return = next_return
        metrics = self.learn_if_ready()
        if done_value:
            self.episodes += 1
        return metrics

    def reset_episode(self) -> None:
        self.episode_return = 0.0

    @staticmethod
    def _absolute_max(values: np.ndarray) -> float | None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not array.size or not np.isfinite(array).all():
            return None
        return float(np.abs(array).max())

    def health_telemetry(self) -> dict[str, Any]:
        """Return a compact, JSON-safe health report for the live learner."""

        replay_size = len(self.memory)
        readiness_threshold = max(self.config.replay_warmup, self.config.batch_size)
        replay_ready = replay_size >= readiness_threshold
        updates = self.trainer.train_steps
        decisions = self.transitions_observed
        metrics = self.trainer.last_metrics
        trainer_finite, invalid_fields = self.trainer.finite_diagnostics()

        numeric_values = {
            "epsilon": self.epsilon,
            "last_reward": self.last_reward,
            "episode_return": self.episode_return,
            "loss": metrics.loss,
            "gradient_norm": metrics.gradient_norm,
            "td_error_abs_mean": metrics.td_error_abs_mean,
        }
        for name, value in numeric_values.items():
            if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
                invalid_fields.append(name)
        for name, values in (
            ("q_values", self.last_q_values),
            ("target_q_values", self.last_target_q_values),
        ):
            if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                invalid_fields.append(name)
        invalid_fields = sorted(set(invalid_fields))
        finite = trainer_finite and not invalid_fields

        q_abs_max = self._absolute_max(self.last_q_values)
        target_q_abs_max = self._absolute_max(self.last_target_q_values)
        clip_threshold = float(self.config.gradient_clip)
        gradient_to_clip_ratio = (
            metrics.gradient_norm / clip_threshold if clip_threshold > 0 else 0.0
        )
        clip_ratio = (
            self.trainer.gradient_clip_events / updates
            if updates > 0 and self.trainer.gradient_clip_history_complete
            else (0.0 if self.trainer.gradient_clip_history_complete else None)
        )
        update_ratio = updates / decisions if decisions > 0 else 0.0
        expected_updates = max(
            0,
            decisions // self.config.train_frequency
            - (readiness_threshold - 1) // self.config.train_frequency,
        )
        update_coverage = (
            updates / expected_updates if expected_updates > 0 else None
        )
        recent = self.memory.tail(64)
        recent_rewards = np.asarray([item.reward for item in recent], dtype=np.float64)
        if recent_rewards.size:
            reward_diagnostics = {
                "window": len(recent),
                "mean": float(recent_rewards.mean()),
                "std": float(recent_rewards.std()),
                "min": float(recent_rewards.min()),
                "max": float(recent_rewards.max()),
                "positive": int(np.count_nonzero(recent_rewards > 0)),
                "negative": int(np.count_nonzero(recent_rewards < 0)),
                "zero": int(np.count_nonzero(recent_rewards == 0)),
                "terminal_count": sum(int(item.done) for item in recent),
                "terminal_ratio": sum(int(item.done) for item in recent) / len(recent),
            }
        else:
            reward_diagnostics = {
                "window": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "positive": 0,
                "negative": 0,
                "zero": 0,
                "terminal_count": 0,
                "terminal_ratio": 0.0,
            }

        alerts: list[str] = []
        critical = False
        if not finite:
            alerts.append("non_finite_learning_state")
            critical = True
        if q_abs_max is not None and q_abs_max >= 1_000.0:
            alerts.append("q_value_magnitude_high")
            critical = critical or q_abs_max >= 1_000_000.0
        if metrics.td_error_abs_mean >= 1_000.0:
            alerts.append("td_error_magnitude_high")
            critical = critical or metrics.td_error_abs_mean >= 1_000_000.0
        if self.trainer.recent_gradient_clip_window >= 8 and self.trainer.recent_gradient_clip_fraction >= 0.5:
            alerts.append("gradient_clipping_frequent")
        if gradient_to_clip_ratio >= 5.0:
            alerts.append("gradient_norm_extreme")
        if (
            replay_ready
            and expected_updates >= 8
            and updates < expected_updates * 0.5
        ):
            alerts.append("optimizer_update_coverage_low")

        if critical:
            status = "critical"
        elif alerts:
            status = "warning"
        elif not replay_ready:
            status = "warming_up"
        else:
            status = "healthy"
        return {
            "status": status,
            "finite": finite,
            "alerts": alerts,
            "replay": {
                "applicable": True,
                "size": replay_size,
                "capacity": self.memory.capacity,
                "fill_ratio": replay_size / self.memory.capacity,
                "ready": replay_ready,
                "warmup_threshold": readiness_threshold,
                "samples_until_ready": max(0, readiness_threshold - replay_size),
            },
            "optimization": {
                "applicable": True,
                "updates": updates,
                "decisions": decisions,
                "update_to_decision_ratio": update_ratio,
                "expected_updates": expected_updates,
                "update_coverage": update_coverage,
                "expected_update_ratio_after_warmup": 1.0 / self.config.train_frequency,
                "gradient_norm": float(metrics.gradient_norm),
                "clip_threshold": clip_threshold,
                "clip_ratio": None if clip_ratio is None else float(clip_ratio),
                "clip_history_complete": self.trainer.gradient_clip_history_complete,
                "gradient_to_clip_ratio": float(gradient_to_clip_ratio),
                "clipped_last_update": bool(metrics.gradient_clipped),
                "clip_events": self.trainer.gradient_clip_events,
                "recent_clip_fraction": self.trainer.recent_gradient_clip_fraction,
                "recent_clip_window": self.trainer.recent_gradient_clip_window,
            },
            "values": {
                "q_applicable": True,
                "td_error_applicable": True,
                "q_abs_max": q_abs_max,
                "target_q_abs_max": target_q_abs_max,
                "td_error_abs_mean": float(metrics.td_error_abs_mean),
                "td_error_abs_max": float(metrics.td_error_abs_max),
            },
            "recent": reward_diagnostics,
            "numeric": {
                "checked": len(numeric_values) + 2,
                "non_finite_fields": invalid_fields,
            },
        }

    def telemetry(self, state: Any | None = None, *, max_neurons_per_layer: int | None = 16) -> dict[str, Any]:
        if state is not None:
            values = self._state(state)
            try:
                self.last_q_values = np.asarray(self.trainer.predict(values), dtype=np.float32)
            except FloatingPointError:
                self.last_q_values = np.full(self.config.action_size, np.nan, dtype=np.float32)
            try:
                self.last_target_q_values = np.asarray(
                    self.trainer.predict(values, target=True),
                    dtype=np.float32,
                )
            except FloatingPointError:
                self.last_target_q_values = np.full(
                    self.config.action_size,
                    np.nan,
                    dtype=np.float32,
                )
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
            "gradient_to_clip_ratio": metrics["gradient_to_clip_ratio"],
            "gradient_clipped": metrics["gradient_clipped"],
            "target_mean": metrics["target_q_mean"],
            "predicted_mean": metrics["predicted_q_mean"],
            "bootstrap_mean": metrics["bootstrap_mean"],
            "td_error_mean": metrics["td_error_mean"],
            "td_error_abs_mean": metrics["td_error_abs_mean"],
            "td_error_abs_max": metrics["td_error_abs_max"],
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
            "health": self.health_telemetry(),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        include_replay: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        trainer_finite, invalid_fields = self.trainer.finite_diagnostics()
        if not trainer_finite:
            raise ValueError(
                "cannot save a non-finite learner: " + ", ".join(invalid_fields)
            )
        env_steps = strict_int(self.env_steps, "env_steps", minimum=0)
        episodes = strict_int(self.episodes, "episodes", minimum=0)
        episode_return = finite_float(self.episode_return, "episode_return")
        last_reward = finite_float(self.last_reward, "last_reward")
        last_action_index = action_index(self.last_action_index, self.config.action_size)
        if not isinstance(self.last_policy_mode, str) or self.last_policy_mode not in {"explore", "exploit"}:
            raise ValueError("last_policy_mode must be 'explore' or 'exploit'")
        metadata_mapping = {} if metadata is None else require_mapping(metadata, "metadata")
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "config": self.config.to_dict(),
            "trainer": self.trainer.state_dict(),
            "agent": {
                "env_steps": env_steps,
                "transitions_observed": strict_int(
                    self.transitions_observed,
                    "transitions_observed",
                    minimum=0,
                ),
                "episodes": episodes,
                "episode_return": episode_return,
                "last_reward": last_reward,
                "last_action_index": last_action_index,
                "last_policy_mode": self.last_policy_mode,
                "numpy_rng_state": self.rng.bit_generator.state,
                "torch_rng_state": torch.random.get_rng_state(),
            },
            "metadata": dict(metadata_mapping),
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
        self.checkpoint_metadata = dict(metadata_mapping)
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
        raw_payload = _safe_torch_load(source, self.trainer.device)
        payload = require_mapping(raw_payload, "checkpoint")
        checkpoint_version = strict_int(
            payload.get("checkpoint_version", 0),
            "checkpoint_version",
            minimum=0,
        )
        if checkpoint_version > CHECKPOINT_VERSION:
            raise ValueError("checkpoint was created by a newer learning-core version")
        if "config" not in payload:
            raise ValueError("checkpoint is missing config")
        saved_config = DQNConfig.from_dict(require_mapping(payload["config"], "config"))
        architecture = (saved_config.observation_size, saved_config.action_size, saved_config.hidden_sizes)
        expected = (self.config.observation_size, self.config.action_size, self.config.hidden_sizes)
        if architecture != expected:
            raise ValueError("checkpoint network architecture does not match this agent")
        if saved_config.algorithm != self.config.algorithm:
            raise ValueError("checkpoint algorithm does not match this agent")
        if "trainer" not in payload:
            raise ValueError("checkpoint is missing trainer")
        trainer_state = require_mapping(payload["trainer"], "trainer")
        agent_state = require_mapping(payload.get("agent", {}), "agent")
        env_steps = strict_int(agent_state.get("env_steps", 0), "env_steps", minimum=0)
        transitions_observed = strict_int(
            agent_state.get("transitions_observed", env_steps),
            "transitions_observed",
            minimum=0,
        )
        episodes = strict_int(agent_state.get("episodes", 0), "episodes", minimum=0)
        episode_return = finite_float(
            agent_state.get("episode_return", 0.0),
            "episode_return",
        )
        last_reward = finite_float(agent_state.get("last_reward", 0.0), "last_reward")
        last_action_index = action_index(
            agent_state.get("last_action_index", 0),
            self.config.action_size,
        )
        last_policy_mode = agent_state.get("last_policy_mode", "explore")
        if not isinstance(last_policy_mode, str) or last_policy_mode not in {"explore", "exploit"}:
            raise ValueError("last_policy_mode must be 'explore' or 'exploit'")
        metadata = require_mapping(payload.get("metadata", {}), "metadata")

        numpy_rng_state = agent_state.get("numpy_rng_state")
        if numpy_rng_state is not None:
            rng_validator = np.random.default_rng()
            try:
                rng_validator.bit_generator.state = numpy_rng_state
            except (TypeError, ValueError) as error:
                raise ValueError("numpy_rng_state is malformed") from error
        torch_rng_state = agent_state.get("torch_rng_state")
        if torch_rng_state is not None:
            if not isinstance(torch_rng_state, torch.Tensor):
                raise ValueError("torch_rng_state must be a tensor")
            torch_rng_state = torch_rng_state.detach().cpu()
            try:
                torch.Generator(device="cpu").set_state(torch_rng_state)
            except (TypeError, RuntimeError) as error:
                raise ValueError("torch_rng_state is malformed") from error

        staged_memory: ReplayBuffer | None = None
        if load_replay and "replay" in payload:
            staged_memory = ReplayBuffer(
                self.config.replay_capacity,
                observation_size=self.config.observation_size,
                action_size=self.config.action_size,
                seed=self.config.seed,
            )
            staged_memory.load_state_dict(require_mapping(payload["replay"], "replay"))

        # Build and validate a complete replacement before touching live state.
        # A malformed optimizer/model/replay/RNG therefore cannot leave a
        # half-restored agent behind.
        with torch.random.fork_rng(devices=[]):
            staged_model = PacmanQNetwork(
                self.config.observation_size,
                self.config.action_size,
                self.config.hidden_sizes,
            )
            staged_trainer = DQNTrainer(staged_model, self.config)
            staged_trainer.load_state_dict(trainer_state, load_optimizer=load_optimizer)

        self.model = staged_model
        self.trainer = staged_trainer
        if staged_memory is not None:
            self.memory = staged_memory
        self.env_steps = env_steps
        self.transitions_observed = transitions_observed
        self.episodes = episodes
        self.episode_return = episode_return
        self.last_reward = last_reward
        self.last_action_index = last_action_index
        self.last_policy_mode = last_policy_mode
        if numpy_rng_state is not None:
            self.rng.bit_generator.state = numpy_rng_state
        if torch_rng_state is not None:
            torch.random.set_rng_state(torch_rng_state)
        self.checkpoint_metadata = dict(metadata)
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
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint not found: {source}")
        payload = require_mapping(_safe_torch_load(source, "cpu"), "checkpoint")
        checkpoint_version = strict_int(
            payload.get("checkpoint_version", 0),
            "checkpoint_version",
            minimum=0,
        )
        if checkpoint_version > CHECKPOINT_VERSION:
            raise ValueError("checkpoint was created by a newer learning-core version")
        if "config" not in payload:
            raise ValueError("checkpoint is missing config")
        config = DQNConfig.from_dict(require_mapping(payload["config"], "config"))
        if device is not None:
            config = replace(config, device=device)
        agent = cls(config)
        agent.load_checkpoint(source, load_optimizer=load_optimizer, load_replay=load_replay)
        return agent

    save = save_checkpoint
    load = load_checkpoint


Agent = PacmanDQNAgent
