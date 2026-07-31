"""Correct DQN and Double-DQN optimization with measurable training state."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .config import Algorithm, DQNConfig, normalize_algorithm
from .models import PacmanQNetwork


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    step: int = 0
    batch_size: int = 0
    loss: float = 0.0
    gradient_norm: float = 0.0
    predicted_q_mean: float = 0.0
    target_q_mean: float = 0.0
    bootstrap_mean: float = 0.0
    td_error_mean: float = 0.0
    td_error_abs_mean: float = 0.0
    reward_mean: float = 0.0
    target_synced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DQNTrainer:
    def __init__(self, model: PacmanQNetwork, config: DQNConfig):
        if model.input_size != config.observation_size or model.output_size != config.action_size:
            raise ValueError("model dimensions do not match DQNConfig")
        self.model = model
        self.config = config
        self.algorithm: Algorithm = normalize_algorithm(config.algorithm)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.target_model = copy.deepcopy(model).to(self.device)
        self.target_model.eval()
        self.target_model.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.SmoothL1Loss(reduction="none")
        self.train_steps = 0
        self.last_metrics = TrainingMetrics()

    def synchronize_target(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def _states_tensor(self, values: Any, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.config.observation_size:
            raise ValueError(f"{name} must have shape (batch, {self.config.observation_size})")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite values")
        return tensor

    def _action_indices(self, values: Any, batch_size: int) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(values), device=self.device)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        elif tensor.ndim == 2:
            if tensor.shape != (batch_size, self.config.action_size):
                raise ValueError("one-hot actions have an unexpected shape")
            if not torch.all((tensor == 0) | (tensor == 1)) or not torch.all(tensor.sum(dim=1) == 1):
                raise ValueError("one-hot actions must contain exactly one selected action")
            tensor = tensor.argmax(dim=1)
        elif tensor.ndim == 1 and batch_size == 1 and tensor.numel() == self.config.action_size:
            if torch.all((tensor == 0) | (tensor == 1)) and tensor.sum() == 1:
                tensor = tensor.argmax().reshape(1)
        if tensor.ndim != 1 or tensor.numel() != batch_size:
            raise ValueError("actions must be indices or one-hot vectors")
        tensor = tensor.long()
        if not torch.all((0 <= tensor) & (tensor < self.config.action_size)):
            raise ValueError("action index is outside the configured action space")
        return tensor

    def _bootstrap_values(self, next_states: torch.Tensor) -> torch.Tensor:
        """Compute the algorithm-specific bootstrap estimate.

        DQN selects and evaluates with the target network. Double-DQN selects
        with the online network and independently evaluates with the target.
        """

        with torch.no_grad():
            target_values = self.target_model(next_states)
            if self.algorithm == "dqn":
                return target_values.max(dim=1).values
            selected = self.model(next_states).argmax(dim=1, keepdim=True)
            return target_values.gather(1, selected).squeeze(1)

    def compute_targets(
        self,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bootstrap = self._bootstrap_values(next_states)
        targets = rewards + (~dones).float() * self.config.gamma * bootstrap
        return targets, bootstrap

    def update(
        self,
        states: Any,
        actions: Any,
        rewards: Any,
        next_states: Any,
        dones: Any,
    ) -> TrainingMetrics:
        state_tensor = self._states_tensor(states, "states")
        next_state_tensor = self._states_tensor(next_states, "next_states")
        if state_tensor.shape[0] != next_state_tensor.shape[0]:
            raise ValueError("states and next_states batch sizes differ")
        batch_size = state_tensor.shape[0]
        action_tensor = self._action_indices(actions, batch_size)
        reward_tensor = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device).reshape(-1)
        done_tensor = torch.as_tensor(np.asarray(dones), dtype=torch.bool, device=self.device).reshape(-1)
        if reward_tensor.numel() != batch_size or done_tensor.numel() != batch_size:
            raise ValueError("reward and done batch sizes must match states")
        if not torch.isfinite(reward_tensor).all():
            raise ValueError("rewards must contain finite values")

        self.model.train()
        predicted = self.model(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        targets, bootstrap = self.compute_targets(next_state_tensor, reward_tensor, done_tensor)
        per_sample_loss = self.criterion(predicted, targets)
        loss = per_sample_loss.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.gradient_clip
        )
        self.optimizer.step()
        self.train_steps += 1
        target_synced = self.train_steps % self.config.target_update_interval == 0
        if target_synced:
            self.synchronize_target()

        td_error = targets.detach() - predicted.detach()
        self.last_metrics = TrainingMetrics(
            step=self.train_steps,
            batch_size=int(batch_size),
            loss=float(loss.detach().cpu()),
            gradient_norm=float(gradient_norm.detach().cpu()),
            predicted_q_mean=float(predicted.detach().mean().cpu()),
            target_q_mean=float(targets.detach().mean().cpu()),
            bootstrap_mean=float(bootstrap.detach().mean().cpu()),
            td_error_mean=float(td_error.mean().cpu()),
            td_error_abs_mean=float(td_error.abs().mean().cpu()),
            reward_mean=float(reward_tensor.mean().cpu()),
            target_synced=target_synced,
        )
        return self.last_metrics

    def train_step(self, state: Any, action: Any, reward: Any, next_state: Any, done: Any) -> float:
        return self.update(state, action, reward, next_state, done).loss

    def predict(self, states: Any, *, target: bool = False) -> np.ndarray:
        tensor = self._states_tensor(states, "states")
        network = self.target_model if target else self.model
        was_training = network.training
        network.eval()
        with torch.no_grad():
            values = network(tensor).detach().cpu().numpy()
        if was_training and network is self.model:
            network.train()
        return values[0] if np.asarray(states).ndim == 1 else values

    @property
    def target_sync_progress(self) -> float:
        return (self.train_steps % self.config.target_update_interval) / self.config.target_update_interval

    def state_dict(self) -> dict[str, Any]:
        return {
            "online_model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "last_metrics": self.last_metrics.to_dict(),
            "algorithm": self.algorithm,
        }

    def load_state_dict(self, state: dict[str, Any], *, load_optimizer: bool = True) -> None:
        saved_algorithm = normalize_algorithm(str(state.get("algorithm", self.algorithm)))
        if saved_algorithm != self.algorithm:
            raise ValueError(f"checkpoint uses {saved_algorithm}, agent uses {self.algorithm}")
        self.model.load_state_dict(state["online_model"])
        self.target_model.load_state_dict(state.get("target_model", state["online_model"]))
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            for optimizer_state in self.optimizer.state.values():
                for key, value in optimizer_state.items():
                    if isinstance(value, torch.Tensor):
                        optimizer_state[key] = value.to(self.device)
        self.train_steps = int(state.get("train_steps", 0))
        metrics = state.get("last_metrics", {})
        self.last_metrics = TrainingMetrics(**metrics) if metrics else TrainingMetrics(step=self.train_steps)
        self.target_model.eval()


class QTrainer(DQNTrainer):
    """Compatibility wrapper accepting the original trainer-style arguments."""

    def __init__(
        self,
        model: PacmanQNetwork,
        learning_rate: float,
        gamma: float,
        algorithm: str = "double_dqn",
        *,
        target_update_interval: int = 1_000,
        gradient_clip: float = 10.0,
    ) -> None:
        config = DQNConfig(
            observation_size=model.input_size,
            action_size=model.output_size,
            hidden_sizes=model.hidden_sizes,
            action_labels=tuple(str(index) for index in range(model.output_size)),
            algorithm=normalize_algorithm(algorithm),
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            gradient_clip=gradient_clip,
        )
        super().__init__(model, config)

