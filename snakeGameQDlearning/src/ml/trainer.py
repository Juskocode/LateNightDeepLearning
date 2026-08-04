"""DQN and Double-DQN target calculations with shared training machinery."""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from typing import List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snakeGameQDlearning.src.config.settings import TARGET_UPDATE_FREQUENCY

Algorithm = Literal["dqn", "double_dqn"]


class QTrainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        gamma: float,
        algorithm: Algorithm = "double_dqn",
        weight_decay: float = 1e-5,
    ):
        if algorithm not in ("dqn", "double_dqn"):
            raise ValueError("algorithm must be 'dqn' or 'double_dqn'")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, Real)
            or not math.isfinite(float(learning_rate))
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a positive finite number")
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, Real)
            or not math.isfinite(float(gamma))
            or not 0.0 <= gamma <= 1.0
        ):
            raise ValueError("gamma must be a finite number between 0 and 1")
        if (
            isinstance(weight_decay, bool)
            or not isinstance(weight_decay, Real)
            or not math.isfinite(float(weight_decay))
            or weight_decay < 0
        ):
            raise ValueError("weight_decay must be a non-negative finite number")
        self.model = model
        self.target_model = deepcopy(model)
        self.target_model.eval()
        self.algorithm = algorithm
        self.gamma = float(gamma)
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.SmoothL1Loss()
        self.update_target_counter = 0
        self.target_update_freq = TARGET_UPDATE_FREQUENCY
        if (
            isinstance(self.target_update_freq, bool)
            or not isinstance(self.target_update_freq, int)
            or self.target_update_freq <= 0
        ):
            raise ValueError("TARGET_UPDATE_FREQUENCY must be a positive integer")
        self.gradient_clip_threshold = 1.0
        self.attempted_updates = 0
        self.rejected_updates = 0
        self.gradient_clip_count = 0
        self.last_batch_size = 0
        self.last_rejection: str | None = None
        self.last_loss = 0.0
        self.last_target_mean = 0.0
        self.last_predicted_mean = 0.0
        self.last_gradient_norm = 0.0
        self.last_q_abs_max = 0.0
        self.last_td_error_abs_mean = 0.0
        self.last_td_error_abs_max = 0.0

    def _reject(self, reason: str) -> None:
        self.rejected_updates += 1
        self.last_rejection = reason

    @staticmethod
    def _require_finite(tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")

    def health_metrics(self, decisions: int) -> dict:
        ratio = self.update_target_counter / decisions if decisions > 0 else 0.0
        clip_ratio = (
            self.gradient_clip_count / self.update_target_counter
            if self.update_target_counter
            else 0.0
        )
        numeric = (
            self.last_loss,
            self.last_target_mean,
            self.last_predicted_mean,
            self.last_gradient_norm,
            self.last_q_abs_max,
            self.last_td_error_abs_mean,
            self.last_td_error_abs_max,
            ratio,
            clip_ratio,
        )
        return {
            "finite": all(math.isfinite(float(value)) for value in numeric),
            "updates": self.update_target_counter,
            "attempted_updates": self.attempted_updates,
            "rejected_updates": self.rejected_updates,
            "update_to_decision_ratio": ratio,
            "gradient_norm": self.last_gradient_norm,
            "clip_threshold": self.gradient_clip_threshold,
            "clip_count": self.gradient_clip_count,
            "clip_ratio": clip_ratio,
            "last_batch_size": self.last_batch_size,
            "q_abs_max": self.last_q_abs_max,
            "td_error_abs_mean": self.last_td_error_abs_mean,
            "td_error_abs_max": self.last_td_error_abs_max,
            "last_rejection": self.last_rejection,
        }

    def _bootstrap_values(self, next_states: torch.Tensor) -> torch.Tensor:
        """Return the bootstrap term that distinguishes DQN from Double DQN."""
        with torch.no_grad():
            target_q = self.target_model(next_states)
            if self.algorithm == "dqn":
                # The target network both selects and evaluates the maximum action.
                return target_q.max(dim=1).values
            # The online network selects; the target network independently evaluates.
            # Dropout regularizes the fitted prediction, but it must not turn the
            # Double-DQN bootstrap rule into a different stochastic policy every
            # time the same replay batch is sampled.
            was_training = self.model.training
            self.model.eval()
            try:
                selected_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            finally:
                self.model.train(was_training)
            return target_q.gather(1, selected_actions).squeeze(1)

    def train_step(
        self,
        state: Union[np.ndarray, List],
        action: Union[np.ndarray, List],
        reward: Union[float, List],
        next_state: Union[np.ndarray, List],
        done: Union[bool, List],
    ) -> float:
        self.attempted_updates += 1
        try:
            states = torch.as_tensor(np.asarray(state), dtype=torch.float32)
            next_states = torch.as_tensor(np.asarray(next_state), dtype=torch.float32)
            raw_actions = torch.as_tensor(np.asarray(action))
            rewards = torch.as_tensor(np.asarray(reward), dtype=torch.float32)
            raw_dones = torch.as_tensor(np.asarray(done))
            if states.ndim == 1:
                states = states.unsqueeze(0)
                next_states = next_states.unsqueeze(0)
                raw_actions = raw_actions.unsqueeze(0)
                rewards = rewards.unsqueeze(0)
                raw_dones = raw_dones.unsqueeze(0)
            input_size = getattr(self.model, "input_size", self.model.linear1.in_features)
            output_size = getattr(
                self.model,
                "output_size",
                getattr(getattr(self.model, "linear3", None), "out_features", None),
            )
            if states.ndim != 2 or states.shape[1] != input_size:
                raise ValueError("state batch has an unexpected shape")
            if next_states.shape != states.shape:
                raise ValueError("next_state batch must match state batch shape")
            if raw_actions.ndim != 2 or raw_actions.shape[1] != output_size:
                raise ValueError("action batch has an unexpected shape")
            if rewards.ndim != 1 or rewards.shape[0] != states.shape[0]:
                raise ValueError("reward batch must contain one scalar per state")
            if raw_dones.ndim != 1 or raw_dones.shape[0] != states.shape[0]:
                raise ValueError("done batch must contain one flag per state")
            self._require_finite(states, "state")
            self._require_finite(next_states, "next_state")
            self._require_finite(rewards, "reward")
            self._require_finite(raw_actions.to(dtype=torch.float32), "action")
            actions = raw_actions.to(dtype=torch.long)
            if not torch.equal(raw_actions, actions.to(dtype=raw_actions.dtype)):
                raise ValueError("actions must contain integer one-hot values")
            if not torch.all((actions == 0) | (actions == 1)) or not torch.all(
                actions.sum(dim=1) == 1
            ):
                raise ValueError("actions must be one-hot encoded")
            if raw_dones.dtype != torch.bool:
                raise ValueError("done values must be booleans")
            dones = raw_dones
        except (TypeError, ValueError, RuntimeError) as error:
            self._reject(str(error))
            raise ValueError(str(error)) from error

        self.model.train()
        selected_action_indices = actions.argmax(dim=1, keepdim=True)
        predicted_q = self.model(states).gather(1, selected_action_indices).squeeze(1)
        bootstrap = self._bootstrap_values(next_states)
        targets = rewards + (~dones).float() * self.gamma * bootstrap
        if not torch.isfinite(predicted_q).all() or not torch.isfinite(targets).all():
            self._reject("network produced non-finite Q values or targets")
            raise FloatingPointError(self.last_rejection)

        self.optimizer.zero_grad()
        loss = self.criterion(predicted_q, targets)
        if not torch.isfinite(loss):
            self._reject("loss became non-finite")
            raise FloatingPointError(self.last_rejection)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.gradient_clip_threshold
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            self._reject("gradient norm became non-finite")
            raise FloatingPointError(self.last_rejection)
        self.optimizer.step()

        self.update_target_counter += 1
        self.last_batch_size = int(states.shape[0])
        self.last_rejection = None
        if float(gradient_norm) > self.gradient_clip_threshold:
            self.gradient_clip_count += 1
        if self.update_target_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        td_errors = (targets - predicted_q).abs()
        self.last_loss = float(loss.detach())
        self.last_target_mean = float(targets.mean().detach())
        self.last_predicted_mean = float(predicted_q.mean().detach())
        self.last_gradient_norm = float(gradient_norm.detach())
        self.last_q_abs_max = float(predicted_q.abs().max().detach())
        self.last_td_error_abs_mean = float(td_errors.mean().detach())
        self.last_td_error_abs_max = float(td_errors.max().detach())
        return self.last_loss
