"""DQN and Double-DQN target calculations with shared training machinery."""

from __future__ import annotations

from copy import deepcopy
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
        self.model = model
        self.target_model = deepcopy(model)
        self.target_model.eval()
        self.algorithm = algorithm
        self.gamma = gamma
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.SmoothL1Loss()
        self.update_target_counter = 0
        self.target_update_freq = TARGET_UPDATE_FREQUENCY
        self.last_loss = 0.0
        self.last_target_mean = 0.0
        self.last_predicted_mean = 0.0
        self.last_gradient_norm = 0.0

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
        states = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        next_states = torch.as_tensor(np.asarray(next_state), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(action), dtype=torch.long)
        rewards = torch.as_tensor(np.asarray(reward), dtype=torch.float32)
        dones = torch.as_tensor(np.asarray(done), dtype=torch.bool)
        if states.ndim == 1:
            states = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
            dones = dones.unsqueeze(0)
        input_size = getattr(self.model, "input_size", self.model.linear1.in_features)
        output_size = getattr(
            self.model,
            "output_size",
            getattr(getattr(self.model, "linear3", None), "out_features", None),
        )
        if states.ndim != 2 or states.shape[1] != input_size:
            raise ValueError("state batch has an unexpected shape")
        if actions.ndim != 2 or actions.shape[1] != output_size:
            raise ValueError("action batch has an unexpected shape")
        if not torch.all((actions == 0) | (actions == 1)) or not torch.all(
            actions.sum(dim=1) == 1
        ):
            raise ValueError("actions must be one-hot encoded")

        self.model.train()
        selected_action_indices = actions.argmax(dim=1, keepdim=True)
        predicted_q = self.model(states).gather(1, selected_action_indices).squeeze(1)
        bootstrap = self._bootstrap_values(next_states)
        targets = rewards + (~dones).float() * self.gamma * bootstrap

        self.optimizer.zero_grad()
        loss = self.criterion(predicted_q, targets)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        self.update_target_counter += 1
        if self.update_target_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.last_loss = float(loss.detach())
        self.last_target_mean = float(targets.mean().detach())
        self.last_predicted_mean = float(predicted_q.mean().detach())
        self.last_gradient_norm = float(gradient_norm.detach())
        return self.last_loss
