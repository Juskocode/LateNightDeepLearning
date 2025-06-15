import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union, Tuple, List
import numpy as np


class QTrainer:
    def __init__(self, model: nn.Module, learning_rate: float, gamma: float):
        self.model = model
        self.target_model = type(model)(model.linear1.in_features,
                                        model.linear1.out_features,
                                        model.linear3.out_features)
        self.target_model.load_state_dict(model.state_dict())

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_counter = 0
        self.target_update_freq = 100

    def train_step(self, state: Union[np.ndarray, List], action: Union[np.ndarray, List],
                   reward: Union[float, List], next_state: Union[np.ndarray, List],
                   done: Union[bool, List]) -> None:

        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Double DQN implementation
        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                # Use main network to select action, target network to evaluate
                next_q_values = self.model(next_state[idx])
                best_action = torch.argmax(next_q_values)
                target_q_values = self.target_model(next_state[idx])
                q_new = reward[idx] + self.gamma * target_q_values[best_action]

            target[idx][torch.argmax(action[idx]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.update_target_counter += 1
        if self.update_target_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())