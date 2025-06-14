import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union, Tuple, List
import numpy as np


class QTrainer:

    def __init__(self, model: nn.Module, learning_rate: float, gamma: float):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(
            self,
            state: Union[np.ndarray, List],
            action: Union[np.ndarray, List],
            reward: Union[float, List],
            next_state: Union[np.ndarray, List],
            done: Union[bool, List]
    ) -> None:
        """
        Perform one training step using the Q-learning algorithm.

        Args:
            state: Current state(s)
            action: Action(s) taken
            reward: Reward(s) received
            next_state: Next state(s) after action
            done: Whether the episode(s) ended
        """
        # Convert inputs to tensors
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # Handle single sample case by adding batch dimension
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Get predicted Q values for current states
        pred = self.model(state)
        target = pred.clone()

        # Update Q values using Bellman equation
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = q_new

        # Perform gradient descent
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()