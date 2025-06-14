import torch
import random
import numpy as np
from collections import deque
from typing import List, Tuple

from .models import LinearQNet
from .trainer import QTrainer
from src.game.constants import Direction, Point
from src.game.snake_game import SnakeGameAI
from src.config.settings import (
    MAX_MEMORY, BATCH_SIZE, LEARNING_RATE, GAMMA, EPSILON_DECAY,
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BLOCK_SIZE
)


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness parameter
        self.gamma = GAMMA  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience replay buffer

        # Initialize neural network and trainer
        self.model = LinearQNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI) -> np.ndarray:
        head = game.snake[0]

        # Points around the head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Create state vector
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location relative to head
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state: np.ndarray, action: List[int], reward: int,
                 next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        """Train on a batch of experiences from memory."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: List[int], reward: int,
                           next_state: np.ndarray, done: bool) -> None:
        """
        Train on a single experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> List[int]:
        """
        Get action using epsilon-greedy strategy.

        Args:
            state: Current game state

        Returns:
            Action as one-hot encoded list [straight, right, left]
        """
        # Epsilon-greedy: balance exploration vs exploitation
        self.epsilon = EPSILON_DECAY - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # Random move (exploration)
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Predicted move (exploitation)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_model(self, filename: str = 'model.pth') -> None:
        self.model.save(filename)

    def load_model(self, filename: str = 'model.pth') -> None:
        self.model.load(filename)