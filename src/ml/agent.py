import torch
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Optional

from .models import LinearQNet
from .trainer import QTrainer
from src.game.constants import Direction, Point
from src.game.snake_game import SnakeGameAI
from src.config.settings import *
from src.utils.helpers import (
    get_next_model_version, save_model_metadata,
    get_best_model_info, get_latest_model_info
)


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.current_version = None
        self.loaded_metadata = None
        self.previous_distances = deque(maxlen=4)  # Track last 4 distances to food

        self.model = LinearQNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI) -> np.ndarray:
        head = game.snake[0]

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def get_distance_to_food(self, game: SnakeGameAI) -> float:
        return abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)

    def calculate_reward(self, game: SnakeGameAI, done: bool, score: int, old_score: int) -> int:
        reward = 0

        if done:
            if game.is_collision():
                reward = COLLISION_PENALTY
            else:  # Timeout
                reward = LOOP_PENALTY
        elif score > old_score:  # Ate food
            reward = 10
        else:
            # Distance-based reward
            current_distance = self.get_distance_to_food(game)
            self.previous_distances.append(current_distance)

            if len(self.previous_distances) >= 2:
                if current_distance < self.previous_distances[-2]:
                    reward = CLOSER_TO_FOOD_REWARD
                elif current_distance > self.previous_distances[-2]:
                    reward = FARTHER_FROM_FOOD_PENALTY

            # Penalty for loops (same position repeated)
            if len(self.previous_distances) >= 4:
                if (self.previous_distances[-1] == self.previous_distances[-3] and
                        self.previous_distances[-2] == self.previous_distances[-4]):
                    reward = LOOP_PENALTY

        return reward

    def remember(self, state: np.ndarray, action: List[int], reward: int,
                 next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: List[int], reward: int,
                           next_state: np.ndarray, done: bool) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> List[int]:
        # Improved epsilon decay
        self.epsilon = max(EPSILON_MIN, EPSILON_DECAY - self.n_games)
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_model(self, score: int) -> None:
        if self.current_version is None:
            self.current_version = get_next_model_version(MODEL_DIR)

        filename = f"model_v{self.current_version:03d}.pth"
        self.model.save(filename)
        save_model_metadata(MODEL_DIR, self.current_version, score, self.n_games)
        print(f"Model saved as version {self.current_version} with score {score}")

    def load_best_model(self) -> bool:
        model_info = get_best_model_info(MODEL_DIR)
        if model_info:
            model_file, metadata = model_info
            try:
                self.model.load(model_file)
                self.loaded_metadata = metadata
                print(f"Loaded best model: {model_file}")
                print(f"  - Version: {metadata['version']}")
                print(f"  - Best Score: {metadata['best_score']}")
                print(f"  - Games Played: {metadata['games_played']}")
                return True
            except Exception as e:
                print(f"Error loading best model: {e}")
                return False
        return False

    def load_latest_model(self) -> bool:
        model_info = get_latest_model_info(MODEL_DIR)
        if model_info:
            model_file, metadata = model_info
            try:
                self.model.load(model_file)
                self.loaded_metadata = metadata
                print(f"Loaded latest model: {model_file}")
                print(f"  - Version: {metadata['version']}")
                print(f"  - Best Score: {metadata['best_score']}")
                print(f"  - Games Played: {metadata['games_played']}")
                return True
            except Exception as e:
                print(f"Error loading latest model: {e}")
                return False
        return False

    def get_loaded_best_score(self) -> int:
        if self.loaded_metadata:
            return self.loaded_metadata.get('best_score', 0)
        return 0