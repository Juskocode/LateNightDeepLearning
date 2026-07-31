"""Snake agent: observation encoding, policy, replay, and checkpoint lifecycle."""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np
import torch

from .models import LinearQNet
from .replay import Experience, ReplayBuffer
from .trainer import QTrainer
from snakeGameQDlearning.src.config.settings import (
    BATCH_SIZE, BLOCK_SIZE, CLOSER_TO_FOOD_REWARD, COLLISION_PENALTY,
    EPSILON_DECAY_GAMES, EPSILON_MIN, EPSILON_START, FARTHER_FROM_FOOD_PENALTY,
    FOOD_REWARD, GAMMA, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE, LOOP_PENALTY,
    MAX_MEMORY, MODEL_DIR, OUTPUT_SIZE, REVISIT_PENALTY, WIN_REWARD,
)
from snakeGameQDlearning.src.game.constants import Direction, Point
from snakeGameQDlearning.src.game.snake_game import SnakeGameAI
from snakeGameQDlearning.src.utils.helpers import (
    get_best_model_info, get_latest_model_info, get_next_model_version,
    save_model_metadata, update_model_metadata,
)


class Agent:
    def __init__(self, algorithm: str = "double_dqn", seed: int | None = None):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        self.algorithm = algorithm
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.memory = ReplayBuffer(MAX_MEMORY, seed=seed)
        self.current_version = None
        self.loaded_metadata = None
        self.previous_distances = deque(maxlen=4)
        self.last_reward = 0.0
        self.last_action_index = 0
        self.last_q_values = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        self.last_target_q_values = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        self.last_policy_mode = "explore"
        self.model = LinearQNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma,
                                algorithm=algorithm)

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
            (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u)) or (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            game.food is not None and game.food.x < game.head.x,
            game.food is not None and game.food.x > game.head.x,
            game.food is not None and game.food.y < game.head.y,
            game.food is not None and game.food.y > game.head.y,
        ]
        return np.asarray(state, dtype=np.float32)

    def get_distance_to_food(self, game: SnakeGameAI) -> float:
        if game.food is None:
            return 0.0
        return abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)

    def calculate_reward(self, game: SnakeGameAI, done: bool, score: int, old_score: int) -> float:
        if done:
            if game.termination_reason == "win":
                reward = FOOD_REWARD + WIN_REWARD
            elif game.termination_reason == "timeout":
                reward = LOOP_PENALTY
            elif game.termination_reason == "quit":
                reward = 0.0
            else:
                reward = COLLISION_PENALTY
        elif game.last_ate_food or score > old_score:
            reward = float(FOOD_REWARD)
        else:
            distance = self.get_distance_to_food(game)
            self.previous_distances.append(distance)
            reward = CLOSER_TO_FOOD_REWARD if game.last_distance_delta < 0 else (
                FARTHER_FROM_FOOD_PENALTY if game.last_distance_delta > 0 else 0.0
            )
            if game.last_visit_count > 1:
                reward += REVISIT_PENALTY
        self.last_reward = float(reward)
        return float(reward)

    def remember(self, state: np.ndarray, action: List[int], reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        self.memory.append(Experience(np.asarray(state, dtype=np.float32).copy(), list(action),
                                      float(reward),
                                      np.asarray(next_state, dtype=np.float32).copy(), bool(done)))

    def train_long_memory(self) -> float:
        sample = self.memory.sample(BATCH_SIZE)
        if not sample:
            return 0.0
        states, actions, rewards, next_states, dones = zip(*sample)
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: List[int], reward: float,
                           next_state: np.ndarray, done: bool) -> float:
        return self.trainer.train_step(state, action, reward, next_state, done)

    @staticmethod
    def _predict_model(model, state: np.ndarray) -> np.ndarray:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            values = model(torch.as_tensor(state, dtype=torch.float32)).cpu().numpy()
        if was_training:
            model.train()
        return values

    def get_action(self, state: np.ndarray) -> List[int]:
        progress = min(1.0, self.n_games / EPSILON_DECAY_GAMES)
        self.epsilon = max(EPSILON_MIN, EPSILON_START + progress * (EPSILON_MIN - EPSILON_START))
        self.last_q_values = self._predict_model(self.model, state)
        self.last_target_q_values = self._predict_model(self.trainer.target_model, state)
        if self.rng.random() < self.epsilon:
            move = int(self.rng.integers(0, OUTPUT_SIZE))
            self.last_policy_mode = "explore"
        else:
            move = int(np.argmax(self.last_q_values))
            self.last_policy_mode = "exploit"
        self.last_action_index = move
        action = [0] * OUTPUT_SIZE
        action[move] = 1
        return action

    def telemetry(self, state: np.ndarray, game: SnakeGameAI | None = None,
                  episode_return: float = 0.0) -> dict:
        recent = self.memory.tail(24)
        sync_offset = self.trainer.update_target_counter % self.trainer.target_update_freq
        return {
            "algorithm": self.algorithm,
            "state": state.tolist(),
            "q_values": self.last_q_values.tolist(),
            "target_q_values": self.last_target_q_values.tolist(),
            "action_index": self.last_action_index,
            "policy_mode": self.last_policy_mode,
            "epsilon": self.epsilon,
            "reward": self.last_reward,
            "loss": self.trainer.last_loss,
            "gradient_norm": self.trainer.last_gradient_norm,
            "target_mean": self.trainer.last_target_mean,
            "target_sync_progress": sync_offset / self.trainer.target_update_freq,
            "games": self.n_games,
            "episode_return": episode_return,
            "memory": len(self.memory),
            "memory_capacity": self.memory.capacity,
            "memory_stats": self.memory.stats(),
            "recent_rewards": [experience.reward for experience in recent],
            "recent_dones": [experience.done for experience in recent],
            "recent_actions": [int(np.argmax(experience.action)) for experience in recent],
            "termination_reason": game.termination_reason if game is not None else None,
        }

    def save_model_new_record(self, best_score: int, mean_score: float) -> None:
        self.current_version = get_next_model_version(str(MODEL_DIR))
        filename = f"model_v{self.current_version:03d}.pth"
        self.model.save(filename)
        save_model_metadata(str(MODEL_DIR), self.current_version, best_score, mean_score, self.n_games)
        print(f"New record: saved {filename} with score {best_score}")

    def update_model_mean_score(self, mean_score: float) -> None:
        if self.current_version is not None:
            update_model_metadata(str(MODEL_DIR), self.current_version, mean_score, self.n_games)

    def _load_model_info(self, model_info) -> bool:
        if not model_info:
            return False
        model_file, metadata = model_info
        try:
            self.model.load(model_file)
            self.trainer.target_model.load_state_dict(self.model.state_dict())
            self.loaded_metadata = metadata
            self.current_version = metadata.get("version")
            self.n_games = int(metadata.get("games_played", 0))
            print(f"Loaded {model_file}: score={metadata.get('best_score', 0)}, games={metadata.get('games_played', 0)}")
            return True
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Could not load {model_file}: {error}")
            return False

    def load_best_model(self) -> bool:
        return self._load_model_info(get_best_model_info(str(MODEL_DIR)))

    def load_latest_model(self) -> bool:
        return self._load_model_info(get_latest_model_info(str(MODEL_DIR)))

    def get_loaded_best_score(self) -> int:
        return int(self.loaded_metadata.get("best_score", 0)) if self.loaded_metadata else 0

    def get_loaded_mean_score(self) -> float:
        return float(self.loaded_metadata.get("mean_score", 0.0)) if self.loaded_metadata else 0.0
