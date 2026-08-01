"""Snake agent: observation encoding, policy, replay, and checkpoint lifecycle."""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np
import torch

from .algorithms import create_algorithm, normalize_algorithm_name
from .replay import Experience, ReplayBuffer
from snakeGameQDlearning.src.config.settings import (
    BATCH_SIZE,
    BLOCK_SIZE,
    CLOSER_TO_FOOD_REWARD,
    COLLISION_PENALTY,
    EPSILON_DECAY_GAMES,
    EPSILON_MIN,
    EPSILON_START,
    FARTHER_FROM_FOOD_PENALTY,
    FOOD_REWARD,
    GAMMA,
    LOOP_PENALTY,
    MAX_MEMORY,
    MODEL_DIR,
    OUTPUT_SIZE,
    REVISIT_PENALTY,
    WIN_REWARD,
)
from snakeGameQDlearning.src.game.constants import Direction, Point
from snakeGameQDlearning.src.game.snake_game import SnakeGameAI
from snakeGameQDlearning.src.utils.helpers import (
    get_best_model_info,
    get_latest_model_info,
    get_next_model_version,
    save_model_metadata,
    update_model_metadata,
)


class Agent:
    def __init__(self, algorithm: str = "double_dqn", seed: int | None = None):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        self.algorithm = normalize_algorithm_name(algorithm)
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
        self.evaluation_metrics = {
            "episodes": 0,
            "mean_score": 0.0,
            "std_score": 0.0,
            "median_score": 0.0,
            "max_score": 0,
            "generalization_gap": 0.0,
        }
        self.curriculum_stage = "orientation"
        self.learning = create_algorithm(self.algorithm)
        # These compatibility aliases keep existing notebooks and tests useful.
        self.model = self.learning.model
        self.trainer = self.learning.trainer

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
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
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

    def calculate_reward(
        self, game: SnakeGameAI, done: bool, score: int, old_score: int
    ) -> float:
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
            reward = (
                CLOSER_TO_FOOD_REWARD
                if game.last_distance_delta < 0
                else (
                    FARTHER_FROM_FOOD_PENALTY if game.last_distance_delta > 0 else 0.0
                )
            )
            if game.last_visit_count > 1:
                reward += REVISIT_PENALTY
        self.last_reward = float(reward)
        return float(reward)

    def remember(
        self,
        state: np.ndarray,
        action: List[int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append(
            Experience(
                np.asarray(state, dtype=np.float32).copy(),
                list(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                bool(done),
            )
        )

    def train_long_memory(self) -> float:
        if not self.learning.supports_replay:
            return 0.0
        sample = self.memory.sample(BATCH_SIZE)
        if not sample:
            return 0.0
        return self.learning.train_step(sample, self.epsilon)

    def train_short_memory(
        self,
        state: np.ndarray,
        action: List[int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        experience = Experience(
            np.asarray(state, dtype=np.float32),
            list(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        return self.learning.train_transition(experience, self.epsilon)

    @staticmethod
    def _predict_model(model, state: np.ndarray) -> np.ndarray:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            values = model(torch.as_tensor(state, dtype=torch.float32)).cpu().numpy()
        if was_training:
            model.train()
        return values

    def get_action(self, state: np.ndarray, *, explore: bool = True) -> List[int]:
        progress = min(1.0, self.n_games / EPSILON_DECAY_GAMES)
        self.epsilon = max(
            EPSILON_MIN, EPSILON_START + progress * (EPSILON_MIN - EPSILON_START)
        )
        self.last_q_values = self.learning.predict(state)
        self.last_target_q_values = self.learning.target_predict(state)
        if explore and self.rng.random() < self.epsilon:
            move = int(self.rng.integers(0, OUTPUT_SIZE))
            self.last_policy_mode = "explore"
        else:
            move = int(np.argmax(self.last_q_values))
            self.last_policy_mode = "exploit" if explore else "evaluate"
        self.last_action_index = move
        action = [0] * OUTPUT_SIZE
        action[move] = 1
        return action

    def update_evaluation_metrics(self, metrics: dict, training_mean: float) -> None:
        """Attach held-out evaluation results without changing learned state."""

        self.evaluation_metrics = dict(metrics)
        self.evaluation_metrics["generalization_gap"] = float(training_mean) - float(
            metrics.get("mean_score", 0.0)
        )

    def telemetry(
        self,
        state: np.ndarray,
        game: SnakeGameAI | None = None,
        episode_return: float = 0.0,
    ) -> dict:
        recent = self.memory.tail(24)
        return {
            "algorithm": self.algorithm,
            "state": state.tolist(),
            "q_values": self.last_q_values.tolist(),
            "target_q_values": self.last_target_q_values.tolist(),
            "action_index": self.last_action_index,
            "policy_mode": self.last_policy_mode,
            "epsilon": self.epsilon,
            "reward": self.last_reward,
            "algorithm_family": self.learning.info.family,
            "algorithm_description": self.learning.info.description,
            "model_structure": self.learning.structure_label,
            "learned_states": self.learning.learned_states,
            "loss": self.learning.last_loss,
            "gradient_norm": self.learning.last_gradient_norm,
            "target_mean": self.learning.last_target_mean,
            "target_sync_progress": self.learning.target_sync_progress,
            "games": self.n_games,
            "episode_return": episode_return,
            "memory": len(self.memory),
            "memory_capacity": self.memory.capacity,
            "memory_stats": self.memory.stats(),
            "recent_rewards": [experience.reward for experience in recent],
            "recent_dones": [experience.done for experience in recent],
            "recent_actions": [
                int(np.argmax(experience.action)) for experience in recent
            ],
            "termination_reason": game.termination_reason if game is not None else None,
            "episode_seed": (
                getattr(game, "episode_seed", None) if game is not None else None
            ),
            "curriculum_stage": self.curriculum_stage,
            "evaluation": dict(self.evaluation_metrics),
        }

    def save_model_checkpoint(
        self,
        best_score: int,
        mean_score: float,
        *,
        reason: str,
        evaluation_mean: float | None = None,
        experiment: dict | None = None,
    ) -> None:
        self.current_version = get_next_model_version(str(MODEL_DIR))
        filename = f"model_v{self.current_version:03d}.pth"
        self.learning.save(filename, MODEL_DIR)
        save_model_metadata(
            str(MODEL_DIR),
            self.current_version,
            best_score,
            mean_score,
            self.n_games,
            algorithm=self.algorithm,
            evaluation_mean=evaluation_mean,
            checkpoint_reason=reason,
            experiment=experiment,
        )
        print(f"Saved {filename}: {reason.replace('_', ' ')}")

    def save_model_new_record(self, best_score: int, mean_score: float) -> None:
        """Backward-compatible record checkpoint entry point."""

        self.save_model_checkpoint(best_score, mean_score, reason="training_record")

    def update_model_mean_score(self, mean_score: float) -> None:
        if self.current_version is not None:
            update_model_metadata(
                str(MODEL_DIR), self.current_version, mean_score, self.n_games
            )

    def _load_model_info(self, model_info) -> bool:
        if not model_info:
            return False
        model_file, metadata = model_info
        try:
            self.learning.load(model_file, MODEL_DIR)
            self.loaded_metadata = metadata
            self.current_version = metadata.get("version")
            self.n_games = int(metadata.get("games_played", 0))
            print(
                f"Loaded {model_file}: score={metadata.get('best_score', 0)}, games={metadata.get('games_played', 0)}"
            )
            return True
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Could not load {model_file}: {error}")
            return False

    def load_best_model(
        self,
        *,
        environment: str | None = None,
        validation_seeds: tuple[int, ...] | None = None,
    ) -> bool:
        return self._load_model_info(
            get_best_model_info(
                str(MODEL_DIR),
                algorithm=self.algorithm,
                environment=environment,
                validation_seeds=validation_seeds,
            )
        )

    def load_latest_model(
        self,
        *,
        environment: str | None = None,
        validation_seeds: tuple[int, ...] | None = None,
    ) -> bool:
        return self._load_model_info(
            get_latest_model_info(
                str(MODEL_DIR),
                algorithm=self.algorithm,
                environment=environment,
                validation_seeds=validation_seeds,
            )
        )

    def get_loaded_best_score(self) -> int:
        return (
            int(self.loaded_metadata.get("best_score", 0))
            if self.loaded_metadata
            else 0
        )

    def get_loaded_mean_score(self) -> float:
        return (
            float(self.loaded_metadata.get("mean_score", 0.0))
            if self.loaded_metadata
            else 0.0
        )

    def get_loaded_evaluation_mean(self) -> float | None:
        if not self.loaded_metadata or "evaluation_mean" not in self.loaded_metadata:
            return None
        return float(self.loaded_metadata["evaluation_mean"])
