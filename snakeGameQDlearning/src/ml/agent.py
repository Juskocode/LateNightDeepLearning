"""Snake agent: observation encoding, policy, replay, and checkpoint lifecycle."""

from __future__ import annotations

from collections import Counter, deque
import json
import math
from numbers import Integral, Real
import pickle
from typing import List

import numpy as np
import torch

from .algorithms import create_algorithm, normalize_algorithm_name
from .replay import Experience, ReplayBuffer, validated_experience
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
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, Integral)
        ):
            raise ValueError("seed must be an integer or None")
        normalized_seed = None if seed is None else int(seed)
        if normalized_seed is not None:
            torch.manual_seed(normalized_seed)
        self.rng = np.random.default_rng(normalized_seed)
        self.memory = ReplayBuffer(MAX_MEMORY, seed=normalized_seed)
        self.current_version = None
        self.loaded_metadata = None
        self.previous_distances = deque(maxlen=4)
        self.last_reward = 0.0
        self.last_action_index = 0
        self.last_q_values = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        self.last_target_q_values = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        self.last_policy_mode = "explore"
        self.decision_count = 0
        self.evaluation_decision_count = 0
        self.rejected_transition_count = 0
        self.last_transition_rejection: str | None = None
        self.termination_counts: Counter[str] = Counter()
        self.recent_terminations = deque(maxlen=50)
        self._parameters_finite = True
        self._finite_checked_at_decision = -32
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
        if done:
            reason = str(game.termination_reason or "unknown")
            self.termination_counts[reason] += 1
            self.recent_terminations.append(reason)
        return float(reward)

    def remember(
        self,
        state: np.ndarray,
        action: List[int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if not self.learning.supports_replay:
            return
        try:
            self.memory.append(
                validated_experience(
                    Experience(state, action, reward, next_state, done)
                )
            )
            self.last_transition_rejection = None
        except (TypeError, ValueError) as error:
            self.rejected_transition_count += 1
            self.last_transition_rejection = str(error)
            raise

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
        try:
            experience = validated_experience(
                Experience(state, action, reward, next_state, done)
            )
            self.last_transition_rejection = None
        except (TypeError, ValueError) as error:
            self.rejected_transition_count += 1
            self.last_transition_rejection = str(error)
            raise
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
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (11,) or not np.isfinite(state).all():
            raise ValueError("state must be a finite 11-feature vector")
        progress = min(1.0, self.n_games / EPSILON_DECAY_GAMES)
        self.epsilon = max(
            EPSILON_MIN, EPSILON_START + progress * (EPSILON_MIN - EPSILON_START)
        )
        self.last_q_values = self.learning.predict(state)
        self.last_target_q_values = self.learning.target_predict(state)
        if (
            self.last_q_values.shape != (OUTPUT_SIZE,)
            or self.last_target_q_values.shape != (OUTPUT_SIZE,)
            or not np.isfinite(self.last_q_values).all()
            or not np.isfinite(self.last_target_q_values).all()
        ):
            raise FloatingPointError("learning backend produced invalid Q values")
        if explore and self.rng.random() < self.epsilon:
            move = int(self.rng.integers(0, OUTPUT_SIZE))
            self.last_policy_mode = "explore"
        else:
            move = int(np.argmax(self.last_q_values))
            self.last_policy_mode = "exploit" if explore else "evaluate"
        self.last_action_index = move
        if explore:
            self.decision_count += 1
        else:
            self.evaluation_decision_count += 1
        action = [0] * OUTPUT_SIZE
        action[move] = 1
        return action

    def update_evaluation_metrics(self, metrics: dict, training_mean: float) -> None:
        """Attach held-out evaluation results without changing learned state."""
        if not isinstance(metrics, dict):
            raise TypeError("evaluation metrics must be a dictionary")
        if (
            isinstance(training_mean, bool)
            or not isinstance(training_mean, Real)
            or not math.isfinite(float(training_mean))
        ):
            raise ValueError("training_mean must be finite")
        clean = dict(metrics)
        episodes = clean.get("episodes", 0)
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 0:
            raise ValueError("evaluation episodes must be a non-negative integer")
        for key in ("mean_score", "std_score", "median_score", "mean_steps", "win_rate"):
            if key not in clean:
                continue
            value = clean[key]
            if isinstance(value, bool) or not isinstance(value, Real) \
                    or not math.isfinite(float(value)):
                raise ValueError(f"evaluation {key} must be finite")
            clean[key] = float(value)
        clean["episodes"] = episodes
        clean["generalization_gap"] = float(training_mean) - float(
            clean.get("mean_score", 0.0)
        )
        self.evaluation_metrics = clean

    def _termination_diagnostics(self, game: SnakeGameAI | None) -> dict:
        known = ("wall", "self", "timeout", "win", "quit", "unknown")
        counts = {reason: int(self.termination_counts.get(reason, 0)) for reason in known}
        counts.update({reason: int(count) for reason, count in self.termination_counts.items()
                       if reason not in counts})
        total = sum(counts.values())
        return {
            "current": game.termination_reason if game is not None else None,
            "total": total,
            "counts": counts,
            "rates": {reason: count / total if total else 0.0
                      for reason, count in counts.items()},
            "collision_rate": (counts["wall"] + counts["self"]) / total if total else 0.0,
            "timeout_rate": counts["timeout"] / total if total else 0.0,
            "win_rate": counts["win"] / total if total else 0.0,
            "recent": list(self.recent_terminations),
        }

    def _neural_finiteness(self) -> bool:
        if self.learning.info.family != "deep":
            return True
        if self.decision_count - self._finite_checked_at_decision >= 32:
            parameters = (
                *self.learning.model.parameters(),
                *self.learning.trainer.target_model.parameters(),
            )
            self._parameters_finite = all(
                torch.isfinite(parameter).all().item() for parameter in parameters
            )
            self._finite_checked_at_decision = self.decision_count
        return self._parameters_finite

    def _health(self, game: SnakeGameAI | None) -> dict:
        metrics = self.learning.health_metrics(self.decision_count)
        replay_applicable = self.learning.supports_replay
        replay_ready = len(self.memory) >= BATCH_SIZE if replay_applicable else None
        replay = {
            "applicable": replay_applicable,
            "size": len(self.memory) if replay_applicable else None,
            "capacity": self.memory.capacity if replay_applicable else None,
            "fill_ratio": len(self.memory) / self.memory.capacity if replay_applicable else None,
            "batch_size": BATCH_SIZE if replay_applicable else None,
            "ready": replay_ready,
        }
        optimization = {
            "applicable": True,
            "updates": metrics["updates"],
            "attempted_updates": metrics["attempted_updates"],
            "rejected_updates": metrics["rejected_updates"],
            "rejected_transitions": self.rejected_transition_count,
            "rejected_total": metrics["rejected_updates"] + self.rejected_transition_count,
            "last_rejection": metrics["last_rejection"] or self.last_transition_rejection,
            "decisions": self.decision_count,
            "evaluation_decisions": self.evaluation_decision_count,
            "update_to_decision_ratio": metrics["update_to_decision_ratio"],
            "gradient_applicable": metrics["gradient_applicable"],
            "gradient_norm": metrics["gradient_norm"],
            "clip_threshold": metrics["clip_threshold"],
            "clip_count": metrics["clip_count"],
            "clip_ratio": metrics["clip_ratio"],
            "last_batch_size": metrics["last_batch_size"],
        }
        values = {
            "applicable": True,
            "q_abs_max": max(
                metrics["q_abs_max"],
                max((abs(float(value)) for value in self.last_q_values), default=0.0),
            ),
            "target_q_abs_max": max(
                (abs(float(value)) for value in self.last_target_q_values), default=0.0
            ),
            "td_error_abs_mean": metrics["td_error_abs_mean"],
            "td_error_abs_max": metrics["td_error_abs_max"],
            "predicted_q_mean": self.learning.last_predicted_mean,
            "target_mean": self.learning.last_target_mean,
        }
        neural = {
            "applicable": self.learning.info.family == "deep",
            "parameters_finite": self._neural_finiteness()
            if self.learning.info.family == "deep"
            else None,
        }
        evaluation_episodes = int(self.evaluation_metrics.get("episodes", 0))
        generalization = {
            "status": "evaluated" if evaluation_episodes else "not_evaluated",
            **dict(self.evaluation_metrics),
        }
        terminations = self._termination_diagnostics(game)
        numeric = [
            self.epsilon,
            self.last_reward,
            self.learning.last_loss,
            *self.last_q_values,
            *self.last_target_q_values,
        ]
        for block in (optimization, values):
            numeric.extend(
                value for value in block.values()
                if isinstance(value, Real) and not isinstance(value, bool)
            )
        finite = metrics["finite"] and neural["parameters_finite"] is not False \
            and all(math.isfinite(float(value)) for value in numeric)
        alerts: list[str] = []
        warming = metrics["updates"] == 0 or replay_ready is False
        if not finite:
            alerts.append("non_finite_learning_state")
        if metrics["updates"] == 0:
            alerts.append("optimizer_warming_up")
        if replay_ready is False:
            alerts.append("replay_warming_up")
        if optimization["rejected_total"]:
            alerts.append("updates_rejected")
        if metrics["gradient_applicable"] and metrics["updates"] >= 20 \
                and metrics["clip_ratio"] > 0.25:
            alerts.append("gradient_clipping_high")
        ratio = metrics["update_to_decision_ratio"]
        if self.decision_count >= 32 and ratio < 0.75:
            alerts.append("optimizer_lagging")
        elif self.decision_count >= 32 and ratio > 1.5:
            alerts.append("optimizer_overactive")
        if values["q_abs_max"] > 100.0:
            alerts.append("q_values_large")
        if values["td_error_abs_mean"] > 50.0:
            alerts.append("td_error_large")
        if terminations["total"] >= 10 and terminations["timeout_rate"] > 0.5:
            alerts.append("timeouts_dominate")
        if terminations["total"] >= 10 and terminations["collision_rate"] > 0.85:
            alerts.append("collisions_dominate")
        warning_alerts = [alert for alert in alerts if not alert.endswith("warming_up")]
        status = (
            "critical" if not finite
            else "warning" if warning_alerts
            else "warming_up" if warming
            else "healthy"
        )
        return {
            "status": status,
            "finite": finite,
            "alerts": alerts,
            "replay": replay,
            "optimization": optimization,
            "values": values,
            "neural": neural,
            "generalization": generalization,
            "terminations": terminations,
        }

    def telemetry(
        self,
        state: np.ndarray,
        game: SnakeGameAI | None = None,
        episode_return: float = 0.0,
    ) -> dict:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (11,) or not np.isfinite(state).all():
            raise ValueError("telemetry state must be a finite 11-feature vector")
        if (
            isinstance(episode_return, bool)
            or not isinstance(episode_return, Real)
            or not math.isfinite(float(episode_return))
        ):
            raise ValueError("episode_return must be finite")
        recent = self.memory.tail(24)
        health = self._health(game)
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
            "episode_return": float(episode_return),
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
            "health": health,
            "termination_diagnostics": health["terminations"],
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
        if isinstance(best_score, bool) or not isinstance(best_score, int) or best_score < 0:
            raise ValueError("best_score must be a non-negative integer")
        if isinstance(mean_score, bool) or not isinstance(mean_score, Real) \
                or not math.isfinite(float(mean_score)):
            raise ValueError("mean_score must be finite")
        if evaluation_mean is not None and (
            isinstance(evaluation_mean, bool)
            or not isinstance(evaluation_mean, Real)
            or not math.isfinite(float(evaluation_mean))
        ):
            raise ValueError("evaluation_mean must be finite or None")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("checkpoint reason must be a non-empty string")
        if experiment is not None:
            if not isinstance(experiment, dict):
                raise TypeError("checkpoint experiment must be a dictionary or None")
            try:
                json.dumps(experiment, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError("checkpoint experiment must be finite JSON data") from error
        if isinstance(self.n_games, bool) or not isinstance(self.n_games, int) or self.n_games < 0:
            raise ValueError("n_games must be a non-negative integer")
        version = get_next_model_version(str(MODEL_DIR))
        filename = f"model_v{version:03d}.pth"
        model_path = MODEL_DIR / filename
        try:
            self.learning.save(filename, MODEL_DIR)
            save_model_metadata(
                str(MODEL_DIR),
                version,
                best_score,
                mean_score,
                self.n_games,
                algorithm=self.algorithm,
                evaluation_mean=evaluation_mean,
                checkpoint_reason=reason,
                experiment=experiment,
            )
        except Exception:
            model_path.unlink(missing_ok=True)
            raise
        self.current_version = version
        print(f"Saved {filename}: {reason.replace('_', ' ')}")

    def save_model_new_record(self, best_score: int, mean_score: float) -> None:
        """Backward-compatible record checkpoint entry point."""

        self.save_model_checkpoint(best_score, mean_score, reason="training_record")

    def update_model_mean_score(self, mean_score: float) -> None:
        if isinstance(mean_score, bool) or not isinstance(mean_score, Real) \
                or not math.isfinite(float(mean_score)):
            raise ValueError("mean_score must be finite")
        if self.current_version is not None:
            update_model_metadata(
                str(MODEL_DIR), self.current_version, mean_score, self.n_games
            )

    def _load_model_info(self, model_info) -> bool:
        if not model_info or not isinstance(model_info, tuple) or len(model_info) != 2:
            return False
        model_file, metadata = model_info
        try:
            if not isinstance(model_file, str) or not model_file:
                raise ValueError("checkpoint filename is invalid")
            if not isinstance(metadata, dict):
                raise ValueError("checkpoint metadata must be a dictionary")
            clean_metadata = dict(metadata)
            for key in ("version", "best_score", "games_played"):
                value = clean_metadata.get(key, 0)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    or value < 0
                    or not float(value).is_integer()
                ):
                    raise ValueError(f"checkpoint metadata {key} is invalid")
                clean_metadata[key] = int(value)
            for key in ("mean_score", "evaluation_mean"):
                if key not in clean_metadata or clean_metadata[key] is None:
                    continue
                value = clean_metadata[key]
                if isinstance(value, bool) or not isinstance(value, Real) \
                        or not math.isfinite(float(value)):
                    raise ValueError(f"checkpoint metadata {key} is invalid")
                clean_metadata[key] = float(value)
            self.learning.load(model_file, MODEL_DIR)
            self.loaded_metadata = clean_metadata
            self.current_version = clean_metadata["version"] or None
            self.n_games = clean_metadata["games_played"]
            print(
                f"Loaded {model_file}: score={clean_metadata['best_score']}, "
                f"games={clean_metadata['games_played']}"
            )
            return True
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
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
