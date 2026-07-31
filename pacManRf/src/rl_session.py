"""Integrated Pacman reinforcement-learning session and observability runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from pacManRf.src.game.constants import REPO_ROOT
from pacManRf.src.game.pacman_env import ACTION_LABELS, OBSERVATION_LABELS, PacmanEnv
from pacManRf.src.ml import DQNConfig, PacmanDQNAgent
from pacManRf.src.visualization import PacmanObservatory


WINDOW_SIZE = (1120, 720)
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "pacManRf" / "models" / "checkpoints"


@dataclass(slots=True)
class SessionConfig:
    algorithm: str = "double_dqn"
    seed: int = 7
    training: bool = True
    fresh: bool = False
    checkpoint: Path | None = None
    hidden_sizes: tuple[int, ...] = (256, 128)
    batch_size: int = 64
    replay_capacity: int = 100_000
    replay_warmup: int = 64
    target_update_interval: int = 250
    epsilon_decay_steps: int = 25_000
    terminate_on_life_loss: bool = True
    max_episode_steps: int = 2_000
    save_replay: bool = False


class PacmanRLSession:
    """Own one environment, agent, pending decision, and observable history."""

    def __init__(self, config: SessionConfig | None = None):
        self.config = config or SessionConfig()
        self.env = PacmanEnv(
            seed=self.config.seed,
            render=False,
            terminate_on_life_loss=self.config.terminate_on_life_loss,
            auto_advance_levels=False,
            max_episode_steps=self.config.max_episode_steps,
        )
        learning_config = DQNConfig(
            observation_size=self.env.observation_size,
            action_size=self.env.action_size,
            hidden_sizes=self.config.hidden_sizes,
            action_labels=ACTION_LABELS,
            algorithm=self.config.algorithm,
            batch_size=self.config.batch_size,
            replay_capacity=self.config.replay_capacity,
            replay_warmup=self.config.replay_warmup,
            target_update_interval=self.config.target_update_interval,
            epsilon_decay_steps=self.config.epsilon_decay_steps,
            seed=self.config.seed,
        )
        self.agent = PacmanDQNAgent(learning_config)
        self.checkpoint_path = Path(
            self.config.checkpoint
            or DEFAULT_CHECKPOINT_DIR / f"pacman_{learning_config.algorithm}_latest.pth"
        )
        self.loaded_checkpoint = False
        if not self.config.fresh and self.checkpoint_path.is_file():
            self.agent.load_checkpoint(self.checkpoint_path, load_replay=True)
            self.loaded_checkpoint = True

        self.best_score = int(self.agent.checkpoint_metadata.get("best_score", 0))
        self.completed_episodes = 0
        self.last_episode: dict[str, Any] | None = None
        self.last_info = dict(self.env.last_info)
        self.history: dict[str, deque] = {
            "rewards": deque(maxlen=500),
            "losses": deque(maxlen=500),
            "scores": deque(maxlen=500),
            "epsilons": deque(maxlen=500),
            "episode_returns": deque(maxlen=500),
        }
        self.state = self.env.observation
        self.pending_action = self._select_action(self.state)

    @property
    def legal_action_mask(self) -> np.ndarray:
        return np.asarray(self.state[:4] > 0.5, dtype=np.bool_)

    def _select_action(self, state: np.ndarray) -> int:
        mask = np.asarray(state[:4] > 0.5, dtype=np.bool_)
        # Reverse should always be available on a valid corridor, but keep a
        # defensive fallback for terminal or custom maze states.
        if not mask.any():
            mask[:] = True
        return self.agent.select_action(
            state,
            explore=self.config.training,
            legal_action_mask=mask,
        )

    def step(self) -> dict[str, Any]:
        """Apply the pending decision, learn once, and prepare the next one."""
        state = self.state
        action = self.pending_action
        next_state, reward, done, info = self.env.step(action)
        if self.config.training:
            metrics = self.agent.observe(state, action, reward, next_state, done)
        else:
            self.agent.last_reward = float(reward)
            self.agent.episode_return += float(reward)
            metrics = None

        self.history["rewards"].append(float(reward))
        self.history["losses"].append(float(self.agent.trainer.last_metrics.loss))
        self.history["scores"].append(float(self.env.game.score))
        self.history["epsilons"].append(float(self.agent.epsilon))
        self.last_info = dict(info)
        episode_finished = False

        if done:
            episode_finished = True
            episode_return = float(self.agent.episode_return)
            score = int(self.env.game.score)
            self.best_score = max(self.best_score, score)
            self.completed_episodes += 1
            self.history["episode_returns"].append(episode_return)
            self.last_episode = {
                "episode": self.agent.episodes,
                "score": score,
                "return": episode_return,
                "steps": self.env.episode_steps,
                "reason": info.get("termination_reason"),
            }
            self.agent.reset_episode()
            self.state = self.env.reset()
        else:
            self.state = next_state

        self.pending_action = self._select_action(self.state)
        return {
            "done": done,
            "episode_finished": episode_finished,
            "reward": reward,
            "info": info,
            "metrics": metrics,
            "last_episode": dict(self.last_episode) if self.last_episode else None,
        }

    def reset_environment(self) -> None:
        self.agent.reset_episode()
        self.state = self.env.reset()
        self.last_info = dict(self.env.last_info)
        self.pending_action = self._select_action(self.state)

    def telemetry(self, *, max_neurons_per_layer: int = 14) -> dict[str, Any]:
        data = self.agent.telemetry(
            self.state,
            max_neurons_per_layer=max_neurons_per_layer,
        )
        data.update(
            {
                "score": self.env.game.score,
                "best_score": self.best_score,
                "episode": self.agent.episodes,
                "reward": self.agent.last_reward,
                "replay_size": len(self.agent.memory),
                "replay_capacity": self.agent.memory.capacity,
                "action_labels": list(ACTION_LABELS),
                "chosen_action": self.pending_action,
                "action_index": self.pending_action,
                "observation_labels": list(OBSERVATION_LABELS),
                "observation": self.env.observation_dict(),
                "environment": dict(self.last_info),
                "lives": self.env.game.lives,
                "level": self.env.game.level,
                "pellets": self.env.game._count_dots(),
                "loaded_checkpoint": self.loaded_checkpoint,
                "history": self.history_snapshot(),
            }
        )
        return data

    def history_snapshot(self) -> dict[str, list[float]]:
        return {name: list(values) for name, values in self.history.items()}

    def render_game(self) -> pygame.Surface:
        return self.env.render()

    def save_checkpoint(self, *, include_replay: bool | None = None) -> Path:
        include = self.config.save_replay if include_replay is None else include_replay
        return self.agent.save_checkpoint(
            self.checkpoint_path,
            include_replay=include,
            metadata={
                "best_score": self.best_score,
                "algorithm": self.agent.config.algorithm,
                "observation_size": self.env.observation_size,
                "action_labels": list(ACTION_LABELS),
            },
        )

    def close(self) -> None:
        self.env.close()


def run_visual_session(
    session: PacmanRLSession,
    *,
    speed: int = 30,
    initial_tab: str = "GAME",
    save_on_exit: bool = True,
) -> None:
    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Pacman DQN Observatory")
    clock = pygame.time.Clock()
    observatory = PacmanObservatory(initial_tab=initial_tab)
    paused = False
    single_step = False
    running = True
    decisions_per_second = max(1, int(speed))

    try:
        while running:
            for event in pygame.event.get():
                if observatory.handle_event(event):
                    continue
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key in (pygame.K_n, pygame.K_PERIOD) and paused:
                        single_step = True
                    elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                        decisions_per_second = max(1, decisions_per_second // 2)
                    elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS):
                        decisions_per_second = min(240, decisions_per_second * 2)
                    elif event.key == pygame.K_r:
                        session.reset_environment()
                    elif event.key == pygame.K_s:
                        session.save_checkpoint()

            if (not paused or single_step) and running:
                session.step()
                single_step = False

            telemetry = session.telemetry()
            telemetry["paused"] = paused
            telemetry["decisions_per_second"] = decisions_per_second
            observatory.render(
                window,
                telemetry,
                history=session.history_snapshot(),
                game_surface=session.render_game(),
            )
            pygame.display.flip()
            clock.tick(decisions_per_second)
    finally:
        if save_on_exit and session.config.training:
            session.save_checkpoint()
        session.close()
        pygame.quit()


def run_headless_session(
    session: PacmanRLSession,
    *,
    episodes: int = 1,
    max_steps: int | None = None,
    save_on_exit: bool = True,
) -> list[dict[str, Any]]:
    """Train without a window and return completed episode summaries."""
    target_episodes = max(0, int(episodes))
    step_limit = None if max_steps is None else max(1, int(max_steps))
    start_episodes = session.completed_episodes
    steps = 0
    summaries: list[dict[str, Any]] = []
    try:
        while target_episodes == 0 or session.completed_episodes - start_episodes < target_episodes:
            result = session.step()
            steps += 1
            if result["episode_finished"] and result["last_episode"]:
                summary = result["last_episode"]
                summaries.append(summary)
                print(
                    f"episode={summary['episode']:4d} score={summary['score']:5d} "
                    f"return={summary['return']:+8.2f} steps={summary['steps']:4d} "
                    f"reason={summary['reason']} replay={len(session.agent.memory):6d} "
                    f"loss={session.agent.trainer.last_metrics.loss:.5f}"
                )
            if step_limit is not None and steps >= step_limit:
                break
    finally:
        if save_on_exit and session.config.training:
            session.save_checkpoint()
        session.close()
    return summaries
