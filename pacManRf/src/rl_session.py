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
SPEED_PRESETS: tuple[tuple[str, int], ...] = (
    ("SLOW", 1),
    ("VERY SLOW", 5),
    ("WATCH", 15),
    ("HALF", 30),
    ("NORMAL", 60),
    ("TURBO", 120),
    ("MAX", 240),
)
RENDER_FPS = 60


class SpeedController:
    """Clamp the simulation FPS and expose keyboard-selectable presets."""

    minimum = SPEED_PRESETS[0][1]
    maximum = SPEED_PRESETS[-1][1]

    def __init__(self, initial_speed: int = RENDER_FPS):
        self._value = self._clamp(initial_speed)

    @classmethod
    def _clamp(cls, value: int) -> int:
        return max(cls.minimum, min(cls.maximum, int(value)))

    @property
    def value(self) -> int:
        return self._value

    @property
    def exact_preset_index(self) -> int | None:
        for index, (_, value) in enumerate(SPEED_PRESETS):
            if value == self._value:
                return index
        return None

    @property
    def nearest_preset_index(self) -> int:
        return min(
            range(len(SPEED_PRESETS)),
            key=lambda index: abs(SPEED_PRESETS[index][1] - self._value),
        )

    @property
    def label(self) -> str:
        index = self.exact_preset_index
        return SPEED_PRESETS[index][0] if index is not None else "CUSTOM"

    def select(self, index: int) -> int:
        if not 0 <= index < len(SPEED_PRESETS):
            raise ValueError("speed preset index is out of range")
        self._value = SPEED_PRESETS[index][1]
        return self._value

    def step(self, direction: int) -> int:
        """Move to the next slower/faster preset and stop at the endpoints."""

        if direction == 0:
            return self._value
        if direction < 0:
            for index in range(len(SPEED_PRESETS) - 1, -1, -1):
                if SPEED_PRESETS[index][1] < self._value:
                    return self.select(index)
            return self.select(0)
        for index in range(len(SPEED_PRESETS)):
            if SPEED_PRESETS[index][1] > self._value:
                return self.select(index)
        return self.select(len(SPEED_PRESETS) - 1)

    def handle_key(self, key: int) -> bool:
        number_keys = (
            (pygame.K_1, pygame.K_KP1),
            (pygame.K_2, pygame.K_KP2),
            (pygame.K_3, pygame.K_KP3),
            (pygame.K_4, pygame.K_KP4),
            (pygame.K_5, pygame.K_KP5),
            (pygame.K_6, pygame.K_KP6),
            (pygame.K_7, pygame.K_KP7),
        )
        for index, keys in enumerate(number_keys):
            if key in keys:
                self.select(index)
                return True
        if key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
            self.step(-1)
            return True
        if key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS):
            self.step(1)
            return True
        if key == pygame.K_HOME:
            self.select(0)
            return True
        if key == pygame.K_END:
            self.select(len(SPEED_PRESETS) - 1)
            return True
        return False

    def telemetry(self) -> dict[str, Any]:
        return {
            "simulation_fps_target": self.value,
            "speed_target_fps": self.value,
            "speed_label": self.label,
            "speed_preset_index": self.nearest_preset_index,
            "speed_preset_count": len(SPEED_PRESETS),
            "speed_preset_values": [value for _, value in SPEED_PRESETS],
        }


class DecisionScheduler:
    """Convert elapsed render time into bounded fixed-rate simulation frames."""

    def __init__(
        self,
        *,
        max_steps_per_frame: int = 16,
        max_elapsed_seconds: float = 0.25,
    ):
        if max_steps_per_frame < 1:
            raise ValueError("max_steps_per_frame must be positive")
        if max_elapsed_seconds <= 0:
            raise ValueError("max_elapsed_seconds must be positive")
        self.max_steps_per_frame = int(max_steps_per_frame)
        self.max_elapsed_seconds = float(max_elapsed_seconds)
        self._fractional_steps = 0.0

    def reset(self) -> None:
        self._fractional_steps = 0.0

    def frames_for_render(
        self,
        elapsed_seconds: float,
        simulation_frames_per_second: int,
        *,
        paused: bool = False,
        single_step: bool = False,
    ) -> int:
        if paused:
            self.reset()
            return int(single_step)
        elapsed = float(elapsed_seconds)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        rate = max(1, int(simulation_frames_per_second))
        self._fractional_steps += min(elapsed, self.max_elapsed_seconds) * rate
        whole_steps = int(self._fractional_steps)
        self._fractional_steps -= whole_steps
        return min(whole_steps, self.max_steps_per_frame)

    def steps_for_frame(
        self,
        elapsed_seconds: float,
        decisions_per_second: int,
        *,
        paused: bool = False,
        single_step: bool = False,
    ) -> int:
        """Backward-compatible alias; values now represent simulation FPS."""

        return self.frames_for_render(
            elapsed_seconds,
            decisions_per_second,
            paused=paused,
            single_step=single_step,
        )


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
    terminate_on_life_loss: bool = False
    auto_advance_levels: bool = True
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
            auto_advance_levels=self.config.auto_advance_levels,
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
        self._recent_termination_reasons: deque[str] = deque(maxlen=64)
        self.history: dict[str, deque] = {
            "rewards": deque(maxlen=500),
            "losses": deque(maxlen=500),
            "scores": deque(maxlen=500),
            "epsilons": deque(maxlen=500),
            "episode_returns": deque(maxlen=500),
            "levels": deque(maxlen=500),
            "projectiles": deque(maxlen=500),
            "slow_fraction": deque(maxlen=500),
        }
        self.state = self.env.observation
        self.pending_action = self._select_action(self.state)
        self._transition_state: np.ndarray | None = None
        self._transition_action: int | None = None

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
            advance_schedule=self.config.training,
        )

    def step(self) -> dict[str, Any]:
        """Apply the pending decision, learn once, and prepare the next one."""

        self._begin_transition()
        result = None
        while result is None:
            result = self.env.advance_step_frame()
        return self._complete_transition(result)

    def advance_simulation_frame(self) -> dict[str, Any] | None:
        """Advance exactly one 60 Hz game frame.

        A result is returned only at the next grid decision, when replay and
        learning are updated. This keeps visual pacing independent from the
        variable number of physics frames consumed by an agent action.
        """

        self._begin_transition()
        result = self.env.advance_step_frame()
        if result is None:
            return None
        return self._complete_transition(result)

    def _begin_transition(self) -> None:
        if self.env.step_in_progress:
            if self._transition_state is None or self._transition_action is None:
                raise RuntimeError("environment step has no matching session transition")
            return
        self._transition_state = self.state.copy()
        self._transition_action = int(self.pending_action)
        self.env.begin_step(self._transition_action)

    def _complete_transition(
        self,
        result: tuple[np.ndarray, float, bool, dict],
    ) -> dict[str, Any]:
        state = self._transition_state
        action = self._transition_action
        if state is None or action is None:
            raise RuntimeError("cannot complete a transition that was not started")
        self._transition_state = None
        self._transition_action = None
        next_state, reward, done, info = result
        next_legal_action_mask = np.asarray(next_state[:4] > 0.5, dtype=np.bool_)
        if not next_legal_action_mask.any():
            next_legal_action_mask[:] = True
        if self.config.training:
            metrics = self.agent.observe(
                state,
                action,
                reward,
                next_state,
                done,
                next_legal_action_mask,
            )
        else:
            self.agent.last_reward = float(reward)
            self.agent.episode_return += float(reward)
            if done:
                self.agent.episodes += 1
            metrics = None

        self.history["rewards"].append(float(reward))
        if metrics is not None:
            self.history["losses"].append(float(metrics.loss))
        self.history["scores"].append(float(self.env.game.score))
        self.history["epsilons"].append(float(self.agent.epsilon))
        projectile_data = self.env.game.projectile_telemetry()
        self.history["levels"].append(float(self.env.game.level))
        self.history["projectiles"].append(float(projectile_data["active_count"]))
        self.history["slow_fraction"].append(float(projectile_data["slow_fraction"]))
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
            self._recent_termination_reasons.append(
                str(info.get("termination_reason") or "unknown")
            )
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
        self._transition_state = None
        self._transition_action = None
        self.last_info = dict(self.env.last_info)
        self.pending_action = self._select_action(self.state)

    def telemetry(self, *, max_neurons_per_layer: int = 11) -> dict[str, Any]:
        data = self.agent.telemetry(
            self.state,
            max_neurons_per_layer=max_neurons_per_layer,
        )
        projectile_data = self.env.game.projectile_telemetry()
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
                "projectiles": projectile_data,
                "projectiles_active": projectile_data["active_count"],
                "player_slowed": projectile_data["player_slowed"],
                "slow_fraction": projectile_data["slow_fraction"],
                "slow_timer": projectile_data["slow_timer"],
                "projectile_shots_fired": projectile_data["shots_fired"],
                "fireball_hits": projectile_data["fireball_hits"],
                "freeze_ball_hits": projectile_data["freeze_ball_hits"],
                "loaded_checkpoint": self.loaded_checkpoint,
                "history": self.history_snapshot(),
            }
        )
        health = data.get("health")
        if isinstance(health, dict):
            reason_counts: dict[str, int] = {}
            for reason in self._recent_termination_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            reward_components = self.last_info.get("reward_components")
            health["termination"] = {
                "last_reason": (
                    self.last_episode.get("reason")
                    if self.last_episode is not None
                    else self.last_info.get("termination_reason")
                ),
                "recent_count": len(self._recent_termination_reasons),
                "recent_counts": reason_counts,
            }
            health["environment"] = {
                "last_reward": float(self.agent.last_reward),
                "last_reward_components": (
                    dict(reward_components) if isinstance(reward_components, dict) else {}
                ),
                "action_blocked": bool(self.last_info.get("action_blocked", False)),
                "stalled": bool(self.last_info.get("stalled", False)),
                "life_lost": bool(self.last_info.get("life_lost", False)),
                "level_cleared": bool(self.last_info.get("level_cleared", False)),
            }
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
    speed: int = RENDER_FPS,
    initial_tab: str = "GAME",
    save_on_exit: bool = True,
) -> None:
    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Pacman DQN Observatory")
    clock = pygame.time.Clock()
    observatory = PacmanObservatory(initial_tab=initial_tab)
    scheduler = DecisionScheduler()
    paused = False
    single_step = False
    running = True
    speed_controller = SpeedController(speed)
    rate_samples: deque[tuple[float, int]] = deque(maxlen=RENDER_FPS)

    try:
        while running:
            elapsed_seconds = clock.tick(RENDER_FPS) / 1_000.0
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    speed_preset = observatory.speed_preset_at(event.pos)
                    if speed_preset is not None:
                        previous_speed = speed_controller.value
                        speed_controller.select(speed_preset)
                        if speed_controller.value != previous_speed:
                            scheduler.reset()
                            rate_samples.clear()
                        continue
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
                    else:
                        previous_speed = speed_controller.value
                        speed_key = speed_controller.handle_key(event.key)
                        if speed_key and speed_controller.value != previous_speed:
                            scheduler.reset()
                            rate_samples.clear()
                        if speed_key:
                            continue
                    if event.key == pygame.K_r:
                        session.reset_environment()
                        scheduler.reset()
                        rate_samples.clear()
                    elif event.key == pygame.K_s:
                        session.save_checkpoint()

            simulation_frames = scheduler.frames_for_render(
                elapsed_seconds,
                speed_controller.value,
                paused=paused,
            )
            executed_frames = 0
            if running and paused and single_step:
                session.step()
            elif running:
                for _ in range(simulation_frames):
                    session.advance_simulation_frame()
                    executed_frames += 1
            single_step = False
            rate_samples.append((elapsed_seconds, executed_frames))

            telemetry = session.telemetry()
            telemetry["paused"] = paused
            telemetry["render_fps"] = clock.get_fps()
            telemetry["render_fps_target"] = RENDER_FPS
            sampled_seconds = sum(sample[0] for sample in rate_samples)
            telemetry["simulation_fps_actual"] = (
                sum(sample[1] for sample in rate_samples) / sampled_seconds
                if sampled_seconds > 0
                else 0.0
            )
            telemetry.update(speed_controller.telemetry())
            observatory.render(
                window,
                telemetry,
                history=session.history_snapshot(),
                game_surface=session.render_game(),
            )
            pygame.display.flip()
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
