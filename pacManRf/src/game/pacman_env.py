"""Deterministic reinforcement-learning adapter for :mod:`pacmanGame`.

The pygame game advances in pixels and frames, while a value-based agent needs
stable decision boundaries.  ``PacmanEnv`` turns one relative action into one
grid-cell transition and exposes a small, named observation vector.  Rendering
is optional and is deliberately kept out of ``step`` so training speed and
results do not depend on the host machine's frame rate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from numbers import Integral
from typing import Sequence

import numpy as np

from .constants import (
    STARTING_LIVES,
    Direction,
    GamePhase,
    GameStatus,
    GhostMode,
)
from .pacmanGame import PacmanGame


class RelativeAction(IntEnum):
    """Actions relative to Pacman's current heading."""

    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2
    REVERSE = 3


ACTION_LABELS = ("STRAIGHT", "TURN RIGHT", "TURN LEFT", "REVERSE")

# Each directional group uses the same egocentric order as ACTION_LABELS.  The
# power-of-two input width is also convenient for the network observatory.
OBSERVATION_LABELS = (
    "path ahead", "path right", "path left", "path reverse",
    "pellet ahead", "pellet right", "pellet left", "pellet reverse",
    "power pellet ahead", "power pellet right", "power pellet left", "power pellet reverse",
    "threat ahead", "threat right", "threat left", "threat reverse",
    "edible ghost ahead", "edible ghost right", "edible ghost left", "edible ghost reverse",
    "heading up", "heading right", "heading down", "heading left",
    "position x", "position y", "frightened time", "pellets remaining",
    "lives", "level", "chase mode", "ghosts released",
)
VISION_LABELS = OBSERVATION_LABELS
OBSERVATION_GROUPS = {
    "paths": slice(0, 4),
    "pellets": slice(4, 8),
    "power_pellets": slice(8, 12),
    "threats": slice(12, 16),
    "edible_ghosts": slice(16, 20),
    "heading": slice(20, 24),
    "context": slice(24, 32),
}

_CLOCKWISE = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)


@dataclass(frozen=True)
class RewardConfig:
    """Reward terms in agent-friendly units (a normal pellet is ``+1``)."""

    score_scale: float = 0.1
    step: float = -0.01
    blocked_action: float = -0.10
    closer_to_pellet: float = 0.05
    farther_from_pellet: float = -0.02
    freeze_hit: float = -5.0
    life_lost: float = -25.0
    game_over: float = -25.0
    level_cleared: float = 500.0


@dataclass
class _PendingStep:
    """State captured while one grid decision advances frame by frame."""

    action_index: int
    requested_direction: Direction
    start_grid: tuple[int, int]
    start_level: int
    score_before: int
    lives_before: int
    dots_before: int
    pellets_before: int
    power_before: int
    eaten_before: tuple[bool, ...]
    shots_before: int
    fireball_hits_before: int
    freeze_ball_hits_before: int
    nearest_pellet_before: int | None
    action_blocked: bool
    internal_frames: int = 0
    stalled: bool = False
    projectile_events: list = field(default_factory=list)


class PacmanEnv:
    """Grid-decision RL environment backed by :class:`PacmanGame`.

    ``step`` accepts either an integer action or a one-hot vector of length
    four and returns ``(observation, reward, terminated, info)``.  By default a
    life loss resets the round but not the episode, and clearing a maze advances
    to the next level.  Both policies are configurable for episodic experiments.
    """

    action_size = len(ACTION_LABELS)
    observation_size = len(OBSERVATION_LABELS)
    action_labels = ACTION_LABELS
    observation_labels = OBSERVATION_LABELS

    def __init__(
        self,
        *,
        seed: int = 7,
        render: bool = False,
        frame_dt: float = 1 / 60,
        terminate_on_life_loss: bool = False,
        auto_advance_levels: bool = True,
        max_episode_steps: int | None = None,
        rewards: RewardConfig | None = None,
    ) -> None:
        if not 0 < frame_dt <= 0.05:
            raise ValueError("frame_dt must be in the interval (0, 0.05]")
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive or None")

        self.seed = int(seed)
        self.frame_dt = float(frame_dt)
        self.terminate_on_life_loss = bool(terminate_on_life_loss)
        self.auto_advance_levels = bool(auto_advance_levels)
        self.max_episode_steps = max_episode_steps
        self.rewards = rewards or RewardConfig()
        self.game = PacmanGame(
            render=render,
            seed=self.seed,
            auto_advance_on_clear=False,
        )
        self.render_enabled = render

        self.episode_steps = 0
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.last_action_index = int(RelativeAction.STRAIGHT)
        self.last_info: dict = {}
        self._terminated = False
        self._pending_step: _PendingStep | None = None
        self._observation = np.zeros(self.observation_size, dtype=np.float32)
        self.reset(seed=self.seed)

    @property
    def observation(self) -> np.ndarray:
        """A defensive copy of the observation currently shown to the agent."""

        return self._observation.copy()

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def step_in_progress(self) -> bool:
        """Whether an action is currently between two grid decisions."""

        return self._pending_step is not None

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        """Start a fresh run and return its active-round observation.

        Passing ``seed`` restarts the environment's pseudo-random sequence.
        Omitting it lets the seeded sequence continue across episodes, which is
        deterministic while still producing different frightened-ghost paths.
        """

        if seed is not None:
            self.seed = int(seed)
            self.game.rng.seed(self.seed)
        self.game.running = True
        self.game.restart()
        self.episode_steps = 0
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.last_action_index = int(RelativeAction.STRAIGHT)
        self._terminated = False
        self._pending_step = None
        self._fast_forward_to_active()
        self._observation = self._get_observation()
        self.last_info = self._base_info()
        self.last_info.update({"reset": True, "termination_reason": None})
        return self.observation

    def step(
        self, action: int | Sequence[int | float]
    ) -> tuple[np.ndarray, float, bool, dict]:
        """Apply one relative action and advance to the next grid decision."""

        if self.step_in_progress:
            raise RuntimeError("step() called while an incremental step is in progress")
        self.begin_step(action)
        result = None
        while result is None:
            result = self.advance_step_frame()
        return result

    def begin_step(self, action: int | Sequence[int | float]) -> None:
        """Start one action without advancing its fixed-rate physics frames.

        Visual runtimes use this together with :meth:`advance_step_frame` so
        rendering can remain smooth. The regular :meth:`step` API stays atomic
        for headless training.
        """

        if self._terminated:
            raise RuntimeError("step() called on a terminated episode; call reset() first")
        if self.step_in_progress:
            raise RuntimeError("begin_step() called while another step is in progress")

        action_index = self._coerce_action(action)
        relative_directions = self._relative_directions(self.game.player.direction)
        requested_direction = relative_directions[action_index]
        start_grid = self.game.player.grid
        self._pending_step = _PendingStep(
            action_index=action_index,
            requested_direction=requested_direction,
            start_grid=start_grid,
            start_level=self.game.level,
            score_before=self.game.score,
            lives_before=self.game.lives,
            dots_before=self.game._count_dots(),
            pellets_before=self._count_cell("."),
            power_before=self._count_cell("o"),
            eaten_before=tuple(ghost.eaten for ghost in self.game.ghosts),
            shots_before=self.game.projectile_shots_fired,
            fireball_hits_before=self.game.fireball_hits,
            freeze_ball_hits_before=self.game.freeze_ball_hits,
            nearest_pellet_before=self._nearest_distance(self._dot_cells()),
            action_blocked=not self.game._can_move(start_grid, requested_direction),
        )
        self.game.next_direction = requested_direction

    def advance_step_frame(self) -> tuple[np.ndarray, float, bool, dict] | None:
        """Advance one fixed physics frame, completing the action if ready."""

        pending = self._pending_step
        if pending is None:
            raise RuntimeError("advance_step_frame() called before begin_step()")

        pending.internal_frames += 1
        self.game._update(self.frame_dt)
        pending.projectile_events.extend(self.game.last_projectile_events)

        finished = (
            self.game.lives < pending.lives_before
            or self.game.status in (GameStatus.LOST, GameStatus.WON)
            or not self.game.running
            or self.game.phase == GamePhase.CLEARING
            or self.game.player.grid != pending.start_grid
        )
        if (
            not finished
            and self.game.phase == GamePhase.ACTIVE
            and self.game.player.target is None
        ):
            pending.stalled = True
            finished = True
        if pending.internal_frames > 240:
            self._pending_step = None
            raise RuntimeError("Pacman transition did not reach a decision boundary")
        if not finished:
            return None
        return self._finish_step(pending)

    def _finish_step(
        self, pending: _PendingStep
    ) -> tuple[np.ndarray, float, bool, dict]:
        """Finalize reward and observations at a completed grid decision."""

        self._pending_step = None

        # Capture events before a life or level transition resets transient game
        # state.  This makes telemetry faithful to the experience put in replay.
        dots_after_event = self.game._count_dots()
        pellets_after_event = self._count_cell(".")
        power_after_event = self._count_cell("o")
        eaten_after_event = tuple(ghost.eaten for ghost in self.game.ghosts)
        score_delta = self.game.score - pending.score_before
        pellets_eaten = max(0, pending.pellets_before - pellets_after_event)
        power_pellets_eaten = max(0, pending.power_before - power_after_event)
        ghosts_eaten = sum(
            int(not was_eaten and is_eaten)
            for was_eaten, is_eaten in zip(pending.eaten_before, eaten_after_event)
        )
        projectiles_fired = self.game.projectile_shots_fired - pending.shots_before
        fireball_hits = self.game.fireball_hits - pending.fireball_hits_before
        freeze_ball_hits = self.game.freeze_ball_hits - pending.freeze_ball_hits_before
        life_lost = self.game.lives < pending.lives_before
        level_cleared = (
            self.game.phase == GamePhase.CLEARING
            or self.game.status == GameStatus.WON
            or (pending.dots_before > 0 and dots_after_event == 0)
        )
        game_over = self.game.status == GameStatus.LOST or self.game.lives <= 0

        components = {
            "step": self.rewards.step,
            "score": score_delta * self.rewards.score_scale,
            "blocked_action": self.rewards.blocked_action if pending.action_blocked else 0.0,
            "pellet_progress": 0.0,
            "freeze_hit": self.rewards.freeze_hit * freeze_ball_hits,
            "life_lost": self.rewards.life_lost if life_lost else 0.0,
            "game_over": self.rewards.game_over if game_over else 0.0,
            "level_cleared": self.rewards.level_cleared if level_cleared else 0.0,
        }

        # Do not compare nearest-pellet distances after consuming one: the old
        # target disappeared, so that delta would punish successful collection.
        if not pellets_eaten and not power_pellets_eaten:
            nearest_pellet_after = self._nearest_distance(self._dot_cells())
            if pending.nearest_pellet_before is not None and nearest_pellet_after is not None:
                if nearest_pellet_after < pending.nearest_pellet_before:
                    components["pellet_progress"] = self.rewards.closer_to_pellet
                elif nearest_pellet_after > pending.nearest_pellet_before:
                    components["pellet_progress"] = self.rewards.farther_from_pellet

        reward = float(sum(components.values()))
        self.episode_steps += 1
        self.episode_return += reward
        self.last_reward = reward
        self.last_action_index = pending.action_index

        termination_reason: str | None = None
        terminated = False
        if game_over:
            terminated = True
            termination_reason = "game_over"
        elif not self.game.running:
            terminated = True
            termination_reason = "quit"
        elif life_lost and self.terminate_on_life_loss:
            terminated = True
            termination_reason = "life_lost"
        elif level_cleared and not self.auto_advance_levels:
            self.game.status = GameStatus.WON
            terminated = True
            termination_reason = "level_cleared"
        elif self.max_episode_steps is not None and self.episode_steps >= self.max_episode_steps:
            terminated = True
            termination_reason = "time_limit"

        if not terminated:
            if level_cleared:
                self.game.next_level()
                self._fast_forward_to_active()
            elif life_lost:
                self._fast_forward_to_active()

        self._terminated = terminated
        self._observation = self._get_observation()
        info = self._base_info()
        info.update(
            {
                "reset": False,
                "action_index": pending.action_index,
                "action_label": ACTION_LABELS[pending.action_index],
                "requested_direction": pending.requested_direction.name,
                "applied_direction": self.game.player.direction.name,
                "action_blocked": pending.action_blocked,
                "stalled": pending.stalled,
                "internal_frames": pending.internal_frames,
                "score_delta": score_delta,
                "pellets_eaten": pellets_eaten,
                "power_pellets_eaten": power_pellets_eaten,
                "ghosts_eaten": ghosts_eaten,
                "projectiles_fired": projectiles_fired,
                "projectile_shots_fired": projectiles_fired,
                "fireball_hits": fireball_hits,
                "freeze_ball_hits": freeze_ball_hits,
                "projectile_events": [
                    {
                        "id": event.projectile_id,
                        "owner": event.owner,
                        "kind": event.kind.value,
                        "reason": event.reason.value,
                        "cell": event.cell,
                        "position_tiles": event.position_tiles,
                        "damage": event.damage,
                        "slow_fraction": event.slow_fraction,
                    }
                    for event in pending.projectile_events
                ],
                "pacman_slowed": self.game.player_slow.active,
                "slow_remaining_seconds": self.game.player_slow.remaining_seconds,
                "life_lost": life_lost,
                "level_cleared": level_cleared,
                "cleared_level": pending.start_level if level_cleared else None,
                "terminated": terminated,
                "termination_reason": termination_reason,
                "reward": reward,
                "reward_components": components,
            }
        )
        self.last_info = info
        return self.observation, reward, terminated, dict(info)

    def observation_dict(self) -> dict[str, float]:
        """Return the current vector keyed by stable human-readable labels."""

        return {
            label: float(value)
            for label, value in zip(OBSERVATION_LABELS, self._observation)
        }

    def render(self):
        """Render the current state on the game's surface and return it."""

        self.game._render(self.frame_dt)
        return self.game.display

    def close(self) -> None:
        """Mark this environment closed without shutting down global pygame."""

        self.game.running = False

    @staticmethod
    def _relative_directions(heading: Direction) -> tuple[Direction, ...]:
        index = _CLOCKWISE.index(heading)
        return (
            heading,
            _CLOCKWISE[(index + 1) % 4],
            _CLOCKWISE[(index - 1) % 4],
            heading.opposite,
        )

    @staticmethod
    def _coerce_action(action: int | Sequence[int | float]) -> int:
        if isinstance(action, Integral):
            index = int(action)
        else:
            values = np.asarray(action, dtype=np.float32)
            if values.shape != (len(ACTION_LABELS),):
                raise ValueError(f"action vector must have shape ({len(ACTION_LABELS)},)")
            if not np.all(np.isfinite(values)):
                raise ValueError("action vector must contain finite values")
            ones = np.isclose(values, 1.0)
            zeros = np.isclose(values, 0.0)
            if int(np.count_nonzero(ones)) != 1 or not bool(np.all(ones | zeros)):
                raise ValueError("action vector must be one-hot")
            index = int(np.argmax(values))
        if not 0 <= index < len(ACTION_LABELS):
            raise ValueError(f"action index must be between 0 and {len(ACTION_LABELS) - 1}")
        return index

    def _fast_forward_to_active(self) -> None:
        frames = 0
        while (
            self.game.running
            and self.game.status == GameStatus.PLAYING
            and self.game.phase in (GamePhase.READY, GamePhase.DYING)
        ):
            frames += 1
            self.game._update(self.frame_dt)
            if frames > 600:
                raise RuntimeError("Pacman round transition did not become active")

    def _get_observation(self) -> np.ndarray:
        directions = self._relative_directions(self.game.player.direction)
        dots = self._dot_cells()
        power_pellets = self._cells_containing("o")
        ghost_threats = {
            ghost.grid
            for ghost in self.game.ghosts
            if ghost.released and not ghost.eaten and self.game.frightened_timer <= 0
        }
        projectile_threats = self.game.projectile_threat_cells()
        edible_ghosts = {
            ghost.grid
            for ghost in self.game.ghosts
            if ghost.released and not ghost.eaten and self.game.frightened_timer > 0
        }

        values: list[float] = []
        values.extend(float(self.game._can_move(self.game.player.grid, direction)) for direction in directions)
        values.extend(self._directional_proximity(directions, dots, horizon=40))
        values.extend(self._directional_proximity(directions, power_pellets, horizon=40))
        ghost_proximity = self._directional_proximity(directions, ghost_threats, horizon=8)
        projectile_proximity = self._directional_proximity(
            directions,
            projectile_threats,
            horizon=15,
        )
        values.extend(
            max(ghost_value, projectile_value)
            for ghost_value, projectile_value in zip(ghost_proximity, projectile_proximity)
        )
        values.extend(self._directional_proximity(directions, edible_ghosts, horizon=12))
        values.extend(
            float(self.game.player.direction == direction)
            for direction in (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)
        )
        values.extend(
            (
                self.game.player.grid_x / max(1, self.game.cols - 1),
                self.game.player.grid_y / max(1, self.game.rows - 1),
                min(1.0, self.game.frightened_timer / self.game.frightened_duration),
                self.game._count_dots() / max(1, self.game.total_dots),
                min(1.0, self.game.lives / (STARTING_LIVES + 1)),
                min(1.0, self.game.level / 10.0),
                float(self.game.ghost_mode == GhostMode.CHASE),
                sum(int(ghost.released) for ghost in self.game.ghosts) / len(self.game.ghosts),
            )
        )
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.observation_size,):
            raise RuntimeError(f"observation contract changed unexpectedly: {observation.shape}")
        return observation

    def _directional_proximity(
        self,
        directions: Sequence[Direction],
        targets: set[tuple[int, int]],
        *,
        horizon: int,
    ) -> list[float]:
        proximities = []
        for direction in directions:
            distance = self._branch_distance(direction, targets, horizon=horizon)
            if distance is None:
                proximities.append(0.0)
            else:
                proximities.append((horizon - distance + 1) / horizon)
        return proximities

    def _branch_distance(
        self,
        first_direction: Direction,
        targets: set[tuple[int, int]],
        *,
        horizon: int,
    ) -> int | None:
        first = self._neighbor(self.game.player.grid, first_direction)
        if first is None or not targets:
            return None
        if first in targets:
            return 1

        # Excluding the origin prevents an invalid "go out, reverse, come back"
        # route from making every first action look equally attractive.
        visited = {self.game.player.grid, first}
        queue = deque([(first, 1)])
        while queue:
            cell, distance = queue.popleft()
            if distance >= horizon:
                continue
            for direction in _CLOCKWISE:
                neighbor = self._neighbor(cell, direction)
                if neighbor is None or neighbor in visited:
                    continue
                if neighbor in targets:
                    return distance + 1
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return None

    def _nearest_distance(self, targets: set[tuple[int, int]]) -> int | None:
        origin = self.game.player.grid
        if origin in targets:
            return 0
        if not targets:
            return None
        visited = {origin}
        queue = deque([(origin, 0)])
        while queue:
            cell, distance = queue.popleft()
            for direction in _CLOCKWISE:
                neighbor = self._neighbor(cell, direction)
                if neighbor is None or neighbor in visited:
                    continue
                if neighbor in targets:
                    return distance + 1
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return None

    def _neighbor(
        self, cell: tuple[int, int], direction: Direction
    ) -> tuple[int, int] | None:
        if not self.game._can_move(cell, direction):
            return None
        dx, dy = direction.vector
        x, y = cell[0] + dx, cell[1] + dy
        if y == 9:
            x %= self.game.cols
        if not (0 <= x < self.game.cols and 0 <= y < self.game.rows):
            return None
        return x, y

    def _cells_containing(self, character: str) -> set[tuple[int, int]]:
        return {
            (x, y)
            for y, row in enumerate(self.game.maze)
            for x, cell in enumerate(row)
            if cell == character
        }

    def _dot_cells(self) -> set[tuple[int, int]]:
        return {
            (x, y)
            for y, row in enumerate(self.game.maze)
            for x, cell in enumerate(row)
            if cell in ".o"
        }

    def _count_cell(self, character: str) -> int:
        return sum(cell == character for row in self.game.maze for cell in row)

    def _base_info(self) -> dict:
        projectile_data = self.game.projectile_telemetry()
        return {
            "score": self.game.score,
            "lives": self.game.lives,
            "level": self.game.level,
            "pellets_remaining": self.game._count_dots(),
            "player_grid": self.game.player.grid,
            "player_direction": self.game.player.direction.name,
            "phase": self.game.phase.name,
            "ghost_mode": self.game.ghost_mode.name,
            "frightened_timer": self.game.frightened_timer,
            "ghost_speed_multiplier": self.game.ghost_speed_multiplier,
            "frightened_duration": self.game.frightened_duration,
            "projectiles": projectile_data,
            "projectiles_active": projectile_data["active_count"],
            "player_slowed": projectile_data["player_slowed"],
            "slow_fraction": projectile_data["slow_fraction"],
            "slow_timer": projectile_data["slow_timer"],
            "episode_steps": self.episode_steps,
            "episode_return": self.episode_return,
        }
