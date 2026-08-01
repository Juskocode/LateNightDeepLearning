"""Snake environment and live reinforcement-learning observatory."""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
from typing import Optional, Sequence, Tuple

import numpy as np
import pygame

from .constants import Direction, FRAME_TIMEOUT_MULTIPLIER, Point
from .sprites import SnakeSpriteAtlas
from snakeGameQDlearning.src.config.settings import (
    BLACK,
    BLOCK_SIZE,
    BLUE2,
    COLLISION_PENALTY,
    FONT_PATH,
    FOOD_REWARD,
    GAME_HEIGHT,
    GAME_SPEED,
    GAME_WIDTH,
    GREEN,
    GRID,
    GRID_HIGHLIGHT,
    HEADER_HEIGHT,
    LOOP_PENALTY,
    MARGIN,
    MUTED,
    ORANGE,
    PANEL_BG,
    PANEL_WIDTH,
    PURPLE,
    RED,
    WHITE,
    WIN_REWARD,
    YELLOW,
)


VISION_LABELS = (
    "danger ahead",
    "danger right",
    "danger left",
    "heading left",
    "heading right",
    "heading up",
    "heading down",
    "food left",
    "food right",
    "food up",
    "food down",
)
ACTION_LABELS = ("STRAIGHT", "TURN RIGHT", "TURN LEFT")
_CLOCKWISE = (Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP)


class SnakeGameAI:
    """Deterministic grid environment with an optional pygame presentation.

    Game state is independent from rendering.  ``play_step`` therefore remains
    fast in headless runs, while ``render`` can be called before a step to show
    the exact observation/action pair currently being evaluated.
    """

    def __init__(
        self,
        width: int = GAME_WIDTH,
        height: int = GAME_HEIGHT,
        render: bool = True,
        speed: int = GAME_SPEED,
        seed: Optional[int] = None,
        randomize_start: bool = False,
        process_events: bool = True,
    ):
        if width % BLOCK_SIZE or height % BLOCK_SIZE:
            raise ValueError(f"width and height must be multiples of {BLOCK_SIZE}")
        if width < BLOCK_SIZE * 4 or height < BLOCK_SIZE * 3:
            raise ValueError("board is too small for the initial snake")

        pygame.init()
        self.width = width
        self.height = height
        self.content_height = max(height, GAME_HEIGHT)
        self.window_width = width + PANEL_WIDTH + MARGIN * 3
        self.window_height = self.content_height + HEADER_HEIGHT + MARGIN
        self.render_enabled = render
        self.speed = max(1, int(speed))
        self.rng = random.Random(seed)
        self.episode_seed = seed
        self.default_randomize_start = bool(randomize_start)
        self.process_events = bool(process_events)
        if render:
            self.display = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            pygame.display.set_caption("Snake RL Observatory")
        else:
            self.display = pygame.Surface((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font = self._font(17)
        self.small_font = self._font(13)
        self.tiny_font = self._font(11)
        self.micro_font = self._font(10)
        self.sprites = SnakeSpriteAtlas()

        self.score = 0
        self.best_score = 0
        self.running = True
        self.paused = False
        self.show_vision = True
        self._single_step = False
        self.telemetry: dict = {}
        self.reset(seed=seed, randomize_start=randomize_start)

    def _font(self, size: int):
        try:
            return pygame.font.Font(str(FONT_PATH), size)
        except (FileNotFoundError, pygame.error):
            return pygame.font.Font(None, size)

    @property
    def starvation_budget(self) -> int:
        return FRAME_TIMEOUT_MULTIPLIER * len(self.snake)

    def reset(
        self, *, seed: Optional[int] = None, randomize_start: Optional[bool] = None
    ) -> None:
        if seed is not None:
            self.rng.seed(seed)
            self.episode_seed = seed
        if randomize_start is None:
            randomize_start = self.default_randomize_start
        columns = self.width // BLOCK_SIZE
        rows = self.height // BLOCK_SIZE
        if randomize_start:
            self.direction, self.head = self._random_start(columns, rows)
        else:
            self.direction = Direction.RIGHT
            self.head = Point((columns // 2) * BLOCK_SIZE, (rows // 2) * BLOCK_SIZE)
        dx, dy = self.direction.value
        self.snake = [
            self.head,
            Point(self.head.x - dx * BLOCK_SIZE, self.head.y - dy * BLOCK_SIZE),
            Point(self.head.x - 2 * dx * BLOCK_SIZE, self.head.y - 2 * dy * BLOCK_SIZE),
        ]
        self.best_score = max(self.score, self.best_score)
        self.score = 0
        self.food: Optional[Point] = None
        self.frame_iteration = 0
        self.steps_since_food = 0
        self.termination_reason: Optional[str] = None
        self.crash_point: Optional[Point] = None
        self.won = False
        self.transition_applied = False
        self.last_action_index = 0
        self.last_environment_reward = 0.0
        self.last_distance_delta = 0
        self.last_ate_food = False
        self.last_visit_count = 1
        self.path_history = deque([self.head], maxlen=120)
        self.visit_counts = Counter({self.head: 1})
        self._place_food()

    def _random_start(self, columns: int, rows: int) -> tuple[Direction, Point]:
        """Choose a valid orientation and head while keeping three cells in-bounds."""

        direction = self.rng.choice(list(Direction))
        dx, dy = direction.value
        candidates = []
        for column in range(columns):
            for row in range(rows):
                tail_column = column - 2 * dx
                tail_row = row - 2 * dy
                next_column = column + dx
                next_row = row + dy
                if (
                    0 <= tail_column < columns
                    and 0 <= tail_row < rows
                    and 0 <= next_column < columns
                    and 0 <= next_row < rows
                ):
                    candidates.append(Point(column * BLOCK_SIZE, row * BLOCK_SIZE))
        return direction, self.rng.choice(candidates)

    def _place_food(self) -> bool:
        occupied = set(self.snake)
        free_cells = [
            Point(x, y)
            for x in range(0, self.width, BLOCK_SIZE)
            for y in range(0, self.height, BLOCK_SIZE)
            if Point(x, y) not in occupied
        ]
        if not free_cells:
            self.food = None
            self.won = True
            return False
        self.food = self.rng.choice(free_cells)
        return True

    def set_debug_info(self, **telemetry) -> None:
        self.telemetry.update(telemetry)

    def _handle_events(self) -> None:
        if not self.process_events:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.termination_reason = "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    self.termination_reason = "quit"
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset()
                    self._control_interrupted = True
                elif event.key == pygame.K_v:
                    self.show_vision = not self.show_vision
                elif event.key in (pygame.K_PERIOD, pygame.K_n) and self.paused:
                    self._single_step = True
                elif event.key in (pygame.K_MINUS, pygame.K_LEFTBRACKET):
                    self.speed = max(1, self.speed // 2)
                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                    pygame.K_RIGHTBRACKET,
                ):
                    self.speed = min(1000, self.speed * 2)

    def play_step(
        self, action: Optional[Sequence[int]], *, render_frame: bool = True
    ) -> Tuple[float, bool, int]:
        """Apply one relative action and return ``(reward, terminal, score)``.

        ``action`` may be ``None`` while paused, allowing the UI event loop to
        remain responsive without adding fake transitions to replay memory.
        """
        self.transition_applied = False
        self._control_interrupted = False
        self.last_ate_food = False
        self.last_distance_delta = 0
        self._handle_events()

        if not self.running:
            self._render_if_requested(render_frame)
            return 0.0, True, self.score
        if self._control_interrupted:
            self._render_if_requested(render_frame)
            return 0.0, False, self.score
        if action is None:
            self._render_if_requested(render_frame)
            return 0.0, False, self.score
        if self.paused and not self._single_step:
            self._render_if_requested(render_frame)
            return 0.0, False, self.score
        self._single_step = False

        new_direction, next_head, action_index = self._resolve_action(action)
        ate_food = self.food is not None and next_head == self.food
        collision = self._collision_kind(next_head, include_tail=ate_food)
        previous_distance = self._distance_to_food(self.head)

        self.frame_iteration += 1
        self.steps_since_food += 1
        self.last_action_index = action_index
        self.transition_applied = True
        self.termination_reason = None
        self.crash_point = None

        if collision:
            self.termination_reason = collision
            self.crash_point = next_head
            self.last_environment_reward = float(COLLISION_PENALTY)
            self._render_if_requested(render_frame)
            return self.last_environment_reward, True, self.score

        self.direction = new_direction
        self.head = next_head
        self.snake.insert(0, self.head)
        self.path_history.append(self.head)
        self.visit_counts[self.head] += 1
        self.last_visit_count = self.visit_counts[self.head]

        reward = 0.0
        game_over = False
        if ate_food:
            self.score += 1
            self.steps_since_food = 0
            self.last_ate_food = True
            reward = float(FOOD_REWARD)
            if not self._place_food():
                self.termination_reason = "win"
                reward += float(WIN_REWARD)
                game_over = True
        else:
            self.snake.pop()

        if self.food is not None:
            self.last_distance_delta = (
                self._distance_to_food(self.head) - previous_distance
            )
        if not game_over and self.steps_since_food > self.starvation_budget:
            self.termination_reason = "timeout"
            reward = float(LOOP_PENALTY)
            game_over = True

        self.last_environment_reward = reward
        self._render_if_requested(render_frame)
        return reward, game_over, self.score

    def _distance_to_food(self, point: Point) -> int:
        if self.food is None:
            return 0
        return abs(point.x - self.food.x) + abs(point.y - self.food.y)

    def _collision_kind(
        self, point: Point, *, include_tail: bool = False
    ) -> Optional[str]:
        if (
            point.x < 0
            or point.x >= self.width
            or point.y < 0
            or point.y >= self.height
        ):
            return "wall"
        occupied = self.snake if include_tail else self.snake[:-1]
        if point in occupied:
            return "self"
        return None

    def is_collision(self, point: Optional[Point] = None) -> bool:
        """Return whether ``point`` is blocked for the next normal move.

        The current tail is deliberately excluded: it vacates its cell on a
        non-growing step and is therefore a legal destination.
        """
        target = self.head if point is None else point
        if (
            target.x < 0
            or target.x >= self.width
            or target.y < 0
            or target.y >= self.height
        ):
            return True
        if point is None:
            return target in self.snake[1:]
        return target in self.snake[:-1]

    def render(self) -> None:
        self._update_ui()
        if self.render_enabled:
            pygame.display.flip()
            self.clock.tick(self.speed)

    def _render_if_requested(self, requested: bool) -> None:
        if requested and self.render_enabled:
            self.render()

    def _update_ui(self) -> None:
        self.display.fill(BLACK)
        self._draw_header()
        self._draw_board()
        self._draw_inspector()

    def _draw_header(self) -> None:
        title = self.font.render("SNAKE  /  RL OBSERVATORY", True, WHITE)
        subtitle = self.small_font.render(
            "SPACE pause   N step   [ ] speed   V vision   R reset   ESC quit",
            True,
            MUTED,
        )
        self.display.blit(title, (MARGIN, 14))
        self.display.blit(subtitle, (MARGIN, 40))

        score = self.font.render(
            f"SCORE  {self.score:03d}     BEST  {self.best_score:03d}", True, GREEN
        )
        self.display.blit(score, (self.window_width - score.get_width() - MARGIN, 16))
        status = "PAUSED" if self.paused else f"{self.speed} FPS"
        status_color = YELLOW if self.paused else MUTED
        status_text = self.tiny_font.render(status, True, status_color)
        self.display.blit(
            status_text, (self.window_width - status_text.get_width() - MARGIN, 43)
        )

    def _draw_board(self) -> None:
        board_x, board_y = MARGIN, HEADER_HEIGHT
        board = pygame.Rect(board_x, board_y, self.width, self.height)
        pygame.draw.rect(self.display, PANEL_BG, board, border_radius=8)

        for x in range(0, self.width + 1, BLOCK_SIZE):
            pygame.draw.line(
                self.display,
                GRID,
                (board_x + x, board_y),
                (board_x + x, board_y + self.height),
            )
        for y in range(0, self.height + 1, BLOCK_SIZE):
            pygame.draw.line(
                self.display,
                GRID,
                (board_x, board_y + y),
                (board_x + self.width, board_y + y),
            )

        # Recent positions form a faint trail; repeated cells become easier to spot.
        trail = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        recent = list(self.path_history)[-42:]
        for index, point in enumerate(recent):
            alpha = 6 + round(24 * (index + 1) / max(1, len(recent)))
            pygame.draw.rect(
                trail,
                (*PURPLE, alpha),
                (point.x + 2, point.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4),
                border_radius=4,
            )
        self.display.blit(trail, (board_x, board_y))

        if self.show_vision:
            self._draw_vision_overlay(board_x, board_y)

        ticks = pygame.time.get_ticks()
        for index in range(len(self.snake) - 1, -1, -1):
            point = self.snake[index]
            if index == 0:
                image = self.sprites.head(self.direction)
            else:
                neighbours = [self.snake[index - 1]]
                if index + 1 < len(self.snake):
                    neighbours.append(self.snake[index + 1])
                connections = [
                    self._direction_between(point, neighbour)
                    for neighbour in neighbours
                ]
                image = self.sprites.body(connections, ticks, index)
            self.display.blit(image, (board_x + point.x, board_y + point.y))

        if self.food is not None:
            self.display.blit(
                self.sprites.food(ticks), (board_x + self.food.x, board_y + self.food.y)
            )
        if (
            self.crash_point is not None
            and 0 <= self.crash_point.x < self.width
            and 0 <= self.crash_point.y < self.height
        ):
            self.display.blit(
                self.sprites.crash(),
                (board_x + self.crash_point.x, board_y + self.crash_point.y),
            )

        budget_fraction = min(
            1.0, self.steps_since_food / max(1, self.starvation_budget)
        )
        meter = pygame.Rect(
            board_x + 10, board_y + self.height - 15, self.width - 20, 5
        )
        pygame.draw.rect(self.display, GRID_HIGHLIGHT, meter, border_radius=3)
        meter.width = round(meter.width * budget_fraction)
        if meter.width:
            pygame.draw.rect(
                self.display,
                RED if budget_fraction > 0.8 else YELLOW,
                meter,
                border_radius=3,
            )

        if self.paused:
            shade = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            shade.fill((7, 10, 22, 125))
            self.display.blit(shade, (board_x, board_y))
            paused = self.font.render("PAUSED  ·  N TO STEP", True, WHITE)
            self.display.blit(
                paused,
                (
                    board_x + (self.width - paused.get_width()) // 2,
                    board_y + self.height // 2 - 12,
                ),
            )

    def _draw_vision_overlay(self, board_x: int, board_y: int) -> None:
        state = list(self.telemetry.get("state", [0] * 11))
        head_center = (
            board_x + self.head.x + BLOCK_SIZE // 2,
            board_y + self.head.y + BLOCK_SIZE // 2,
        )
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        if self.food is not None:
            food_center = (
                board_x + self.food.x + BLOCK_SIZE // 2,
                board_y + self.food.y + BLOCK_SIZE // 2,
            )
            self._dashed_line(
                self.display, PURPLE, head_center, food_center, dash=7, gap=7, width=1
            )

        for index, (label, direction) in enumerate(
            zip(("S", "R", "L"), self._relative_directions())
        ):
            dx, dy = direction.value
            target = Point(self.head.x + dx * BLOCK_SIZE, self.head.y + dy * BLOCK_SIZE)
            danger = (
                bool(state[index]) if index < len(state) else self.is_collision(target)
            )
            color = RED if danger else GREEN
            if 0 <= target.x < self.width and 0 <= target.y < self.height:
                pygame.draw.rect(
                    overlay,
                    (*color, 42),
                    (target.x + 1, target.y + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2),
                    border_radius=4,
                )
                rect = pygame.Rect(
                    board_x + target.x + 1,
                    board_y + target.y + 1,
                    BLOCK_SIZE - 2,
                    BLOCK_SIZE - 2,
                )
                pygame.draw.rect(self.display, color, rect, 1, border_radius=4)
                marker = self.micro_font.render(label, True, color)
                self.display.blit(marker, (rect.x + 3, rect.y + 2))
            end = (head_center[0] + dx * BLOCK_SIZE, head_center[1] + dy * BLOCK_SIZE)
            pygame.draw.line(self.display, color, head_center, end, 2)
        self.display.blit(overlay, (board_x, board_y))

    @staticmethod
    def _dashed_line(
        surface, color, start, end, *, dash: int, gap: int, width: int
    ) -> None:
        vector = pygame.Vector2(end) - pygame.Vector2(start)
        length = vector.length()
        if not length:
            return
        direction = vector.normalize()
        distance = 0.0
        while distance < length:
            segment_end = min(distance + dash, length)
            pygame.draw.line(
                surface,
                color,
                pygame.Vector2(start) + direction * distance,
                pygame.Vector2(start) + direction * segment_end,
                width,
            )
            distance += dash + gap

    def _draw_inspector(self) -> None:
        panel_left = MARGIN * 2 + self.width
        panel = pygame.Rect(panel_left, HEADER_HEIGHT, PANEL_WIDTH, self.content_height)
        pygame.draw.rect(self.display, PANEL_BG, panel, border_radius=8)
        x = panel_left + 16
        width = PANEL_WIDTH - 32
        y = HEADER_HEIGHT + 12

        algorithm = (
            str(self.telemetry.get("algorithm", "double_dqn")).upper().replace("_", " ")
        )
        policy = str(self.telemetry.get("policy_mode", "explore")).upper()
        self._label("DECISION ENGINE", x, y)
        self._chip(algorithm, x + 124, y - 4, PURPLE)
        policy_text = self.micro_font.render(policy, True, WHITE)
        self._chip(
            policy,
            x + width - policy_text.get_width() - 12,
            y - 4,
            YELLOW if policy == "EXPLORE" else GREEN,
        )
        y += 25

        q_values = [float(value) for value in self.telemetry.get("q_values", [0.0] * 3)]
        target_values = [
            float(value) for value in self.telemetry.get("target_q_values", [0.0] * 3)
        ]
        selected = int(self.telemetry.get("action_index", 0))
        scale = max(1.0, *(abs(value) for value in q_values))
        bar_x, bar_width = x + 94, width - 140
        center = bar_x + bar_width // 2
        for index, label in enumerate(ACTION_LABELS):
            row_y = y + index * 23
            self._text(
                label,
                x,
                row_y + 2,
                WHITE if index == selected else MUTED,
                self.tiny_font,
            )
            bar = pygame.Rect(bar_x, row_y, bar_width, 16)
            pygame.draw.rect(self.display, GRID, bar, border_radius=4)
            pygame.draw.line(
                self.display, MUTED, (center, row_y + 2), (center, row_y + 14), 1
            )
            magnitude = max(
                1, round((bar_width // 2 - 2) * abs(q_values[index]) / scale)
            )
            if q_values[index] >= 0:
                value_bar = pygame.Rect(center, row_y + 2, magnitude, 12)
                color = GREEN if index == selected else BLUE2
            else:
                value_bar = pygame.Rect(center - magnitude, row_y + 2, magnitude, 12)
                color = ORANGE if index == selected else PURPLE
            pygame.draw.rect(self.display, color, value_bar, border_radius=3)
            if index == selected:
                pygame.draw.rect(self.display, WHITE, bar, 1, border_radius=4)
            self._text(
                f"{q_values[index]:+.2f}",
                x + width - 39,
                row_y + 2,
                WHITE,
                self.tiny_font,
            )
        y += 76

        reward = float(self.telemetry.get("reward", 0.0))
        reason = str(self.telemetry.get("termination_reason") or "transition active")
        self._text(
            f"LAST  r {reward:+.2f}  ·  {reason}",
            x,
            y,
            RED if reward < 0 else GREEN if reward > 0 else MUTED,
            self.tiny_font,
        )
        target_gap = max(
            (abs(a - b) for a, b in zip(q_values, target_values)), default=0.0
        )
        self._text(
            f"target gap {target_gap:.3f}", x + width - 104, y, MUTED, self.tiny_font
        )
        y += 21

        state = list(self.telemetry.get("state", [0] * 11))
        bits = "".join("1" if bool(value) else "0" for value in state)
        self._label("VISION  /  11 BINARY FEATURES", x, y)
        self._text(
            f"{bits[:3]} · {bits[3:7]} · {bits[7:]}",
            x + width - 105,
            y,
            MUTED,
            self.micro_font,
        )
        y += 18
        for index, label in enumerate(VISION_LABELS):
            column = index // 6
            row = index % 6
            bx = x + column * 184
            by = y + row * 17
            active = bool(state[index]) if index < len(state) else False
            pygame.draw.circle(
                self.display, GREEN if active else GRID_HIGHLIGHT, (bx + 4, by + 6), 4
            )
            self._text(label, bx + 13, by, WHITE if active else MUTED, self.micro_font)
        y += 108

        self._label("LEARNING SIGNALS", x, y)
        y += 17
        metrics = (
            ("EPSILON", f"{float(self.telemetry.get('epsilon', 0.0)):.3f}"),
            ("LOSS", f"{float(self.telemetry.get('loss', 0.0)):.4f}"),
            ("RETURN", f"{float(self.telemetry.get('episode_return', 0.0)):+.1f}"),
            ("EPISODES", str(self.telemetry.get("games", 0))),
            ("STEP", str(self.frame_iteration)),
            ("GRAD NORM", f"{float(self.telemetry.get('gradient_norm', 0.0)):.3f}"),
        )
        cell_width = width // 3
        for index, (label, value) in enumerate(metrics):
            bx = x + (index % 3) * cell_width
            by = y + (index // 3) * 29
            self._text(label, bx, by, MUTED, self.micro_font)
            self._text(value, bx, by + 10, WHITE, self.tiny_font)
        y += 62

        memory = int(self.telemetry.get("memory", 0))
        capacity = max(1, int(self.telemetry.get("memory_capacity", 100_000)))
        self._label("REPLAY MEMORY", x, y)
        self._text(
            f"{memory:,} / {capacity:,}", x + width - 97, y, WHITE, self.micro_font
        )
        y += 16
        pygame.draw.rect(self.display, GRID, (x, y, width, 8), border_radius=4)
        fill = min(width, round(width * memory / capacity))
        if fill:
            pygame.draw.rect(self.display, BLUE2, (x, y, fill, 8), border_radius=4)
        y += 13
        recent_rewards = list(self.telemetry.get("recent_rewards", []))[-24:]
        recent_dones = list(self.telemetry.get("recent_dones", []))[-24:]
        slot_width = max(3, (width - 23) // 24)
        for index in range(24):
            value = recent_rewards[index] if index < len(recent_rewards) else None
            terminal = bool(recent_dones[index]) if index < len(recent_dones) else False
            color = (
                GRID_HIGHLIGHT
                if value is None
                else (GREEN if value > 0 else RED if value < 0 else BLUE2)
            )
            if terminal:
                color = YELLOW
            pygame.draw.rect(
                self.display,
                color,
                (x + index * (slot_width + 1), y, slot_width, 7),
                border_radius=2,
            )
        y += 14

        sync_progress = float(self.telemetry.get("target_sync_progress", 0.0))
        structure = str(
            self.telemetry.get("model_structure", "ONLINE  11 → 512 → 256 → 3")
        )
        self._text(structure, x, y, WHITE, self.micro_font)
        if self.telemetry.get("algorithm_family", "deep") == "deep":
            self._text(
                f"SYNC {sync_progress:5.1%}",
                x + width - 72,
                y,
                MUTED,
                self.micro_font,
            )
        y += 14
        pygame.draw.line(self.display, GRID_HIGHLIGHT, (x, y), (x + width, y), 1)
        evaluation = dict(self.telemetry.get("evaluation", {}))
        evaluation_episodes = int(evaluation.get("episodes", 0))
        stage = str(self.telemetry.get("curriculum_stage", "orientation")).replace(
            "_", " "
        )
        if evaluation_episodes:
            evaluation_text = (
                f"VALIDATION {float(evaluation.get('mean_score', 0.0)):.2f}"
                f" ±{float(evaluation.get('std_score', 0.0)):.2f}  "
                f"GAP {float(evaluation.get('generalization_gap', 0.0)):+.2f}"
                f"  ·  {stage}"
            )
        else:
            evaluation_text = f"VALIDATION pending  ·  curriculum {stage}"
        self._text(evaluation_text, x, y + 6, MUTED, self.micro_font)

    def _chip(self, value: str, x: int, y: int, color) -> None:
        text = self.micro_font.render(value, True, color)
        rect = pygame.Rect(x, y, text.get_width() + 12, 17)
        pygame.draw.rect(self.display, GRID_HIGHLIGHT, rect, border_radius=8)
        self.display.blit(text, (x + 6, y + 3))

    def _label(self, value, x, y):
        self._text(value, x, y, MUTED, self.micro_font)

    def _text(self, value, x, y, color, font):
        self.display.blit(font.render(str(value), True, color), (x, y))

    def _relative_directions(self):
        index = _CLOCKWISE.index(self.direction)
        return (
            _CLOCKWISE[index],
            _CLOCKWISE[(index + 1) % 4],
            _CLOCKWISE[(index - 1) % 4],
        )

    @staticmethod
    def _direction_between(origin: Point, neighbour: Point) -> Direction:
        delta = (neighbour.x - origin.x, neighbour.y - origin.y)
        try:
            return Direction((int(np.sign(delta[0])), int(np.sign(delta[1]))))
        except ValueError as error:
            raise ValueError(
                "snake segments must occupy adjacent grid cells"
            ) from error

    def _resolve_action(self, action: Sequence[int]):
        values = np.asarray(action)
        if (
            values.shape != (3,)
            or not np.isin(values, (0, 1)).all()
            or int(values.sum()) != 1
        ):
            raise ValueError("action must be one-hot: [straight, right, left]")
        action_index = int(values.argmax())
        index = _CLOCKWISE.index(self.direction)
        offsets = (0, 1, -1)
        new_direction = _CLOCKWISE[(index + offsets[action_index]) % 4]
        dx, dy = new_direction.value
        next_head = Point(self.head.x + dx * BLOCK_SIZE, self.head.y + dy * BLOCK_SIZE)
        return new_direction, next_head, action_index

    def _move(self, action: Sequence[int]) -> None:
        """Compatibility helper used by simple environment experiments."""
        self.direction, self.head, self.last_action_index = self._resolve_action(action)

    def save_screenshot(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._update_ui()
        pygame.image.save(self.display, str(output))
        return output
