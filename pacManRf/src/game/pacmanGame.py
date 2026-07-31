"""A compact, complete Pacman-style game built around explicit game state."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pygame

from .constants import (
    BLACK, CLEAR_SECONDS, CYAN, DEATH_SECONDS, EATEN_GHOST_SPEED,
    EXTRA_LIFE_SCORE, FPS, FRIGHTENED_DECREASE_PER_LEVEL, FRIGHTENED_SECONDS,
    FRIGHTENED_SPEED, GHOST_SPEED, GHOST_SPEED_INCREASE_PER_LEVEL, HUD_BG,
    HUD_HEIGHT, MAX_GHOST_SPEED_MULTIPLIER, MIN_FRIGHTENED_SECONDS, MUTED,
    PELLET, PINK, PLAYER_SPEED, READY_SECONDS, STARTING_LIVES, TILE_SIZE,
    WALL_BLUE, WALL_FILL, WALL_GLOW, WHITE, YELLOW, Direction, FONT_PATH,
    GamePhase, GameStatus, GhostMode,
)
from .pacmanSprite import GhostSprite, PacmanSprite
from .projectileSprite import ProjectileSprite
from .projectiles import (
    DEFAULT_PROJECTILE_SPECS,
    GhostProjectileSystem,
    PacmanSlowState,
    Projectile,
    ProjectileEvent,
    ProjectileKind,
)


MAZE_TEMPLATE = (
    "####################",
    "#o......##........o#",
    "#.##.###.##.###.##.#",
    "#..................#",
    "#.##.#.######.#.##.#",
    "#....#...##...#....#",
    "####.### ## ###.####",
    "   #.#        #.#   ",
    "####.# ##  ## #.####",
    "T......#    #......T",
    "####.# ###### #.####",
    "   #.#        #.#   ",
    "####.# ##  ## #.####",
    "#........##........#",
    "#.##.###.##.###.##.#",
    "#o.#..... ......#.o#",
    "##.#.#.######.#.#.##",
    "#....#...##...#....#",
    "#.######.##.######.#",
    "#..................#",
    "####################",
)

GHOST_SPECS = (
    ("BLINKY", (234, 63, 76), (9, 9)),
    ("PINKY", (255, 132, 194), (10, 9)),
    ("INKY", (55, 219, 223), (9, 11)),
    ("CLYDE", (255, 166, 55), (10, 11)),
)


@dataclass
class Mover:
    grid_x: int
    grid_y: int
    direction: Direction
    speed: float
    sprite: object
    target: Optional[tuple[int, int]] = None

    def __post_init__(self):
        self.position = pygame.Vector2(
            self.grid_x * TILE_SIZE + TILE_SIZE / 2,
            self.grid_y * TILE_SIZE + TILE_SIZE / 2,
        )

    @property
    def grid(self) -> tuple[int, int]:
        return self.grid_x, self.grid_y

    def reset_position(self, grid: tuple[int, int], direction: Direction) -> None:
        self.grid_x, self.grid_y = grid
        self.direction = direction
        self.target = None
        self.position.update(
            self.grid_x * TILE_SIZE + TILE_SIZE / 2,
            self.grid_y * TILE_SIZE + TILE_SIZE / 2,
        )


@dataclass
class Ghost(Mover):
    name: str = "GHOST"
    spawn: tuple[int, int] = (9, 9)
    eaten: bool = False
    released: bool = False
    release_timer: float = 0.0
    pending_reverse: bool = False


@dataclass
class ScorePopup:
    text: str
    position: pygame.Vector2
    color: tuple[int, int, int]
    lifetime: float = 0.8
    age: float = 0.0


@dataclass
class ProjectileImpact:
    """Short-lived visual left behind when a projectile despawns."""

    kind: ProjectileKind
    position: pygame.Vector2
    lifetime: float = 0.34
    age: float = 0.0


MODE_SCHEDULE = (
    (GhostMode.SCATTER, 7.0),
    (GhostMode.CHASE, 20.0),
    (GhostMode.SCATTER, 7.0),
    (GhostMode.CHASE, 20.0),
    (GhostMode.SCATTER, 5.0),
    (GhostMode.CHASE, math.inf),
)

RELEASE_DELAYS = (0.0, 1.4, 3.6, 5.8)
SCATTER_TARGETS = ((19, 0), (0, 0), (19, 20), (0, 20))
PROJECTILE_READY_DELAY = 1.0
PLAYER_PROJECTILE_COLLISION_RADIUS_TILES = 0.30


class PacmanGame:
    """Owns rules, state transitions, rendering, and a deterministic update loop."""

    def __init__(
        self,
        render: bool = True,
        seed: int = 7,
        *,
        auto_advance_on_clear: bool = True,
    ):
        pygame.init()
        self.rng = random.Random(seed)
        self.projectile_system = GhostProjectileSystem(self.rng)
        self.player_slow = PacmanSlowState()
        self.projectile_sprites = {
            spec.kind: ProjectileSprite(spec, TILE_SIZE)
            for spec in DEFAULT_PROJECTILE_SPECS
        }
        self.projectile_impacts: list[ProjectileImpact] = []
        self.last_projectile_events: tuple[ProjectileEvent, ...] = ()
        self.projectile_shots_fired = 0
        self.fireball_hits = 0
        self.freeze_ball_hits = 0
        self.render_enabled = render
        self.auto_advance_on_clear = bool(auto_advance_on_clear)
        self.cols = len(MAZE_TEMPLATE[0])
        self.rows = len(MAZE_TEMPLATE)
        self.maze_width = self.cols * TILE_SIZE
        self.maze_height = self.rows * TILE_SIZE
        self.display_height = self.maze_height + HUD_HEIGHT
        if render:
            self.display = pygame.display.set_mode((self.maze_width, self.display_height))
            pygame.display.set_caption("Late Night Pacman")
        else:
            # A private surface keeps headless environments independent from a
            # separate observability window created in the same process.
            self.display = pygame.Surface((self.maze_width, self.display_height))
        self.clock = pygame.time.Clock()
        self.font = self._font(18)
        self.small_font = self._font(13)
        self.tiny_font = self._font(11)
        self.running = True
        self.status = GameStatus.PLAYING
        self.score = 0
        self.high_score = 0
        self.lives = STARTING_LIVES
        self.level = 1
        self.phase = GamePhase.READY
        self.phase_timer = READY_SECONDS
        self.animation_time = 0.0
        self.frightened_timer = 0.0
        self.ghost_chain = 0
        self.mode_index = 0
        self.ghost_mode = MODE_SCHEDULE[0][0]
        self.mode_timer = MODE_SCHEDULE[0][1]
        self.extra_life_awarded = False
        self.score_popups: list[ScorePopup] = []
        self.flash_timer = 0.0
        self.next_direction = Direction.LEFT
        self.maze = [list(row) for row in MAZE_TEMPLATE]
        self.total_dots = self._count_dots()

        self.player = Mover(9, 15, Direction.LEFT, PLAYER_SPEED, PacmanSprite(TILE_SIZE - 2))
        self.ghosts = []
        for index, (name, color, spawn) in enumerate(GHOST_SPECS):
            ghost = Ghost(
                spawn[0], spawn[1], Direction.LEFT if index % 2 == 0 else Direction.RIGHT,
                GHOST_SPEED, GhostSprite(color, TILE_SIZE - 2), name=name, spawn=spawn,
            )
            self.ghosts.append(ghost)
        self.maze_surface = self._build_maze_surface()
        self._reset_round_positions()
        self._eat_current_cell()
        self._render(0.0)

    def _font(self, size: int) -> pygame.font.Font:
        try:
            return pygame.font.Font(str(FONT_PATH), size)
        except (FileNotFoundError, pygame.error):
            return pygame.font.Font(None, size)

    @property
    def pacman(self):
        """Compatibility alias used by older examples."""
        return self.player.position

    @property
    def direction(self):
        return self.player.direction

    @property
    def ghost_speed_multiplier(self) -> float:
        increase = (self.level - 1) * GHOST_SPEED_INCREASE_PER_LEVEL
        return min(MAX_GHOST_SPEED_MULTIPLIER, 1.0 + max(0.0, increase))

    @property
    def frightened_duration(self) -> float:
        decrease = (self.level - 1) * FRIGHTENED_DECREASE_PER_LEVEL
        return max(MIN_FRIGHTENED_SECONDS, FRIGHTENED_SECONDS - max(0.0, decrease))

    @property
    def active_projectiles(self) -> tuple[Projectile, ...]:
        """Read-only projectile snapshot used by rendering and observability."""

        return self.projectile_system.active_projectiles

    @property
    def player_speed_multiplier(self) -> float:
        return self.player_slow.speed_multiplier

    def _count_dots(self) -> int:
        return sum(cell in ".o" for row in self.maze for cell in row)

    def play_step(self, action: Optional[Direction] = None, dt: Optional[float] = None):
        """Advance one frame and return ``(terminal, score)``.

        Supplying ``action`` and ``dt`` makes the loop deterministic for tests and
        reinforcement-learning wrappers; keyboard control remains the default.
        """
        frame_dt = min(dt if dt is not None else self.clock.tick(FPS) / 1000.0, 0.05)
        self._handle_events()
        if action is not None:
            self.next_direction = action

        if self.status == GameStatus.PLAYING or self.phase == GamePhase.DYING:
            self._update(frame_dt)
        self._render(frame_dt)
        terminal = self.status in (GameStatus.WON, GameStatus.LOST) or not self.running
        return terminal, self.score

    def _handle_events(self) -> None:
        key_directions = {
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT,
            pygame.K_UP: Direction.UP,
            pygame.K_DOWN: Direction.DOWN,
            pygame.K_a: Direction.LEFT,
            pygame.K_d: Direction.RIGHT,
            pygame.K_w: Direction.UP,
            pygame.K_s: Direction.DOWN,
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in key_directions:
                    self.next_direction = key_directions[event.key]
                elif event.key == pygame.K_SPACE and self.status in (GameStatus.PLAYING, GameStatus.PAUSED):
                    self.status = GameStatus.PAUSED if self.status == GameStatus.PLAYING else GameStatus.PLAYING
                elif event.key == pygame.K_r:
                    self.restart()
                elif event.key in (pygame.K_RETURN, pygame.K_n) and self.status == GameStatus.WON:
                    self.next_level()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

    def _update(self, dt: float) -> None:
        self.animation_time += dt
        self.flash_timer = max(0.0, self.flash_timer - dt)
        for popup in self.score_popups:
            popup.age += dt
            popup.position.y -= 24 * dt
        self.score_popups = [popup for popup in self.score_popups if popup.age < popup.lifetime]
        for impact in self.projectile_impacts:
            impact.age += dt
        self.projectile_impacts = [
            impact for impact in self.projectile_impacts if impact.age < impact.lifetime
        ]
        self.last_projectile_events = ()

        if self.phase == GamePhase.READY:
            self.phase_timer -= dt
            if self.phase_timer <= 0:
                self.phase = GamePhase.ACTIVE
            return
        if self.phase == GamePhase.DYING:
            self.phase_timer -= dt
            if self.phase_timer <= 0 and self.lives > 0:
                self._reset_round_positions()
                self.phase = GamePhase.READY
                self.phase_timer = READY_SECONDS * 0.72
            return
        if self.phase == GamePhase.CLEARING:
            self.phase_timer -= dt
            if self.phase_timer <= 0:
                if self.auto_advance_on_clear:
                    self.next_level()
                else:
                    self.status = GameStatus.WON
                    self.high_score = max(self.high_score, self.score)
            return

        self.player_slow.update(dt)
        self.last_projectile_events = self._update_projectiles(dt)
        self.player.speed = PLAYER_SPEED * self.player_speed_multiplier
        if self.phase != GamePhase.ACTIVE:
            return

        frightened_before = self.frightened_timer
        self.frightened_timer = max(0.0, self.frightened_timer - dt)
        if frightened_before > 0 and self.frightened_timer == 0:
            self.ghost_chain = 0
        if self.frightened_timer == 0:
            self._update_ghost_mode(dt)

        # Reversals are responsive even half-way between two cells. Other turns
        # remain buffered until the next intersection.
        if self.player.target is not None and self.next_direction == self.player.direction.opposite:
            self.player.direction = self.next_direction
            self.player.target = self.player.grid
        if self.player.target is None:
            if self._can_move(self.player.grid, self.next_direction):
                self.player.direction = self.next_direction
            self._start_move(self.player, self.player.direction)
        if self._advance(self.player, dt):
            self._eat_current_cell()
        self._check_ghost_collisions()
        if self.phase != GamePhase.ACTIVE:
            return

        speed_scale = self.ghost_speed_multiplier
        for index, ghost in enumerate(self.ghosts):
            if not ghost.released:
                ghost.release_timer -= dt
                if ghost.release_timer <= 0:
                    ghost.released = True
                    ghost.direction = Direction.UP
                else:
                    continue
            if ghost.eaten:
                ghost.speed = EATEN_GHOST_SPEED
            elif self.frightened_timer > 0:
                ghost.speed = FRIGHTENED_SPEED * speed_scale
            else:
                ghost.speed = GHOST_SPEED * speed_scale
            if ghost.target is None:
                direction = self._choose_ghost_direction(ghost, index)
                ghost.direction = direction
                self._start_move(ghost, direction)
            arrived = self._advance(ghost, dt)
            if arrived and ghost.eaten and ghost.grid == ghost.spawn:
                ghost.eaten = False
                ghost.released = False
                ghost.release_timer = 0.65
                ghost.target = None
            self._check_ghost_collisions()
            if self.phase != GamePhase.ACTIVE:
                return
            if ghost.target is None:
                self._try_fire_projectile(ghost)

        if self._count_dots() == 0:
            self.phase = GamePhase.CLEARING
            self.phase_timer = CLEAR_SECONDS
            self.high_score = max(self.high_score, self.score)
            self._clear_combat_transients(reset_cooldowns=False)

    def _update_ghost_mode(self, dt: float) -> None:
        if math.isinf(self.mode_timer):
            return
        self.mode_timer -= dt
        if self.mode_timer > 0:
            return
        self.mode_index = min(self.mode_index + 1, len(MODE_SCHEDULE) - 1)
        self.ghost_mode, self.mode_timer = MODE_SCHEDULE[self.mode_index]
        for ghost in self.ghosts:
            if ghost.released and not ghost.eaten:
                ghost.pending_reverse = True

    def _cell(self, x: int, y: int) -> str:
        if y < 0 or y >= self.rows:
            return "#"
        if x < 0 or x >= self.cols:
            return " " if y == 9 else "#"
        return self.maze[y][x]

    def _can_move(self, grid: tuple[int, int], direction: Direction) -> bool:
        dx, dy = direction.vector
        return self._cell(grid[0] + dx, grid[1] + dy) != "#"

    def _start_move(self, mover: Mover, direction: Direction) -> None:
        if not self._can_move(mover.grid, direction):
            return
        dx, dy = direction.vector
        mover.target = (mover.grid_x + dx, mover.grid_y + dy)

    def _advance(self, mover: Mover, dt: float) -> bool:
        if mover.target is None:
            return False
        tx, ty = mover.target
        target_position = pygame.Vector2(tx * TILE_SIZE + TILE_SIZE / 2, ty * TILE_SIZE + TILE_SIZE / 2)
        delta = target_position - mover.position
        step = mover.speed * dt
        if delta.length() <= step:
            mover.position = target_position
            mover.grid_x, mover.grid_y = tx, ty
            if mover.grid_x < 0:
                mover.grid_x = self.cols - 1
                mover.position.x = mover.grid_x * TILE_SIZE + TILE_SIZE / 2
            elif mover.grid_x >= self.cols:
                mover.grid_x = 0
                mover.position.x = TILE_SIZE / 2
            mover.target = None
            return True
        mover.position += delta.normalize() * step
        return False

    def _projectile_next_cell(
        self,
        cell: tuple[int, int],
        direction: tuple[int, int],
    ) -> tuple[int, int]:
        """Advance one projectile tile while preserving the maze tunnel."""

        x, y = cell[0] + direction[0], cell[1] + direction[1]
        if y == 9:
            x %= self.cols
        return x, y

    def _projectile_cell_is_walkable(self, cell: tuple[int, int]) -> bool:
        return self._cell(*cell) != "#"

    def _projectile_hits_player(
        self,
        projectile: Projectile,
        _cell: tuple[int, int],
    ) -> bool:
        """Use interpolated sprite positions instead of stale grid ownership."""

        projectile_x, projectile_y = self._projectile_pixel_position(projectile)
        player_x = self.player.position.x
        if projectile.cell[1] == 9 and self.player.grid_y == 9:
            player_x %= self.maze_width
            horizontal = min(
                abs(projectile_x - player_x),
                self.maze_width - abs(projectile_x - player_x),
            )
        else:
            horizontal = abs(projectile_x - player_x)
        vertical = abs(projectile_y - self.player.position.y)
        radius = TILE_SIZE * (
            PLAYER_PROJECTILE_COLLISION_RADIUS_TILES + projectile.spec.radius_tiles
        )
        return horizontal * horizontal + vertical * vertical <= radius * radius

    def _try_fire_projectile(self, ghost: Ghost) -> None:
        suppressed = (
            self.phase != GamePhase.ACTIVE
            or not ghost.released
            or ghost.eaten
            or self.frightened_timer > 0
        )
        projectile = self.projectile_system.try_fire_at_target(
            ghost.name,
            self.level,
            ghost.grid,
            self.player.grid,
            self._projectile_cell_is_walkable,
            next_cell=self._projectile_next_cell,
            suppressed=suppressed,
        )
        if projectile is not None:
            self.projectile_shots_fired += 1

    def _update_projectiles(self, dt: float) -> tuple[ProjectileEvent, ...]:
        events = self.projectile_system.update(
            dt,
            None,
            self._projectile_cell_is_walkable,
            next_cell=self._projectile_next_cell,
            pacman_collision_test=self._projectile_hits_player,
        )
        processed = []
        for event in events:
            self._apply_projectile_event(event)
            processed.append(event)
            if self.phase != GamePhase.ACTIVE:
                break
        return tuple(processed)

    def _apply_projectile_event(self, event: ProjectileEvent) -> None:
        x, y = event.position_tiles
        pixel_x = (x + 0.5) * TILE_SIZE
        if event.cell[1] == 9:
            pixel_x %= self.maze_width
        position = pygame.Vector2(
            pixel_x,
            (y + 0.5) * TILE_SIZE,
        )
        self.projectile_impacts.append(ProjectileImpact(event.kind, position))
        if not event.hit_pacman:
            return

        if event.kind == ProjectileKind.FIREBALL and event.damage:
            self.fireball_hits += 1
            self.score_popups.append(
                ScorePopup("FIRE HIT", position, (255, 125, 55), lifetime=0.9)
            )
            self._lose_life()
        elif event.kind == ProjectileKind.FREEZE_BALL and self.player_slow.apply(event):
            self.freeze_ball_hits += 1
            self.player.speed = PLAYER_SPEED * self.player_speed_multiplier
            self.flash_timer = max(self.flash_timer, 0.18)
            self.score_popups.append(
                ScorePopup("FROZEN -15%", position, CYAN, lifetime=1.05)
            )

    def _clear_combat_transients(
        self,
        *,
        reset_cooldowns: bool,
        clear_impacts: bool = True,
    ) -> None:
        if reset_cooldowns:
            self.projectile_system.reset_level(
                initial_cooldown_seconds=PROJECTILE_READY_DELAY,
            )
        else:
            self.projectile_system.clear_projectiles()
        self.player_slow.clear()
        self.player.speed = PLAYER_SPEED
        self.last_projectile_events = ()
        if clear_impacts:
            self.projectile_impacts.clear()

    def projectile_threat_cells(self) -> set[tuple[int, int]]:
        """Return occupied and future ray cells for checkpoint-safe vision."""

        cells: set[tuple[int, int]] = set()
        for projectile in self.active_projectiles:
            cell = projectile.cell
            cells.add(cell)
            remaining = projectile.spec.range_tiles - projectile.tiles_travelled
            for _ in range(max(0, remaining)):
                cell = self._projectile_next_cell(cell, projectile.direction)
                if not self._projectile_cell_is_walkable(cell):
                    break
                cells.add(cell)
        return cells

    def projectile_telemetry(self) -> dict[str, object]:
        """Expose live projectile rules without leaking mutable game objects."""

        cooldowns = self.projectile_system.cooldowns
        weapons = {}
        for roll in self.projectile_system.unlocks.rolls:
            spec = self.projectile_system.spec_for(roll.owner)
            weapons[roll.owner] = {
                "kind": roll.kind.value,
                "unlocked": self.projectile_system.is_unlocked(roll.owner, self.level),
                "unlocked_early": roll.unlocked_early,
                "early_unlock_chance": roll.probability,
                "unlock_level": spec.unlock_level if spec is not None else 3,
                "range_tiles": spec.range_tiles if spec is not None else 0,
                "cooldown_seconds": cooldowns.get(roll.owner, 0.0),
            }
        active = []
        for projectile in self.active_projectiles:
            x, y = projectile.position_tiles
            active.append(
                {
                    "id": projectile.projectile_id,
                    "owner": projectile.owner,
                    "kind": projectile.kind.value,
                    "cell": projectile.cell,
                    "direction": projectile.direction,
                    "progress": projectile.progress,
                    "position_tiles": (x, y),
                    "tiles_travelled": projectile.tiles_travelled,
                    "remaining_range_tiles": projectile.remaining_range_tiles,
                    "range_tiles": projectile.spec.range_tiles,
                }
            )
        return {
            "weapons": weapons,
            "active": active,
            "active_count": len(active),
            "player_slowed": self.player_slow.active,
            "slow_fraction": self.player_slow.fraction,
            "slow_timer": self.player_slow.remaining_seconds,
            "shots_fired": self.projectile_shots_fired,
            "fireball_hits": self.fireball_hits,
            "freeze_ball_hits": self.freeze_ball_hits,
        }

    def _eat_current_cell(self) -> None:
        x, y = self.player.grid
        if not (0 <= x < self.cols and 0 <= y < self.rows):
            return
        cell = self.maze[y][x]
        if cell == ".":
            self.maze[y][x] = " "
            self._add_score(10)
        elif cell == "o":
            self.maze[y][x] = " "
            self._add_score(50, "POWER", CYAN)
            self.frightened_timer = self.frightened_duration
            self.ghost_chain = 0
            self.flash_timer = 0.22
            for ghost in self.ghosts:
                if ghost.released and not ghost.eaten:
                    ghost.pending_reverse = True

    def _add_score(
        self,
        points: int,
        label: str | None = None,
        color: tuple[int, int, int] = WHITE,
        position: pygame.Vector2 | None = None,
    ) -> None:
        old_score = self.score
        self.score += points
        self.high_score = max(self.high_score, self.score)
        if label:
            self.score_popups.append(
                ScorePopup(label, pygame.Vector2(position or self.player.position), color)
            )
        if not self.extra_life_awarded and old_score < EXTRA_LIFE_SCORE <= self.score:
            self.extra_life_awarded = True
            self.lives += 1
            self.score_popups.append(
                ScorePopup("EXTRA LIFE", pygame.Vector2(self.player.position), YELLOW, lifetime=1.2)
            )

    def _ghost_target(self, ghost: Ghost, index: int) -> tuple[int, int]:
        px, py = self.player.grid
        dx, dy = self.player.direction.vector
        if ghost.eaten:
            return ghost.spawn
        if self.ghost_mode == GhostMode.SCATTER:
            return SCATTER_TARGETS[index]
        if index == 0:  # direct chase
            return px, py
        if index == 1:  # ambush ahead
            return px + dx * 4, py + dy * 4
        if index == 2:  # flank relative to Blinky
            bx, by = self.ghosts[0].grid
            return px * 2 - bx + dx * 2, py * 2 - by + dy * 2
        distance = abs(ghost.grid_x - px) + abs(ghost.grid_y - py)
        return (0, self.rows - 1) if distance < 7 else (px, py)

    def _choose_ghost_direction(self, ghost: Ghost, index: int) -> Direction:
        available = [direction for direction in Direction if self._can_move(ghost.grid, direction)]
        if ghost.pending_reverse and ghost.direction.opposite in available:
            ghost.pending_reverse = False
            return ghost.direction.opposite
        non_reverse = [direction for direction in available if direction != ghost.direction.opposite]
        choices = non_reverse or available or [ghost.direction.opposite]
        if self.frightened_timer > 0 and not ghost.eaten:
            return self.rng.choice(choices)
        target_x, target_y = self._ghost_target(ghost, index)
        priority = {Direction.UP: 0, Direction.LEFT: 1, Direction.DOWN: 2, Direction.RIGHT: 3}
        return min(
            choices,
            key=lambda direction: (
                abs(ghost.grid_x + direction.vector[0] - target_x)
                + abs(ghost.grid_y + direction.vector[1] - target_y),
                priority[direction],
            ),
        )

    def _check_ghost_collisions(self) -> None:
        for ghost in self.ghosts:
            if self.player.position.distance_to(ghost.position) >= TILE_SIZE * 0.62:
                continue
            if self.frightened_timer > 0 and not ghost.eaten:
                self.ghost_chain += 1
                points = min(1600, 200 * (2 ** (self.ghost_chain - 1)))
                self._add_score(points, str(points), CYAN, ghost.position)
                ghost.eaten = True
                ghost.pending_reverse = False
                ghost.speed = EATEN_GHOST_SPEED
                self.flash_timer = 0.12
            elif not ghost.eaten:
                self._lose_life()
                return

    def _lose_life(self) -> None:
        self.lives -= 1
        self.high_score = max(self.high_score, self.score)
        self.frightened_timer = 0.0
        self.ghost_chain = 0
        self._clear_combat_transients(
            reset_cooldowns=False,
            clear_impacts=False,
        )
        self.phase = GamePhase.DYING
        self.phase_timer = DEATH_SECONDS
        self.flash_timer = 0.28
        if self.lives <= 0:
            self.status = GameStatus.LOST
            return

    def _reset_round_positions(self) -> None:
        self._clear_combat_transients(reset_cooldowns=True)
        self.player.reset_position((9, 15), Direction.LEFT)
        self.next_direction = Direction.LEFT
        for index, ghost in enumerate(self.ghosts):
            ghost.eaten = False
            ghost.released = index == 0
            ghost.release_timer = RELEASE_DELAYS[index]
            ghost.pending_reverse = False
            ghost.reset_position(ghost.spawn, Direction.LEFT if index % 2 == 0 else Direction.RIGHT)

    def restart(self) -> None:
        self.projectile_system.start_new_run()
        self.projectile_shots_fired = 0
        self.fireball_hits = 0
        self.freeze_ball_hits = 0
        self.maze = [list(row) for row in MAZE_TEMPLATE]
        self.score = 0
        self.lives = STARTING_LIVES
        self.level = 1
        self.status = GameStatus.PLAYING
        self.phase = GamePhase.READY
        self.phase_timer = READY_SECONDS
        self.frightened_timer = 0.0
        self.mode_index = 0
        self.ghost_mode, self.mode_timer = MODE_SCHEDULE[0]
        self.extra_life_awarded = False
        self.score_popups.clear()
        self._reset_round_positions()
        self._eat_current_cell()

    def next_level(self) -> None:
        """Continue from a cleared maze while preserving score and lives."""
        self.level += 1
        self.maze = [list(row) for row in MAZE_TEMPLATE]
        self.status = GameStatus.PLAYING
        self.phase = GamePhase.READY
        self.phase_timer = READY_SECONDS
        self.frightened_timer = 0.0
        self.mode_index = 0
        self.ghost_mode, self.mode_timer = MODE_SCHEDULE[0]
        self._reset_round_positions()
        self._eat_current_cell()

    def _render(self, dt: float) -> None:
        self.display.fill(BLACK)
        self._draw_maze()
        self._draw_projectiles()
        pygame.draw.ellipse(
            self.display, (0, 0, 0, 105),
            (self.player.position.x - 10, self.player.position.y + 6, 20, 7),
        )
        self.player.sprite.set_direction(self.player.direction)
        death_progress = None
        if self.phase == GamePhase.DYING:
            death_progress = 1 - max(0.0, self.phase_timer) / DEATH_SECONDS
        self.player.sprite.draw(self.display, self.player.position, dt, death_progress)
        if self.phase != GamePhase.DYING:
            for ghost in self.ghosts:
                self._draw_armed_ghost_aura(ghost)
                pygame.draw.ellipse(
                    self.display, (0, 0, 0, 90),
                    (ghost.position.x - 10, ghost.position.y + 7, 20, 6),
                )
                frightened = self.frightened_timer > 0 and not ghost.eaten
                ghost.sprite.set_state(
                    ghost.direction, frightened, 0 < self.frightened_timer < 2, ghost.eaten
                )
                ghost.sprite.draw(self.display, ghost.position, dt)
        self._draw_score_popups()
        self._draw_hud()
        if self.status != GameStatus.PLAYING or self.phase == GamePhase.READY:
            self._draw_overlay()
        if self.flash_timer > 0:
            flash = pygame.Surface((self.maze_width, self.maze_height), pygame.SRCALPHA)
            alpha = min(90, round(90 * self.flash_timer / 0.28))
            flash.fill((100, 175, 255, alpha))
            self.display.blit(flash, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        if self.render_enabled:
            pygame.display.flip()

    def _projectile_pixel_position(self, projectile: Projectile) -> tuple[float, float]:
        x, y = projectile.position_tiles
        pixel_x = (x + 0.5) * TILE_SIZE
        if projectile.cell[1] == 9:
            pixel_x %= self.maze_width
        return pixel_x, (y + 0.5) * TILE_SIZE

    def _draw_projectiles(self) -> None:
        for projectile in self.active_projectiles:
            sprite = self.projectile_sprites[projectile.kind]
            sprite.draw(
                self.display,
                self._projectile_pixel_position(projectile),
                self.animation_time,
            )
        for impact in self.projectile_impacts:
            progress = impact.age / impact.lifetime
            self.projectile_sprites[impact.kind].draw_impact(
                self.display,
                impact.position,
                progress,
            )

    def _draw_armed_ghost_aura(self, ghost: Ghost) -> None:
        if (
            ghost.eaten
            or self.frightened_timer > 0
            or not self.projectile_system.is_unlocked(ghost.name, self.level)
        ):
            return
        spec = self.projectile_system.spec_for(ghost.name)
        if spec is None:
            return
        phase_offset = 0 if ghost.name == "BLINKY" else 1.7
        pulse = (math.sin(self.animation_time * 5.5 + phase_offset) + 1) / 2
        radius = round(TILE_SIZE * (0.54 + pulse * 0.06))
        pygame.draw.circle(
            self.display,
            spec.glow_color,
            (round(ghost.position.x), round(ghost.position.y)),
            radius,
            1,
        )
        angle = self.animation_time * (3.4 if ghost.name == "BLINKY" else -2.7)
        particle = (
            round(ghost.position.x + math.cos(angle) * radius),
            round(ghost.position.y + math.sin(angle) * radius),
        )
        pygame.draw.circle(self.display, spec.color, particle, 2)

    def _build_maze_surface(self) -> pygame.Surface:
        surface = pygame.Surface((self.maze_width, self.maze_height))
        surface.fill(BLACK)
        contour_inset = 2
        contour_paths: list[tuple[tuple[int, int], tuple[int, int]]] = []

        def is_wall(x: int, y: int) -> bool:
            if not (0 <= x < self.cols and 0 <= y < self.rows):
                return True
            return MAZE_TEMPLATE[y][x] == "#"

        for row, cells in enumerate(MAZE_TEMPLATE):
            for col, cell in enumerate(cells):
                if cell != "#":
                    continue
                x, y = col * TILE_SIZE, row * TILE_SIZE
                pygame.draw.rect(surface, WALL_FILL, (x, y, TILE_SIZE, TILE_SIZE))
                edges = (
                    (
                        not is_wall(col, row - 1),
                        (x, y + contour_inset),
                        (x + TILE_SIZE, y + contour_inset),
                    ),
                    (
                        not is_wall(col, row + 1),
                        (x, y + TILE_SIZE - contour_inset),
                        (x + TILE_SIZE, y + TILE_SIZE - contour_inset),
                    ),
                    (
                        not is_wall(col - 1, row),
                        (x + contour_inset, y),
                        (x + contour_inset, y + TILE_SIZE),
                    ),
                    (
                        not is_wall(col + 1, row),
                        (x + TILE_SIZE - contour_inset, y),
                        (x + TILE_SIZE - contour_inset, y + TILE_SIZE),
                    ),
                )
                for visible, start, end in edges:
                    if visible:
                        contour_paths.append((start, end))

        # A three-wall vertex has one open quadrant. Its two exposed edges stop
        # short of one another because both are inset, so route an L-shaped
        # join through the opposite (wall) quadrant. Vertices with two diagonal
        # walls are intentionally left alone: joining those would visually seal
        # two corridors that only meet at a point.
        for vertex_y in range(1, self.rows):
            for vertex_x in range(1, self.cols):
                x, y = vertex_x * TILE_SIZE, vertex_y * TILE_SIZE
                quadrants = (
                    (is_wall(vertex_x - 1, vertex_y - 1), -1, -1),
                    (is_wall(vertex_x, vertex_y - 1), 1, -1),
                    (is_wall(vertex_x - 1, vertex_y), -1, 1),
                    (is_wall(vertex_x, vertex_y), 1, 1),
                )
                if sum(wall for wall, _, _ in quadrants) != 3:
                    continue
                _, floor_dx, floor_dy = next(
                    quadrant for quadrant in quadrants if not quadrant[0]
                )
                join = (
                    x - floor_dx * contour_inset,
                    y - floor_dy * contour_inset,
                )
                contour_paths.extend(
                    (
                        ((join[0], y), join),
                        (join, (x, join[1])),
                    )
                )

        # Paint in layers so a later glow never washes over an already crisp
        # blue contour at intersections and tightly packed corners.
        for start, end in contour_paths:
            pygame.draw.line(surface, WALL_GLOW, start, end, 6)
        for start, end in contour_paths:
            pygame.draw.line(surface, WALL_BLUE, start, end, 2)

        # The ghost-house gate is visual only; the movement graph deliberately
        # lets returning eyes pass through it.
        gate_y = 8 * TILE_SIZE + 1
        pygame.draw.line(surface, (116, 45, 112), (9 * TILE_SIZE, gate_y), (11 * TILE_SIZE, gate_y), 5)
        pygame.draw.line(surface, PINK, (9 * TILE_SIZE, gate_y), (11 * TILE_SIZE, gate_y), 2)
        return surface

    def _draw_maze(self) -> None:
        self.display.blit(self.maze_surface, (0, 0))
        for row, cells in enumerate(self.maze):
            for col, cell in enumerate(cells):
                x, y = col * TILE_SIZE, row * TILE_SIZE
                if cell == ".":
                    pygame.draw.circle(self.display, PELLET, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), 2)
                elif cell == "o":
                    pulse = 5 + round((math.sin(self.animation_time * 8) + 1) * 0.8)
                    pygame.draw.circle(
                        self.display, (87, 63, 82),
                        (x + TILE_SIZE // 2, y + TILE_SIZE // 2), pulse + 3,
                    )
                    pygame.draw.circle(self.display, PELLET, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), pulse)

        if self.phase == GamePhase.CLEARING and int(self.animation_time * 9) % 2:
            wash = pygame.Surface((self.maze_width, self.maze_height), pygame.SRCALPHA)
            wash.fill((255, 255, 255, 34))
            self.display.blit(wash, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_score_popups(self) -> None:
        for popup in self.score_popups:
            alpha = max(0, round(255 * (1 - popup.age / popup.lifetime)))
            image = self.small_font.render(popup.text, True, popup.color)
            image.set_alpha(alpha)
            self.display.blit(image, image.get_rect(center=(round(popup.position.x), round(popup.position.y))))

    def _draw_hud(self) -> None:
        y = self.maze_height
        pygame.draw.rect(self.display, HUD_BG, (0, y, self.maze_width, HUD_HEIGHT))
        pygame.draw.line(self.display, WALL_BLUE, (0, y), (self.maze_width, y), 2)
        score = self.font.render(f"SCORE  {self.score:05d}", True, WHITE)
        high = self.font.render(f"HIGH  {max(self.high_score, self.score):05d}", True, WHITE)
        dots = self.small_font.render(f"PELLETS  {self._count_dots():03}/{self.total_dots}", True, MUTED)
        level = self.small_font.render(f"LEVEL  {self.level:02}", True, MUTED)
        self.display.blit(score, (14, y + 12))
        self.display.blit(high, (174, y + 12))
        self.display.blit(dots, (14, y + 43))
        self.display.blit(level, (174, y + 43))
        if self.player_slow.active:
            slowed = self.tiny_font.render(
                f"FROZEN -15%  {self.player_slow.remaining_seconds:0.1f}s",
                True,
                CYAN,
            )
            self.display.blit(slowed, (258, y + 44))
        for life in range(self.lives):
            pygame.draw.circle(self.display, YELLOW, (self.maze_width - 20 - life * 25, y + 24), 8)
            pygame.draw.polygon(self.display, HUD_BG, [(self.maze_width - 20 - life * 25, y + 24),
                                                       (self.maze_width - 10 - life * 25, y + 18),
                                                       (self.maze_width - 10 - life * 25, y + 30)])
        if self.frightened_timer > 0:
            power = self.tiny_font.render(f"POWER  {self.frightened_timer:0.1f}s", True, CYAN)
            self.display.blit(power, (self.maze_width - 122, y + 48))
            bar = pygame.Rect(self.maze_width - 122, y + 68, 108, 5)
            pygame.draw.rect(self.display, (29, 40, 68), bar, border_radius=2)
            fill = bar.copy()
            fill.width = round(bar.width * self.frightened_timer / self.frightened_duration)
            pygame.draw.rect(self.display, CYAN, fill, border_radius=2)
        else:
            mode = self.tiny_font.render(self.ghost_mode.name, True, MUTED)
            self.display.blit(mode, (self.maze_width - 75, y + 51))

        for x, owner, label in ((14, "BLINKY", "FIRE"), (174, "INKY", "FREEZE")):
            spec = self.projectile_system.spec_for(owner)
            unlocked = self.projectile_system.is_unlocked(owner, self.level)
            cooldown = self.projectile_system.seconds_until_ready(owner)
            if unlocked and cooldown <= 0:
                suffix = "READY"
            elif unlocked:
                suffix = f"{cooldown:0.1f}s"
            else:
                suffix = "LOCKED"
            color = spec.color if spec is not None and unlocked else MUTED
            weapon = self.tiny_font.render(f"{label}  {suffix}", True, color)
            self.display.blit(weapon, (x, y + 69))

    def _draw_overlay(self) -> None:
        overlay = pygame.Surface((self.maze_width, 92), pygame.SRCALPHA)
        overlay.fill((5, 7, 18, 225))
        y = self.maze_height // 2 - 46
        self.display.blit(overlay, (0, y))
        if self.status == GameStatus.PLAYING and self.phase == GamePhase.READY:
            title, subtitle = "READY!", f"LEVEL {self.level}"
        else:
            title = {
                GameStatus.PAUSED: "PAUSED",
                GameStatus.WON: "MAZE CLEARED",
                GameStatus.LOST: "GAME OVER",
            }[self.status]
            if self.status == GameStatus.PAUSED:
                subtitle = "Space to continue"
            elif self.status == GameStatus.WON:
                subtitle = "Enter: next maze   R: new run"
            else:
                subtitle = "Press R to restart"
        title_surface = self.font.render(title, True, YELLOW)
        sub_surface = self.small_font.render(subtitle, True, WHITE)
        self.display.blit(title_surface, title_surface.get_rect(center=(self.maze_width // 2, y + 32)))
        self.display.blit(sub_surface, sub_surface.get_rect(center=(self.maze_width // 2, y + 62)))

    def save_screenshot(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._render(0.016)
        pygame.image.save(self.display, str(output))
        return output
