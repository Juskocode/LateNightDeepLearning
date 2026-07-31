"""Shared Pacman constants and immutable game types."""

from collections import namedtuple
from enum import Enum, auto
from pathlib import Path


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def vector(self):
        return self.value

    @property
    def opposite(self):
        return {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }[self]


class GameStatus(Enum):
    PLAYING = auto()
    PAUSED = auto()
    WON = auto()
    LOST = auto()


class GamePhase(Enum):
    """Short-lived round phases kept separate from pause/win/loss status."""

    READY = auto()
    ACTIVE = auto()
    DYING = auto()
    CLEARING = auto()


class GhostMode(Enum):
    SCATTER = auto()
    CHASE = auto()


Point = namedtuple("Point", "x y")

TILE_SIZE = 24
FPS = 60
PLAYER_SPEED = 144.0
GHOST_SPEED = 108.0
FRIGHTENED_SPEED = 75.0
EATEN_GHOST_SPEED = 192.0
GHOST_SPEED_INCREASE_PER_LEVEL = 0.01
MAX_GHOST_SPEED_MULTIPLIER = 1.20
HUD_HEIGHT = 92
STARTING_LIVES = 3
FRIGHTENED_SECONDS = 7.0
FRIGHTENED_DECREASE_PER_LEVEL = 0.1
MIN_FRIGHTENED_SECONDS = 4.5
READY_SECONDS = 1.35
DEATH_SECONDS = 1.25
CLEAR_SECONDS = 1.1
EXTRA_LIFE_SCORE = 10_000

BLACK = (5, 7, 18)
WHITE = (241, 244, 255)
YELLOW = (255, 214, 40)
WALL_BLUE = (42, 83, 255)
WALL_GLOW = (21, 38, 112)
WALL_FILL = (9, 17, 52)
PELLET = (255, 205, 181)
HUD_BG = (13, 17, 36)
MUTED = (135, 145, 180)
CYAN = (76, 224, 255)
PINK = (255, 132, 194)

REPO_ROOT = Path(__file__).resolve().parents[3]
FONT_PATH = REPO_ROOT / "assets" / "fonts" / "arial.ttf"
PACMAN_SPRITE_PATH = REPO_ROOT / "assets" / "sprites" / "Pacman.svg"
