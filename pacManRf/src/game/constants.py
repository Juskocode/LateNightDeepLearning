
from enum import Enum
from collections import namedtuple

# Game constants
BLOCK_SIZE = 20
PACMAN_SPEED = 10
SPEED = 10

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Direction enum
class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

# Point for coordinates
Point = namedtuple('Point', 'x, y')

# Font paths
FONT_PATH = "../../../assets/fonts/arial.ttf"

# Sprite paths
PACMAN_SPRITE_PATH = "../../assets/sprites/Pacman.svg"