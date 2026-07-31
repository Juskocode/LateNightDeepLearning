from collections import namedtuple
from enum import Enum

class Direction(Enum):
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    UP = (0, -1)
    DOWN = (0, 1)

Point = namedtuple('Point', 'x, y')

# Game mechanics
FRAME_TIMEOUT_MULTIPLIER = 100  # Starvation budget = multiplier * snake length.
