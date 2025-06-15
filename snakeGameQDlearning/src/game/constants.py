from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Game mechanics
FRAME_TIMEOUT_MULTIPLIER = 100  # Max frames = 100 * snake_length
FOOD_REWARD = 10
COLLISION_PENALTY = -10