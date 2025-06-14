
# Game settings
GAME_WIDTH = 640
GAME_HEIGHT = 480
BLOCK_SIZE = 20
GAME_SPEED = 16000

# Colors (RGB)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# AI Training settings
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.0005
GAMMA = 0.99  # Higher discount for long-term thinking
EPSILON_DECAY = 100
EPSILON_MIN = 10

# Model settings
INPUT_SIZE = 11
HIDDEN_SIZE = 512  # Increased capacity
OUTPUT_SIZE = 3

# Reward settings
FOOD_REWARD = 10
COLLISION_PENALTY = -10
CLOSER_TO_FOOD_REWARD = 1
FARTHER_FROM_FOOD_PENALTY = -1
LOOP_PENALTY = -11

# File paths
MODEL_DIR = "models/saved_models"
FONT_PATH = "assets/arial.ttf"