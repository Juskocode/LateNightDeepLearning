
"""Configuration settings for the Snake AI project."""

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
LEARNING_RATE = 0.001
GAMMA = 0.9  # Discount rate
EPSILON_DECAY = 80  # For exploration-exploitation tradeoff

# Model settings
INPUT_SIZE = 11
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

# File paths
MODEL_DIR = "models/saved_models"
FONT_PATH = "assets/arial.ttf"