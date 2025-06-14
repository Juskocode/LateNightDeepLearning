import pygame
import random
import numpy as np
from typing import Tuple, List, Optional

from .constants import Direction, Point, FOOD_REWARD, COLLISION_PENALTY, FRAME_TIMEOUT_MULTIPLIER
from src.config.settings import (
    GAME_WIDTH, GAME_HEIGHT, BLOCK_SIZE, GAME_SPEED,
    WHITE, RED, BLUE1, BLUE2, BLACK, FONT_PATH
)

pygame.init()
try:
    font = pygame.font.Font(FONT_PATH, 25)
except:
    font = pygame.font.SysFont('arial', 25)


class SnakeGameAI:

    def __init__(self, width: int = GAME_WIDTH, height: int = GAME_HEIGHT):
        self.width = width
        self.height = height

        # Initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()

        # Game state
        self.score = 0
        self.best_score = 0
        self.reset()

    def reset(self) -> None:
        # Initialize snake
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        # Update scores
        self.best_score = max(self.score, self.best_score)
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self) -> None:
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def play_step(self, action: List[int]) -> Tuple[int, bool, int]:
        self.frame_iteration += 1

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move snake
        self._move(action)
        self.snake.insert(0, self.head)

        # Check game over conditions
        reward = 0
        game_over = False

        if (self.is_collision() or
                self.frame_iteration > FRAME_TIMEOUT_MULTIPLIER * len(self.snake)):
            game_over = True
            reward = COLLISION_PENALTY
            return reward, game_over, self.score

        # Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = FOOD_REWARD
            self._place_food()
        else:
            self.snake.pop()

        # Update display
        self._update_ui()
        self.clock.tick(GAME_SPEED)

        return reward, game_over, self.score

    def is_collision(self, point: Optional[Point] = None) -> bool:
        if point is None:
            point = self.head

        # Check boundary collision
        if (point.x > self.width - BLOCK_SIZE or point.x < 0 or
                point.y > self.height - BLOCK_SIZE or point.y < 0):
            return True

        # Check self collision
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self) -> None:
        self.display.fill(BLACK)

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1,
                             pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        best_score_text = font.render(f"Best: {self.best_score}", True, WHITE)
        self.display.blit(score_text, [0, 0])
        self.display.blit(best_score_text, [self.width - 150, 0])

        pygame.display.flip()

    def _move(self, action: List[int]) -> None:
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
