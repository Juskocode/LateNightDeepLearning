import pygame
import random
from .constants import *
from .pacmanSprite import PacmanSprite

pygame.init()
font = pygame.font.Font(None, 25)


class PacmanGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Pacman')
        self.clock = pygame.time.Clock()

        # init game state
        self.direction = Direction.RIGHT
        self.pacman = Point(self.w / 2, self.h / 2)

        # Initialize Pacman sprite
        self.pacman_sprite = PacmanSprite(PACMAN_SPRITE_PATH, (BLOCK_SIZE, BLOCK_SIZE))

        self.score = 0
        self.food = None
        self.special_food = None

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # 2. move
        self._move(self.direction)

        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 4. place new food or just move
        if self.pacman == self.food:
            self.score += 1

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score

    def _is_collision(self):
        # hits boundary
        if (self.pacman.x > self.w - BLOCK_SIZE or self.pacman.x < 0 or
                self.pacman.y > self.h - BLOCK_SIZE or self.pacman.y < 0):
            return True
        # hits ghost
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # Update sprite direction and draw Pacman
        self.pacman_sprite.set_direction(self.direction)
        self.pacman_sprite.draw(self.display, (self.pacman.x, self.pacman.y))

        # Draw score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        x = self.pacman.x
        y = self.pacman.y
        if direction == Direction.RIGHT:
            x += PACMAN_SPEED
        elif direction == Direction.LEFT:
            x -= PACMAN_SPEED
        elif direction == Direction.DOWN:
            y += PACMAN_SPEED
        elif direction == Direction.UP:
            y -= PACMAN_SPEED

        self.pacman = Point(x, y)
