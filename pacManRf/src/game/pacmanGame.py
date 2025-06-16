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

        # Create the maze layout (1 = wall, 0 = empty/path, 2 = dot, 3 = tunnel)
        self.maze = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1],
            [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1],
            [3, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 3],
            [1, 1, 1, 1, 2, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 1, 1, 1, 1],
            [0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0],
            [1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1],
            [3, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 3],
            [1, 1, 1, 1, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1],
            [1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1],
            [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1],
            [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
            [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]

        # Calculate block size and adjust screen
        self.block_size = 25
        self.maze_width = len(self.maze[0]) * self.block_size
        self.maze_height = len(self.maze) * self.block_size

        # Adjust display size to accommodate maze + score area
        self.display_height = self.maze_height + 60  # Extra space for score
        self.display = pygame.display.set_mode((self.maze_width, self.display_height))
        pygame.display.set_caption('Pacman')
        self.clock = pygame.time.Clock()

        # init game state
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT  # Store the intended direction

        # Start Pacman in grid-aligned position
        self.pacman_grid_x = 9
        self.pacman_grid_y = 15
        self.pacman = Point(self.pacman_grid_x * self.block_size + self.block_size // 2,
                            self.pacman_grid_y * self.block_size + self.block_size // 2)

        # Movement tracking
        self.move_progress = 0  # Progress towards next grid cell (0.0 to 1.0)
        self.is_moving = False
        self.target_x = self.pacman.x
        self.target_y = self.pacman.y

        # Initialize Pacman sprite
        self.pacman_sprite = PacmanSprite(PACMAN_SPRITE_PATH, (self.block_size, self.block_size))

        self.score = 0
        self.total_dots = self._count_dots()

    def _count_dots(self):
        """Count total dots in the maze"""
        count = 0
        for row in self.maze:
            for cell in row:
                if cell == 2:
                    count += 1
        return count

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.next_direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.next_direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.next_direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.next_direction = Direction.DOWN

        # 2. move
        self._move()

        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 4. check if pacman ate a dot
        self._check_dot_collision()

        # 5. check if all dots are eaten (win condition)
        if self._count_dots() == 0:
            game_over = True

        # 6. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 7. return game over and score
        return game_over, self.score

    def _can_move_in_direction(self, grid_x, grid_y, direction):
        """Check if movement in a direction is possible from given grid position"""
        next_x, next_y = grid_x, grid_y

        if direction == Direction.RIGHT:
            next_x += 1
        elif direction == Direction.LEFT:
            next_x -= 1
        elif direction == Direction.DOWN:
            next_y += 1
        elif direction == Direction.UP:
            next_y -= 1

        # Handle tunnel exits
        if next_x < 0 or next_x >= len(self.maze[0]):
            return True  # Allow tunnel movement

        # Check bounds
        if next_y < 0 or next_y >= len(self.maze):
            return False

        # Check if next cell is not a wall
        return self.maze[next_y][next_x] != 1

    def _is_collision(self):
        """Check collision with walls"""
        # Check bounds
        if (self.pacman_grid_y < 0 or self.pacman_grid_y >= len(self.maze) or
                self.pacman_grid_x < 0 or self.pacman_grid_x >= len(self.maze[0])):
            return False  # Allow for tunnel movement

        # Check wall collision
        if self.maze[self.pacman_grid_y][self.pacman_grid_x] == 1:
            return True

        return False

    def _check_dot_collision(self):
        """Check if Pacman ate a dot"""
        if (0 <= self.pacman_grid_x < len(self.maze[0]) and
                0 <= self.pacman_grid_y < len(self.maze) and
                self.maze[self.pacman_grid_y][self.pacman_grid_x] == 2):
            self.maze[self.pacman_grid_y][self.pacman_grid_x] = 0  # Remove the dot
            self.score += 10

    def _handle_tunnel(self):
        """Handle tunnel teleportation"""
        # Left tunnel exit (teleport to right)
        if self.pacman_grid_x < 0:
            self.pacman_grid_x = len(self.maze[0]) - 1
            self.pacman = Point(self.pacman_grid_x * self.block_size + self.block_size // 2, self.pacman.y)
        # Right tunnel exit (teleport to left)
        elif self.pacman_grid_x >= len(self.maze[0]):
            self.pacman_grid_x = 0
            self.pacman = Point(self.pacman_grid_x * self.block_size + self.block_size // 2, self.pacman.y)

    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw the maze
        self._draw_maze()

        # Update sprite direction and draw Pacman
        self.pacman_sprite.set_direction(self.direction)
        self.pacman_sprite.draw(self.display,
                                (self.pacman.x - self.block_size // 2, self.pacman.y - self.block_size // 2))

        # Draw score in the bottom area
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [10, self.maze_height + 10])

        # Draw remaining dots counter
        remaining_text = font.render("Dots: " + str(self._count_dots()), True, WHITE)
        self.display.blit(remaining_text, [200, self.maze_height + 10])

        pygame.display.flip()

    def _draw_maze(self):
        """Draw the maze with styled walls and dots"""
        for row_idx, row in enumerate(self.maze):
            for col_idx, cell in enumerate(row):
                x = col_idx * self.block_size
                y = row_idx * self.block_size

                if cell == 1:  # Wall
                    # Check if it's an outer wall or inner wall
                    is_outer_wall = (row_idx == 0 or row_idx == len(self.maze) - 1 or
                                     col_idx == 0 or col_idx == len(row) - 1)

                    if is_outer_wall:
                        # Outer walls - thinner blue border
                        pygame.draw.rect(self.display, BLUE,
                                         (x, y, self.block_size, self.block_size))
                        pygame.draw.rect(self.display, BLACK,
                                         (x + 2, y + 2, self.block_size - 4, self.block_size - 4))
                    else:
                        # Inner walls - thicker blue border
                        pygame.draw.rect(self.display, BLUE,
                                         (x, y, self.block_size, self.block_size))
                        pygame.draw.rect(self.display, BLACK,
                                         (x + 3, y + 3, self.block_size - 6, self.block_size - 6))

                elif cell == 2:  # Dot
                    dot_x = x + self.block_size // 2
                    dot_y = y + self.block_size // 2
                    pygame.draw.circle(self.display, YELLOW,
                                       (dot_x, dot_y), 3)

                elif cell == 3:  # Tunnel entrance - visual indicator
                    pygame.draw.rect(self.display, BLUE,
                                     (x, y - 5, self.block_size, self.block_size + 10))

    def _move(self):
        """Handle grid-based movement like original Pac-Man"""
        # If we're not moving, check if we can start moving
        if not self.is_moving:
            # Check if we can change direction
            if (self.next_direction != self.direction and
                    self._can_move_in_direction(self.pacman_grid_x, self.pacman_grid_y, self.next_direction)):
                self.direction = self.next_direction

            # Check if we can move in current direction
            if self._can_move_in_direction(self.pacman_grid_x, self.pacman_grid_y, self.direction):
                self.is_moving = True
                self.move_progress = 0

                # Set target position
                if self.direction == Direction.RIGHT:
                    self.target_x = (self.pacman_grid_x + 1) * self.block_size + self.block_size // 2
                    self.target_y = self.pacman.y
                elif self.direction == Direction.LEFT:
                    self.target_x = (self.pacman_grid_x - 1) * self.block_size + self.block_size // 2
                    self.target_y = self.pacman.y
                elif self.direction == Direction.DOWN:
                    self.target_x = self.pacman.x
                    self.target_y = (self.pacman_grid_y + 1) * self.block_size + self.block_size // 2
                elif self.direction == Direction.UP:
                    self.target_x = self.pacman.x
                    self.target_y = (self.pacman_grid_y - 1) * self.block_size + self.block_size // 2

        # If we're moving, update position
        if self.is_moving:
            self.move_progress += 0.33333334  # Movement speed

            if self.move_progress >= 1.0:
                # Reached target - update grid position
                self.move_progress = 0
                self.is_moving = False

                if self.direction == Direction.RIGHT:
                    self.pacman_grid_x += 1
                elif self.direction == Direction.LEFT:
                    self.pacman_grid_x -= 1
                elif self.direction == Direction.DOWN:
                    self.pacman_grid_y += 1
                elif self.direction == Direction.UP:
                    self.pacman_grid_y -= 1

                # Handle tunnel teleportation
                self._handle_tunnel()

                # Update actual position to grid-aligned
                self.pacman = Point(self.pacman_grid_x * self.block_size + self.block_size // 2,
                                    self.pacman_grid_y * self.block_size + self.block_size // 2)
            else:
                # Interpolate position
                start_x = self.pacman.x
                start_y = self.pacman.y

                new_x = start_x + (self.target_x - start_x) * self.move_progress
                new_y = start_y + (self.target_y - start_y) * self.move_progress

                self.pacman = Point(new_x, new_y)