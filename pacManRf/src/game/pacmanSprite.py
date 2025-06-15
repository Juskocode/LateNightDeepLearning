import pygame
import math
from enum import Enum
from typing import Tuple
from .constants import Direction

class PacmanSprite:
    def __init__(self, sprite_path: str, size: Tuple[int, int] = (20, 20)):
        # Load and scale the original sprite
        self.original_sprite = pygame.image.load(sprite_path).convert_alpha()
        self.original_sprite = pygame.transform.scale(self.original_sprite, size)

        # Animation properties
        self.animation_frame = 0
        self.animation_speed = 2  # aka pacman speed for each step changes mouth angle
        self.max_mouth_angle = 90  # Maximum mouth opening angle

        # Direction sprites cache
        self.direction_sprites = {}
        self._create_direction_sprites()

        # Current state
        self.current_direction = Direction.RIGHT

    def _create_direction_sprites(self):
        # RIGHT is the original sprite (facing right)
        self.direction_sprites[Direction.RIGHT] = self.original_sprite

        # LEFT - flip horizontally
        self.direction_sprites[Direction.LEFT] = pygame.transform.flip(self.original_sprite, True, False)

        # UP - rotate 90 degrees counterclockwise
        self.direction_sprites[Direction.UP] = pygame.transform.rotate(self.original_sprite, 90)

        # DOWN - rotate 90 degrees clockwise
        self.direction_sprites[Direction.DOWN] = pygame.transform.rotate(self.original_sprite, -90)

    def _create_mouth_animation(self, base_sprite: pygame.Surface, mouth_angle: float) -> pygame.Surface:
        """
        Create mouth animation by drawing a black triangle to simulate mouth opening

        Args:
            base_sprite: The base sprite to animate
            mouth_angle: Current mouth opening angle (0-max_mouth_angle)

        Returns:
            Animated sprite surface
        """
        if mouth_angle <= 0:
            return base_sprite

        # Create a copy of the sprite
        animated_sprite = base_sprite.copy()

        # Get sprite dimensions
        width, height = animated_sprite.get_size()
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 2

        # Calculate mouth triangle points based on direction
        if self.current_direction == Direction.LEFT:
            # Mouth opens to the right
            mouth_tip = (width - 1, center_y)
            angle_rad = math.radians(mouth_angle / 2)
            mouth_top = (center_x - 10, center_y + radius * math.sin(angle_rad))
            mouth_bottom = (center_x - 10, center_y - radius * math.sin(angle_rad))

        elif self.current_direction == Direction.RIGHT:
            # Mouth opens to the left
            mouth_tip = (0, center_y)
            angle_rad = math.radians(mouth_angle / 2)
            mouth_top = (center_x + 10, center_y - radius * math.sin(angle_rad))
            mouth_bottom = (center_x + 10, center_y + radius * math.sin(angle_rad))

        elif self.current_direction == Direction.DOWN:
            # Mouth opens upward
            mouth_tip = (center_x, 0)
            angle_rad = math.radians(mouth_angle / 2)
            mouth_left = (center_x - radius * math.sin(angle_rad), center_y + 10)
            mouth_right = (center_x + radius * math.sin(angle_rad), center_y + 10)
            mouth_top = mouth_left
            mouth_bottom = mouth_right

        else:  # DOWN
            # Mouth opens downward
            mouth_tip = (center_x, height - 1)
            angle_rad = math.radians(mouth_angle / 2)
            mouth_left = (center_x - radius * math.sin(angle_rad), center_y - 10)
            mouth_right = (center_x + radius * math.sin(angle_rad), center_y - 10)
            mouth_top = mouth_left
            mouth_bottom = mouth_right

        # Draw mouth triangle with black color to create opening effect
        points = [mouth_tip, mouth_top, mouth_bottom]
        pygame.draw.polygon(animated_sprite, (0, 0, 0), points)

        return animated_sprite

    def update_animation(self):
        self.animation_frame += 1

    def get_current_sprite(self) -> pygame.Surface:
        # Get base sprite for current direction
        base_sprite = self.direction_sprites[self.current_direction]

        # Calculate mouth animation
        animation_cycle = self.animation_frame // self.animation_speed
        if animation_cycle % 2 == 0:
            # Mouth closed
            mouth_angle = 45
        else:
            # Mouth open
            mouth_angle = self.max_mouth_angle

        # Apply mouth animation
        return self._create_mouth_animation(base_sprite, mouth_angle)

    def set_direction(self, direction: Direction):
        self.current_direction = direction

    def draw(self, surface: pygame.Surface, position: Tuple[int, int]):
        current_sprite = self.get_current_sprite()
        surface.blit(current_sprite, position)
        self.update_animation()