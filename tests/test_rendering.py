import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from pacManRf.src.game.pacmanGame import PacmanGame
from snakeGameQDlearning.src.game import SnakeGameAI
from snakeGameQDlearning.src.ml.agent import Agent


class RenderingSmokeTests(unittest.TestCase):
    def test_both_renderers_save_nonempty_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            pacman_path = Path(directory) / "pacman.png"
            snake_path = Path(directory) / "snake.png"

            pacman = PacmanGame(render=False, seed=4)
            pacman.save_screenshot(pacman_path)

            snake = SnakeGameAI(render=False, seed=4)
            agent = Agent(seed=4)
            state = agent.get_state(snake)
            agent.get_action(state)
            snake.set_debug_info(**agent.telemetry(state, snake))
            snake.save_screenshot(snake_path)

            self.assertGreater(pacman_path.stat().st_size, 1_000)
            self.assertGreater(snake_path.stat().st_size, 1_000)
            self.assertEqual(pygame.image.load(pacman_path).get_size(),
                             (pacman.maze_width, pacman.display_height))
            self.assertEqual(pygame.image.load(snake_path).get_size(),
                             (snake.window_width, snake.window_height))


if __name__ == "__main__":
    unittest.main()
