import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pacManRf.src.game.constants import Direction, FRIGHTENED_SECONDS, GameStatus
from pacManRf.src.game.pacmanGame import MAZE_TEMPLATE, PacmanGame


class PacmanGameTests(unittest.TestCase):
    def setUp(self):
        self.game = PacmanGame(render=False, seed=1)

    def test_maze_is_rectangular(self):
        self.assertTrue(all(len(row) == 20 for row in MAZE_TEMPLATE))

    def test_power_pellet_sets_frightened_mode(self):
        self.game.player.reset_position((1, 1), Direction.RIGHT)
        self.game._eat_current_cell()
        self.assertEqual(self.game.score, 50)
        self.assertEqual(self.game.frightened_timer, FRIGHTENED_SECONDS)

    def test_life_loss_resets_then_ends_game(self):
        for expected_lives in (2, 1, 0):
            self.game._lose_life()
            self.assertEqual(self.game.lives, expected_lives)
        self.assertEqual(self.game.status, GameStatus.LOST)

    def test_tunnel_cells_are_walkable(self):
        self.assertTrue(self.game._can_move((0, 9), Direction.LEFT))
        self.assertTrue(self.game._can_move((19, 9), Direction.RIGHT))


if __name__ == "__main__":
    unittest.main()
