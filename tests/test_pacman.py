import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pacManRf.src.game.constants import (
    Direction,
    FRIGHTENED_SECONDS,
    GamePhase,
    GameStatus,
)
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

    def test_next_level_preserves_run_state_and_refills_maze(self):
        self.game.score = 2_340
        self.game.high_score = 2_500
        self.game.lives = 2
        self.game.extra_life_awarded = True
        self.game.maze = [[" " if cell in ".o" else cell for cell in row] for row in self.game.maze]

        self.game.next_level()

        self.assertEqual(self.game.level, 2)
        self.assertEqual(self.game.score, 2_340)
        self.assertEqual(self.game.high_score, 2_500)
        self.assertEqual(self.game.lives, 2)
        self.assertTrue(self.game.extra_life_awarded)
        self.assertEqual(self.game._count_dots(), self.game.total_dots)

    def test_level_difficulty_changes_are_small_and_bounded(self):
        self.assertEqual(self.game.ghost_speed_multiplier, 1.0)
        self.assertEqual(self.game.frightened_duration, FRIGHTENED_SECONDS)

        self.game.level = 2
        self.assertAlmostEqual(self.game.ghost_speed_multiplier, 1.01)
        self.assertAlmostEqual(self.game.frightened_duration, 6.9)

        self.game.level = 100
        self.assertAlmostEqual(self.game.ghost_speed_multiplier, 1.20)
        self.assertAlmostEqual(self.game.frightened_duration, 4.5)

    def test_manual_clear_automatically_starts_next_level(self):
        self.game.score = 900
        self.game.lives = 2
        self.game.phase = GamePhase.CLEARING
        self.game.phase_timer = 0.0

        self.game._update(0.01)

        self.assertEqual(self.game.level, 2)
        self.assertEqual(self.game.score, 900)
        self.assertEqual(self.game.lives, 2)
        self.assertEqual(self.game.phase, GamePhase.READY)


if __name__ == "__main__":
    unittest.main()
