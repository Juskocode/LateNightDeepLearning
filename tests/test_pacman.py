import os
import unittest
from collections import deque

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

    def test_eaten_ghost_routes_strictly_descend_from_every_reachable_cell(self):
        def neighboring_cell(cell, direction):
            dx, dy = direction.vector
            x, y = cell[0] + dx, cell[1] + dy
            if y == 9:
                x %= self.game.cols
            return x, y

        for index, ghost in enumerate(self.game.ghosts):
            distance_to_home = {ghost.spawn: 0}
            frontier = deque([ghost.spawn])
            while frontier:
                cell = frontier.popleft()
                for direction in Direction:
                    if not self.game._can_move(cell, direction):
                        continue
                    neighbor = neighboring_cell(cell, direction)
                    if neighbor in distance_to_home:
                        continue
                    distance_to_home[neighbor] = distance_to_home[cell] + 1
                    frontier.append(neighbor)

            self.assertGreater(len(distance_to_home), 1)
            ghost.eaten = True
            ghost.released = True
            ghost.target = None
            for cell, distance in distance_to_home.items():
                if cell == ghost.spawn:
                    continue
                ghost.grid_x, ghost.grid_y = cell
                for incoming_direction in Direction:
                    ghost.direction = incoming_direction
                    chosen = self.game._choose_ghost_direction(ghost, index)
                    next_cell = neighboring_cell(cell, chosen)
                    self.assertEqual(
                        distance_to_home[next_cell],
                        distance - 1,
                        (ghost.name, cell, incoming_direction, chosen),
                    )

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

    def test_eaten_ghosts_return_revive_and_render_their_body(self):
        self.game.phase = GamePhase.ACTIVE
        self.game.player.reset_position((1, 1), Direction.LEFT)
        self.game.next_direction = Direction.LEFT

        for tested_ghost in self.game.ghosts:
            for ghost in self.game.ghosts:
                ghost.released = False
                ghost.release_timer = 999.0
                ghost.eaten = False
                ghost.target = None

            tested_ghost.reset_position((9, 7), Direction.DOWN)
            tested_ghost.released = True
            self.game.player.reset_position(tested_ghost.grid, Direction.LEFT)
            self.game.frightened_timer = 3.0
            self.game._check_ghost_collisions()
            self.assertTrue(tested_ghost.eaten)
            self.game._render(0.0)
            self.assertTrue(tested_ghost.sprite.eaten)

            self.game.player.reset_position((1, 1), Direction.LEFT)

            for _ in range(180):
                self.game._update(1 / 60)
                if not tested_ghost.eaten:
                    break

            self.assertFalse(tested_ghost.eaten, tested_ghost.name)
            self.assertEqual(tested_ghost.grid, tested_ghost.spawn)
            self.assertFalse(tested_ghost.released)

            self.game._render(0.0)
            self.assertFalse(tested_ghost.sprite.eaten)

            for _ in range(60):
                self.game._update(1 / 60)
                if tested_ghost.released:
                    break
            self.assertTrue(tested_ghost.released)
            self.game._render(0.0)
            self.assertFalse(tested_ghost.sprite.eaten)

    def test_revived_ghost_cannot_be_immediately_re_eaten_inside_home(self):
        ghost = self.game.ghosts[0]
        self.game.phase = GamePhase.ACTIVE
        self.game.frightened_timer = 2.0
        self.game.player.reset_position(ghost.spawn, Direction.LEFT)
        ghost.reset_position(ghost.spawn, Direction.UP)
        ghost.released = True
        ghost.eaten = True

        self.game._update(1 / 60)

        self.assertFalse(ghost.eaten)
        self.assertFalse(ghost.released)
        self.assertEqual(self.game.ghost_chain, 0)


if __name__ == "__main__":
    unittest.main()
