import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from snakeGameQDlearning.src.config.settings import BLOCK_SIZE, FOOD_REWARD, WIN_REWARD
from snakeGameQDlearning.src.game import Direction, Point, SnakeGameAI
from snakeGameQDlearning.src.ml.agent import Agent


class SnakeEnvironmentUpgradeTests(unittest.TestCase):
    def test_moving_into_vacating_tail_is_legal(self):
        game = SnakeGameAI(width=80, height=60, render=False, seed=1)
        game.direction = Direction.UP
        game.head = Point(20, 20)
        game.snake = [game.head, Point(20, 40), Point(0, 40), Point(0, 20)]
        game.food = Point(60, 0)

        _, done, _ = game.play_step([0, 0, 1], render_frame=False)

        self.assertFalse(done)
        self.assertEqual(game.head, Point(0, 20))
        self.assertEqual(len(game.snake), 4)

    def test_eating_resets_starvation_budget(self):
        game = SnakeGameAI(render=False, seed=2)
        game.food = Point(game.head.x + BLOCK_SIZE, game.head.y)
        game.steps_since_food = game.starvation_budget

        reward, done, score = game.play_step([1, 0, 0], render_frame=False)

        self.assertFalse(done)
        self.assertEqual(reward, FOOD_REWARD)
        self.assertEqual(score, 1)
        self.assertEqual(game.steps_since_food, 0)

    def test_filling_board_is_a_win_instead_of_food_placement_error(self):
        game = SnakeGameAI(width=80, height=60, render=False, seed=3)
        free = Point(20, 0)
        all_cells = [Point(x, y) for y in range(0, 60, 20) for x in range(0, 80, 20)]
        game.head = Point(0, 0)
        game.direction = Direction.RIGHT
        game.snake = [game.head] + [cell for cell in all_cells if cell not in (game.head, free)]
        game.food = free

        reward, done, _ = game.play_step([1, 0, 0], render_frame=False)

        self.assertTrue(done)
        self.assertTrue(game.won)
        self.assertEqual(game.termination_reason, "win")
        self.assertEqual(reward, FOOD_REWARD + WIN_REWARD)

    def test_invalid_non_one_hot_action_is_rejected(self):
        game = SnakeGameAI(render=False, seed=4)
        with self.assertRaises(ValueError):
            game.play_step([1, 1, 0], render_frame=False)

    def test_terminal_win_state_remains_observable(self):
        game = SnakeGameAI(render=False, seed=5)
        game.food = None
        game.won = True

        state = Agent(seed=5).get_state(game)

        self.assertEqual(state.shape, (11,))
        self.assertEqual(state[7:].tolist(), [0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
