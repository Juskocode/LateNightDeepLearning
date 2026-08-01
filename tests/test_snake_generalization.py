"""Held-out seeds, curriculum, and domain-randomized Snake starts."""

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from snakeGameQDlearning.src.game import (
    EpisodeSeedStreams,
    SnakeCurriculum,
    SnakeGameAI,
    get_environment_preset,
)
from snakeGameQDlearning.src.game.constants import Direction
from snakeGameQDlearning.src.ml.agent import Agent
from snakeGameQDlearning.src.ml.evaluation import (
    evaluate_agent,
    resolve_evaluation_suite,
)


class SnakeGeneralizationTests(unittest.TestCase):
    def test_seed_streams_are_repeatable_and_separate(self):
        first = EpisodeSeedStreams(17)
        second = EpisodeSeedStreams(17)
        train = tuple(first.train_seed(index) for index in range(20))
        validation = first.validation_seeds(20)
        final_test = first.final_test_seeds(20)

        self.assertEqual(train, tuple(second.train_seed(index) for index in range(20)))
        self.assertEqual(validation, second.validation_seeds(20))
        self.assertEqual(final_test, second.final_test_seeds(20))
        self.assertEqual(first.evaluation_seeds(0, 20), validation)
        self.assertEqual(first.evaluation_seeds(99, 20), validation)
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertTrue(set(train).isdisjoint(final_test))
        self.assertTrue(set(validation).isdisjoint(final_test))

    def test_checkpoint_suite_is_reused_only_for_matching_environment(self):
        streams = EpisodeSeedStreams(17, 1001, 2001)
        stored_validation = [101, 202, 303]
        metadata = {
            "experiment": {
                "environment": "compact",
                "validation_seed_root": 1001,
                "validation_seeds": stored_validation,
                "evaluation_round": 4,
            }
        }

        reproduced = resolve_evaluation_suite(
            streams,
            "validation",
            8,
            loaded_metadata=metadata,
            environment="compact",
            prefer_stored=True,
        )
        regenerated = resolve_evaluation_suite(
            streams,
            "validation",
            8,
            loaded_metadata=metadata,
            environment="wide",
            prefer_stored=True,
        )

        self.assertEqual(reproduced.seeds, tuple(stored_validation))
        self.assertEqual(reproduced.source, "checkpoint")
        self.assertEqual(reproduced.evaluation_round, 4)
        self.assertEqual(regenerated.seeds, streams.validation_seeds(8))
        self.assertEqual(regenerated.source, "cli")

    def test_curriculum_increases_spawn_randomization(self):
        curriculum = SnakeCurriculum()
        self.assertEqual(curriculum.stage_for(0).name, "orientation")
        self.assertEqual(curriculum.stage_for(20).name, "spawn_shift")
        self.assertEqual(curriculum.stage_for(75).name, "generalization")
        self.assertLess(
            curriculum.stage_for(0).randomized_start_probability,
            curriculum.stage_for(75).randomized_start_probability,
        )

    def test_randomized_start_is_seeded_and_always_valid(self):
        first = SnakeGameAI(
            width=200, height=160, render=False, seed=91, randomize_start=True
        )
        second = SnakeGameAI(
            width=200, height=160, render=False, seed=91, randomize_start=True
        )

        self.assertEqual(first.direction, second.direction)
        self.assertEqual(first.snake, second.snake)
        self.assertEqual(first.food, second.food)
        self.assertEqual(len(set(first.snake)), 3)
        for point in first.snake:
            self.assertFalse(point.x < 0 or point.x >= first.width)
            self.assertFalse(point.y < 0 or point.y >= first.height)
        dx, dy = first.direction.value
        self.assertEqual(first.snake[1].x, first.head.x - dx * 20)
        self.assertEqual(first.snake[1].y, first.head.y - dy * 20)
        self.assertFalse(
            first.is_collision(
                type(first.head)(first.head.x + dx * 20, first.head.y + dy * 20)
            )
        )

    def test_all_educational_presets_create_valid_games(self):
        for name in ("tutorial", "compact", "standard", "wide"):
            with self.subTest(environment=name):
                preset = get_environment_preset(name)
                game = SnakeGameAI(
                    width=preset.width,
                    height=preset.height,
                    render=False,
                    seed=4,
                    randomize_start=True,
                )
                self.assertIsInstance(game.direction, Direction)
                self.assertIsNotNone(game.food)

    def test_held_out_evaluation_is_repeatable_and_read_only(self):
        agent = Agent(algorithm="q_learning", seed=21)
        preset = get_environment_preset("tutorial")
        seeds = EpisodeSeedStreams(21).evaluation_seeds(0, 3)
        rng_before = repr(agent.rng.bit_generator.state)

        first = evaluate_agent(agent, seeds, preset, max_steps=30)
        second = evaluate_agent(agent, seeds, preset, max_steps=30)

        self.assertEqual(first, second)
        self.assertEqual(len(agent.memory), 0)
        self.assertEqual(agent.learning.learned_states, 0)
        self.assertEqual(repr(agent.rng.bit_generator.state), rng_before)

    def test_evaluation_metrics_are_visible_in_telemetry(self):
        agent = Agent(algorithm="double_dqn", seed=6)
        game = SnakeGameAI(render=False, seed=6)
        agent.update_evaluation_metrics(
            {"episodes": 4, "mean_score": 1.5, "std_score": 0.5},
            training_mean=3.0,
        )

        telemetry = agent.telemetry(agent.get_state(game), game)

        self.assertEqual(telemetry["evaluation"]["episodes"], 4)
        self.assertEqual(telemetry["evaluation"]["generalization_gap"], 1.5)
        self.assertEqual(telemetry["episode_seed"], 6)


if __name__ == "__main__":
    unittest.main()
