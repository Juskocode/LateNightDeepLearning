"""Integration tests for learning sessions and the playable champion race."""

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import numpy as np
import torch

from drivingGameRL.main import build_parser
from drivingGameRL.src.learning_game import DrivingLearningGame
from drivingGameRL.src.learning_runtime import (
    ChampionRace,
    DrivingLearningSession,
    LearningRuntimeConfig,
)
from drivingGameRL.src.ml import DQNConfig
from drivingGameRL.src.vehicle import DriverControls


def tiny_dqn(seed: int = 13) -> DQNConfig:
    return DQNConfig(
        hidden_sizes=(8,),
        replay_capacity=32,
        batch_size=2,
        warmup_steps=0,
        target_sync_interval=2,
        epsilon_decay_steps=20,
        seed=seed,
    )


class LearningRuntimeConfigTests(unittest.TestCase):
    def test_population_constraints_are_validated(self):
        with self.assertRaises(ValueError):
            LearningRuntimeConfig(population_size=1)
        with self.assertRaises(ValueError):
            LearningRuntimeConfig(population_size=2, elite_count=2)
        with self.assertRaises(ValueError):
            LearningRuntimeConfig(mutation_rate=1.1)


class DrivingLearningSessionTests(unittest.TestCase):
    def test_dqn_episode_trains_and_advances_generation(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="double_dqn",
                evaluation_steps=2,
                population_size=2,
                elite_count=1,
                seed=13,
            ),
            dqn_config=tiny_dqn(),
        )

        session.step()
        session.step()
        telemetry = session.telemetry()

        self.assertEqual(session.completed_generations, 1)
        self.assertEqual(telemetry["generation"], 2)
        self.assertEqual(telemetry["gradient_steps"], 1)
        self.assertEqual(len(telemetry["observation"]), 12)
        self.assertEqual(len(telemetry["q_values"]), 5)
        self.assertEqual(telemetry["network"]["architecture"], [12, 8, 5])
        self.assertEqual(telemetry["replay_size"], 2)

    def test_pure_genetic_population_evolves_after_every_member(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="genetic",
                evaluation_steps=1,
                population_size=2,
                elite_count=1,
                tournament_size=2,
                seed=5,
            ),
            dqn_config=tiny_dqn(seed=5),
        )

        first = session.step()
        second = session.step()
        telemetry = session.telemetry()

        self.assertTrue(first.member_completed)
        self.assertTrue(second.generation_completed)
        self.assertTrue(second.evolved)
        self.assertEqual(session.completed_generations, 1)
        self.assertEqual(len(telemetry["population"]), 2)
        self.assertEqual(len(telemetry["generation_history"]), 1)
        self.assertEqual(telemetry["gradient_steps"], 0)

    def test_hybrid_population_performs_real_td_learning(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="genetic_dqn",
                evaluation_steps=2,
                population_size=2,
                elite_count=1,
                seed=9,
            ),
            dqn_config=tiny_dqn(seed=9),
        )

        session.step()
        session.step()

        completed_member = session._population_trainer.population[0]
        self.assertIsNotNone(completed_member.result)
        self.assertGreater(completed_member.agent.gradient_steps, 0)

    def test_loaded_dqn_is_the_best_available_race_policy(self):
        config = LearningRuntimeConfig(
            algorithm="double_dqn",
            evaluation_steps=10,
            population_size=2,
            elite_count=1,
            seed=31,
        )
        trained = DrivingLearningSession(config, dqn_config=tiny_dqn(seed=31))
        trained.step()
        trained.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "driver.pth"
            trained.save(path)
            restored = DrivingLearningSession(config, dqn_config=tiny_dqn(seed=31))
            restored.load(path)

        observation = restored.env.observation()
        np.testing.assert_array_equal(
            restored.agent.q_values(observation),
            restored.champion_agent().q_values(observation),
        )


class ChampionRaceTests(unittest.TestCase):
    def test_race_uses_independent_environments_and_frozen_policy(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="dqn",
                evaluation_steps=10,
                population_size=2,
                elite_count=1,
                seed=4,
            ),
            dqn_config=tiny_dqn(seed=4),
        )
        session.step()
        training_steps = session.env.steps
        replay_size = len(session.agent.replay)
        weights = [parameter.detach().clone() for parameter in session.agent.network.parameters()]

        race = ChampionRace(session)
        for _ in range(3):
            race.step(DriverControls(throttle=1.0))

        self.assertAlmostEqual(race.elapsed, 3.0 / 60.0)
        self.assertEqual(session.env.steps, training_steps)
        self.assertEqual(len(session.agent.replay), replay_size)
        self.assertIsNot(race.human_env, session.env)
        self.assertIsNot(race.champion_env, session.env)
        for before, after in zip(weights, session.agent.network.parameters()):
            self.assertTrue(torch.equal(before, after))


class DrivingLearningGameTests(unittest.TestCase):
    def setUp(self):
        self.session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="dqn",
                evaluation_steps=10,
                population_size=2,
                elite_count=1,
                seed=21,
            ),
            dqn_config=tiny_dqn(seed=21),
        )
        self.game = DrivingLearningGame(
            self.session, render=False, learning_speed=4
        )

    def tearDown(self):
        self.game.close()

    def test_p_toggles_an_isolated_champion_race(self):
        training_env = self.session.env
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
        self.game.handle_events()

        self.assertIsNotNone(self.game.race)
        self.assertIs(self.session.env, training_env)
        self.assertIsNot(self.game.race.human_env, training_env)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
        self.game.handle_events()
        self.assertIsNone(self.game.race)

    def test_speed_keys_cover_slow_through_max(self):
        self.assertEqual(self.game.steps_per_frame, 4)
        for _ in range(10):
            self.game.change_speed(1)
        self.assertEqual(self.game.steps_per_frame, 256)
        self.assertEqual(self.game.speed_label, "MAX")
        for _ in range(10):
            self.game.change_speed(-1)
        self.assertEqual(self.game.steps_per_frame, 1)

    def test_learning_cli_exposes_all_algorithms_and_population_controls(self):
        args = build_parser().parse_args(
            [
                "--learn",
                "--algorithm",
                "genetic_dqn",
                "--population",
                "14",
                "--elite-count",
                "3",
                "--evaluation-steps",
                "1200",
            ]
        )
        self.assertTrue(args.learn)
        self.assertEqual(args.algorithm, "genetic_dqn")
        self.assertEqual(args.population, 14)
        self.assertEqual(args.elite_count, 3)
        self.assertEqual(args.evaluation_steps, 1200)


if __name__ == "__main__":
    unittest.main()
