"""Integration tests for learning sessions and the playable champion race."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
from drivingGameRL.src.population_rollout import PopulationRolloutManager
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
        for invalid in (True, 0, -1, 2.5):
            with self.subTest(parallel_workers=invalid):
                with self.assertRaises(ValueError):
                    LearningRuntimeConfig(parallel_workers=invalid)


class DrivingLearningSessionTests(unittest.TestCase):
    def test_default_population_replay_starts_early_and_updates_periodically(self):
        automatic = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="genetic_dqn",
                population_size=2,
                elite_count=1,
                evaluation_steps=900,
                seed=3,
            )
        )
        explicit_config = tiny_dqn(seed=3)
        explicit = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="genetic_dqn",
                population_size=2,
                elite_count=1,
                evaluation_steps=900,
                seed=3,
            ),
            dqn_config=explicit_config,
        )
        self.addCleanup(automatic.close)
        self.addCleanup(explicit.close)

        self.assertEqual(automatic.agent.config.warmup_steps, 96)
        self.assertEqual(automatic.agent.config.train_interval, 4)
        self.assertEqual(
            explicit.agent.config.warmup_steps,
            explicit_config.warmup_steps,
        )
        self.assertEqual(
            explicit.agent.config.train_interval,
            explicit_config.train_interval,
        )
        self.assertEqual(
            explicit.agent.config.hidden_sizes,
            explicit_config.hidden_sizes,
        )
        telemetry = automatic.telemetry()
        self.assertEqual(telemetry["warmup_steps"], 96)
        self.assertEqual(telemetry["train_interval"], 4)

    def test_population_step_many_stops_exactly_and_counts_all_car_decisions(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="genetic",
                evaluation_steps=3,
                population_size=4,
                elite_count=1,
                tournament_size=2,
                parallel_workers=4,
                seed=8,
            ),
            dqn_config=tiny_dqn(seed=8),
        )
        self.addCleanup(session.close)

        results = session.step_many(20, stop_after_generation=True)

        self.assertEqual(len(results), 3)
        self.assertTrue(results[-1].evolved)
        self.assertEqual(session.completed_generations, 1)
        self.assertEqual(session.environment_decisions, 12)
        telemetry = session.telemetry()
        self.assertEqual(telemetry["last_batch_ticks"], 3)
        self.assertEqual(telemetry["last_batch_decisions"], 12)

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
        self.assertEqual(len(telemetry["observation"]), 16)
        self.assertEqual(len(telemetry["q_values"]), 5)
        self.assertEqual(telemetry["network"]["architecture"], [16, 8, 5])
        self.assertEqual(telemetry["replay_size"], 2)

    def test_pure_genetic_population_evolves_after_one_lockstep_tick(self):
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

        result = session.step()
        telemetry = session.telemetry()

        self.assertTrue(result.member_completed)
        self.assertTrue(result.generation_completed)
        self.assertTrue(result.evolved)
        self.assertEqual(len(result.member_results), 2)
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
        completed = session.step()

        self.assertEqual(len(completed.member_results), 2)
        self.assertTrue(
            all(result.training_updates > 0 for result in completed.member_results)
        )

    def test_population_session_exposes_parallel_workers_and_real_scored_cars(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="genetic",
                evaluation_steps=4,
                population_size=4,
                elite_count=1,
                tournament_size=2,
                parallel_workers=2,
                seed=19,
            ),
            dqn_config=tiny_dqn(seed=19),
        )
        self.addCleanup(session.close)

        session.step()
        telemetry = session.telemetry()
        cars = session.scored_population_telemetry(include_rays=True)

        self.assertEqual(
            telemetry["parallel_workers"],
            min(2, max(1, os.cpu_count() or 1), 4, 32),
        )
        self.assertEqual(telemetry["active_member_indices"], [0, 1, 2, 3])
        self.assertEqual(
            {item["status"] for item in telemetry["population"]}, {"evaluating"}
        )
        self.assertEqual(len(cars), 4)
        self.assertEqual({item["steps"] for item in cars}, {1})
        self.assertTrue(all(item["source"] == "training" for item in cars))
        self.assertTrue(all(len(item["sensor_rays"]) == 9 for item in cars))
        self.assertEqual(
            tuple(id(env) for env in session._population_trainer.member_environments),
            tuple(id(env) for env in PopulationRolloutManager(session).environments),
        )

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

    def test_population_game_shows_real_generation_cars_by_default(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="genetic",
                evaluation_steps=4,
                population_size=4,
                elite_count=1,
                tournament_size=2,
                seed=17,
            ),
            dqn_config=tiny_dqn(seed=17),
        )
        game = DrivingLearningGame(session, render=False)
        self.addCleanup(game.close)

        self.assertTrue(game.show_population_cars)
        telemetry = game._training_telemetry()
        self.assertEqual(len(telemetry["population_rollouts"]), 4)
        self.assertTrue(
            all(
                item["source"] == "training"
                for item in telemetry["population_rollouts"]
            )
        )

    def test_speed_keys_cover_slow_through_max(self):
        self.assertEqual(self.game.steps_per_frame, 4)
        for _ in range(10):
            self.game.change_speed(1)
        self.assertEqual(self.game.steps_per_frame, 256)
        self.assertEqual(self.game.speed_label, "MAX")
        for _ in range(10):
            self.game.change_speed(-1)
        self.assertEqual(self.game.steps_per_frame, 1)

    def test_v_and_m_toggle_real_rays_and_generation_rollouts(self):
        self.assertTrue(self.game.show_sensor_rays)
        self.assertFalse(self.game.show_population_cars)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        self.game.handle_events()
        telemetry = self.game._training_telemetry()

        self.assertFalse(self.game.show_sensor_rays)
        self.assertTrue(self.game.show_population_cars)
        self.assertFalse(telemetry["show_sensor_rays"])
        self.assertTrue(telemetry["show_population_cars"])
        self.assertEqual(len(telemetry["population_rollouts"]), 1)
        self.assertEqual(telemetry["population_rollouts"][0]["sensor_rays"], [])

    def test_dashboard_buttons_control_training_without_keyboard_shortcuts(self):
        self.game.draw()

        pause = self.game.dashboard.control_rects["toggle_pause"].center
        pygame.event.post(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pause)
        )
        self.game.handle_events()
        self.assertTrue(self.game.paused)

        faster = self.game.dashboard.control_rects["speed_up"].center
        pygame.event.post(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=faster)
        )
        self.game.handle_events()
        self.assertEqual(self.game.steps_per_frame, 16)

        cars = self.game.dashboard.control_rects["toggle_population_cars"].center
        rays = self.game.dashboard.control_rects["toggle_sensor_rays"].center
        pygame.event.post(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=cars)
        )
        pygame.event.post(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rays)
        )
        self.game.handle_events()
        self.assertTrue(self.game.show_population_cars)
        self.assertFalse(self.game.show_sensor_rays)

    def test_c_cycles_preview_breadth_without_work_while_cars_are_hidden(self):
        self.assertEqual(self.game.population_car_limit, 8)
        self.assertIsNone(self.game.population_rollouts)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        self.game.handle_events()

        self.assertEqual(self.game.population_car_limit, 12)
        self.assertIsNone(self.game.population_rollouts)
        telemetry = self.game._training_telemetry()
        self.assertEqual(telemetry["population_car_limit"], 12)
        self.assertEqual(telemetry["population_rollouts"], [])

        self.game.toggle_population_cars()
        first_manager = self.game.population_rollouts
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        self.game.handle_events()

        self.assertEqual(self.game.population_car_limit, 2)
        self.assertIsNotNone(self.game.population_rollouts)
        self.assertIsNot(self.game.population_rollouts, first_manager)

    def test_disabled_generation_cars_do_no_preview_work(self):
        self.game.toggle_population_cars()
        manager = self.game.population_rollouts
        self.assertIsNotNone(manager)
        self.game.toggle_population_cars()

        with patch.object(manager, "step", wraps=manager.step) as preview_step:
            self.game.run(max_training_steps=4)
            self.game._training_telemetry()

        preview_step.assert_not_called()

    def test_interactive_training_slice_yields_at_the_ui_budget(self):
        self.game.render_enabled = True
        self.game.training_frame_budget_ms = 10.0
        try:
            with patch(
                "drivingGameRL.src.learning_game.perf_counter",
                side_effect=(
                    100.0,
                    100.0,
                    100.0,
                    100.011,
                    100.011,
                    100.011,
                ),
            ):
                advanced = self.game._advance_training_slice(
                    starting_generations=0,
                    max_training_steps=None,
                    max_generations=None,
                )
        finally:
            self.game.render_enabled = False

        self.assertEqual(advanced, 1)
        self.assertEqual(self.game.training_steps, 1)
        telemetry = self.game._training_telemetry()
        self.assertEqual(telemetry["requested_training_steps_per_frame"], 4)
        self.assertEqual(telemetry["effective_training_steps_per_frame"], 1)
        self.assertEqual(telemetry["frame_training_steps"], 1)
        self.assertTrue(telemetry["training_slice_capped"])
        self.assertAlmostEqual(telemetry["training_slice_ms"], 11.0)
        self.assertGreater(telemetry["training_ticks_per_second"], 0.0)
        self.assertGreater(telemetry["environment_decisions_per_second"], 0.0)

    def test_headless_training_slice_executes_the_exact_requested_batch(self):
        advanced = self.game._advance_training_slice(
            starting_generations=0,
            max_training_steps=None,
            max_generations=None,
        )

        self.assertEqual(advanced, 4)
        self.assertEqual(self.game.training_steps, 4)
        self.assertFalse(self.game._training_slice_capped)
        telemetry = self.game._training_telemetry()
        self.assertGreater(telemetry["training_slice_ms"], 0.0)
        self.assertGreater(telemetry["training_ticks_per_second"], 0.0)

    def test_frame_budget_and_preview_limit_reject_invalid_values(self):
        for invalid in (True, 3, 8.0):
            with self.subTest(population_car_limit=invalid):
                with self.assertRaises(ValueError):
                    DrivingLearningGame(
                        self.session,
                        render=False,
                        population_car_limit=invalid,
                    )
        for invalid in (True, 0, -1, float("nan"), float("inf"), "12"):
            with self.subTest(training_frame_budget_ms=invalid):
                with self.assertRaises(ValueError):
                    DrivingLearningGame(
                        self.session,
                        render=False,
                        training_frame_budget_ms=invalid,
                    )

    def test_close_releases_session_resources(self):
        with patch.object(self.session, "close", wraps=self.session.close) as close:
            self.game.close()

        close.assert_called_once_with()

    def test_paused_single_step_advances_enabled_generation_cars(self):
        self.game.toggle_population_cars()
        self.game.paused = True
        manager = self.game.population_rollouts
        self.assertIsNotNone(manager)
        before = manager.telemetry(include_rays=False)[0]["steps"]

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
        self.game.handle_events()

        after = manager.telemetry(include_rays=False)[0]["steps"]
        self.assertEqual(self.game.training_steps, 1)
        self.assertEqual(after, before + 1)

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
                "--workers",
                "6",
                "--population-cars",
                "--preview-cars",
                "12",
                "--no-sensors",
            ]
        )
        self.assertTrue(args.learn)
        self.assertEqual(args.algorithm, "genetic_dqn")
        self.assertEqual(args.population, 14)
        self.assertEqual(args.elite_count, 3)
        self.assertEqual(args.evaluation_steps, 1200)
        self.assertEqual(args.workers, 6)
        self.assertTrue(args.population_cars)
        self.assertEqual(args.preview_cars, 12)
        self.assertTrue(args.no_sensors)

    def test_population_car_cli_defaults_can_be_explicitly_overridden(self):
        parser = build_parser()

        default = parser.parse_args(["--learn", "--algorithm", "genetic_dqn"])
        hidden = parser.parse_args(
            ["--learn", "--algorithm", "genetic_dqn", "--no-population-cars"]
        )
        standalone = parser.parse_args(["--learn", "--algorithm", "double_dqn"])

        self.assertIsNone(default.population_cars)
        self.assertFalse(hidden.population_cars)
        self.assertIsNone(standalone.population_cars)


if __name__ == "__main__":
    unittest.main()
