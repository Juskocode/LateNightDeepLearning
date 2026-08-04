"""Learning-runtime integration for the driving random-start curriculum."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from drivingGameRL.src.environment import DrivingEnv, StepResult
from drivingGameRL.src.learning_runtime import (
    ChampionRace,
    DrivingLearningSession,
    LearningRuntimeConfig,
)
from drivingGameRL.src.ml import DQNConfig
from drivingGameRL.src.ml.evolution import EvolutionConfig, PopulationTrainer


def _dqn(seed: int) -> DQNConfig:
    return DQNConfig(
        hidden_sizes=(8,),
        replay_capacity=16,
        batch_size=2,
        warmup_steps=0,
        epsilon_start=0.0,
        epsilon_end=0.0,
        seed=seed,
    )


def _evolution(seed: int = 29) -> EvolutionConfig:
    return EvolutionConfig(
        algorithm="genetic",
        population_size=2,
        elite_count=1,
        tournament_size=2,
        evaluation_steps=3,
        initial_lap_target=1,
        max_lap_target=1,
        seed=seed,
    )


class DrivingRandomSpawnRuntimeTests(unittest.TestCase):
    def test_population_default_target_scales_the_per_lap_budget(self):
        config = EvolutionConfig(
            algorithm="genetic",
            population_size=2,
            elite_count=1,
            tournament_size=2,
            evaluation_steps=7,
            seed=61,
        )
        trainer = PopulationTrainer(config, _dqn(61), auto_evolve=False)
        self.addCleanup(trainer.close)

        self.assertEqual(trainer.lap_target, 2)
        self.assertEqual(trainer.evaluation_step_budget, 14)
        self.assertEqual(trainer.env.max_steps, 14)
        self.assertEqual(
            {env.lap_target for env in trainer.member_environments},
            {2},
        )

    def test_episode_and_population_learning_opt_into_random_starts(self):
        episode = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="double_dqn",
                evaluation_steps=3,
                population_size=2,
                elite_count=1,
                seed=17,
            ),
            dqn_config=_dqn(17),
        )
        population = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="genetic_dqn",
                evaluation_steps=3,
                population_size=2,
                elite_count=1,
                seed=17,
            ),
            dqn_config=_dqn(17),
        )

        self.assertTrue(episode.env.random_start_curriculum)
        self.assertTrue(population.env.random_start_curriculum)
        self.assertEqual(episode.env.telemetry()["spawn_mode"], "random_track")
        self.assertEqual(population.env.telemetry()["spawn_mode"], "random_track")

    def test_learning_random_starts_are_seed_reproducible(self):
        config = LearningRuntimeConfig(
            algorithm="dqn",
            evaluation_steps=5,
            population_size=2,
            elite_count=1,
            seed=71,
        )
        first = DrivingLearningSession(config, dqn_config=_dqn(71))
        second = DrivingLearningSession(config, dqn_config=_dqn(71))

        self.assertEqual(
            first.env.telemetry()["position"], second.env.telemetry()["position"]
        )
        first.env.load_curriculum_state({"ready": True})
        second.env.load_curriculum_state({"ready": True})
        first_observation = first.env.reset(seed=908)
        second_observation = second.env.reset(seed=908)

        self.assertEqual(first_observation, second_observation)
        self.assertEqual(
            first.env.telemetry()["spawn_mode"],
            second.env.telemetry()["spawn_mode"],
        )

    def test_champion_race_always_uses_the_normal_start_line(self):
        session = DrivingLearningSession(
            LearningRuntimeConfig(
                algorithm="dqn",
                evaluation_steps=5,
                population_size=2,
                elite_count=1,
                seed=23,
            ),
            dqn_config=_dqn(23),
        )

        race = ChampionRace(session)
        start_position, _ = race.human_env.circuit.start_pose()

        self.assertFalse(race.human_env.random_start_curriculum)
        self.assertFalse(race.champion_env.random_start_curriculum)
        self.assertEqual(race.human_env.vehicle.state.position, start_position)
        self.assertEqual(race.champion_env.vehicle.state.position, start_position)

    def test_episode_checkpoint_preserves_curriculum_readiness(self):
        config = LearningRuntimeConfig(
            algorithm="double_dqn",
            evaluation_steps=5,
            population_size=2,
            elite_count=1,
            seed=37,
        )
        session = DrivingLearningSession(config, dqn_config=_dqn(37))
        session.env.load_curriculum_state({"unlocked": True})
        # Move beyond the constructor's first RNG draw so a naive reseed on
        # load cannot accidentally match the expected continuation.
        session.env.reset()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = session.save(Path(directory) / "driver.pth")
            expected_observation = session.env.reset()
            expected_mode = session.env.spawn_mode
            expected_progress = session.env.spawn_progress
            restored = DrivingLearningSession(config, dqn_config=_dqn(37))
            restored.load(checkpoint)

        self.assertTrue(restored.env.curriculum_ready)
        self.assertEqual(
            restored.env.curriculum_state(),
            {"unlocked": True, "lap_target": 2},
        )
        self.assertEqual(restored.observation, expected_observation)
        self.assertEqual(restored.env.spawn_mode, expected_mode)
        self.assertEqual(restored.env.spawn_progress, expected_progress)
        self.assertEqual(restored.episode_return, 0.0)

    def test_standalone_target_completion_promotes_and_resizes_next_budget(self):
        config = LearningRuntimeConfig(
            algorithm="double_dqn",
            evaluation_steps=11,
            initial_lap_target=2,
            max_lap_target=3,
            population_size=2,
            elite_count=1,
            seed=41,
        )
        session = DrivingLearningSession(config, dqn_config=_dqn(41))
        session.episode_return = 725.0
        completed = StepResult(
            observation=session.observation,
            reward=350.0,
            terminated=True,
            truncated=False,
            info={
                "laps": 2,
                "lap_target": 2,
                "lap_target_completed": True,
                "episode_target_progress": 1.0,
                "episode_best_lap_time": 19.0,
                "episode_mean_lap_time": 20.0,
                "episode_lap_time_bonus_total": 140.0,
            },
        )

        session._finish_dqn_episode(completed)

        self.assertEqual(session.env.lap_target, 3)
        self.assertEqual(session.env.max_steps, 33)
        history = session.generation_history[-1]
        self.assertEqual(history["lap_target"], 2)
        self.assertEqual(history["target_finishers"], 1)
        self.assertEqual(history["laps_completed"], 2)
        self.assertEqual(history["best_target_progress"], 1.0)
        self.assertEqual(history["lap_time_bonus_total"], 140.0)
        session.episode_return = 900.0
        slower_longer_target = StepResult(
            observation=session.observation,
            reward=350.0,
            terminated=True,
            truncated=False,
            info={
                "laps": 3,
                "lap_target": 3,
                "lap_target_completed": True,
                "episode_target_progress": 1.0,
                "episode_best_lap_time": 21.0,
                "episode_mean_lap_time": 22.0,
                "episode_lap_time_bonus_total": 180.0,
            },
        )
        session._finish_dqn_episode(slower_longer_target)
        self.assertEqual(session.best_fitness, 725.0)
        self.assertEqual(
            session.generation_history[-1]["fitness_per_target_lap"],
            300.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = session.save(Path(directory) / "promoted-driver.pth")
            restored = DrivingLearningSession(config, dqn_config=_dqn(41))
            restored.load(checkpoint)
        self.assertEqual(restored.env.lap_target, 3)
        self.assertEqual(restored.env.max_steps, 33)

    def test_population_trainer_defaults_to_curriculum_but_respects_custom_env(self):
        default = PopulationTrainer(
            _evolution(), _dqn(29), auto_evolve=False
        )
        normal_start_env = DrivingEnv(
            "harbor_loop", seed=29, max_steps=3, random_start_curriculum=False
        )
        custom = PopulationTrainer(
            _evolution(),
            _dqn(29),
            env=normal_start_env,
            auto_evolve=False,
        )

        self.assertTrue(default.env.random_start_curriculum)
        self.assertFalse(custom.env.random_start_curriculum)
        self.assertIs(custom.env, normal_start_env)

    def test_population_members_share_one_seeded_spawn_and_advance_together(self):
        trainer = PopulationTrainer(
            _evolution(seed=31), _dqn(31), auto_evolve=False
        )
        environments = trainer.member_environments
        positions = tuple(env.vehicle.state.position for env in environments)

        trainer.step()

        self.assertEqual(len({id(env) for env in environments}), 2)
        self.assertEqual(positions[0], positions[1])
        self.assertEqual({env.steps for env in environments}, {1})
        self.assertEqual(trainer.active_member_indices, (0, 1))
        self.assertEqual(
            {env.telemetry()["spawn_mode"] for env in environments},
            {"random_track"},
        )

    def test_population_unlock_is_deferred_until_generation_boundary(self):
        trainer = PopulationTrainer(
            _evolution(seed=53), _dqn(53), auto_evolve=False
        )
        observation = trainer.observation
        completed = StepResult(
            observation=observation,
            reward=20.0,
            terminated=True,
            truncated=False,
            info={
                "curriculum_lap_completed": True,
                "lap_target": 1,
                "lap_target_completed": True,
                "laps": 1,
                "progress": 0.42,
                "episode_lap_progress": 1.0,
                "episode_target_progress": 1.0,
                "max_episode_target_progress": 1.0,
            },
        )
        ordinary_end = StepResult(
            observation=observation,
            reward=0.0,
            terminated=False,
            truncated=True,
            info={
                "curriculum_lap_completed": False,
                "lap_target": 1,
                "lap_target_completed": False,
                "laps": 0,
                "progress": 0.0,
                "episode_target_progress": 0.0,
            },
        )

        first_env, second_env = trainer.member_environments
        with (
            patch.object(first_env, "step", return_value=completed),
            patch.object(second_env, "step", return_value=ordinary_end),
        ):
            result = trainer.step()

        self.assertTrue(result.member_completed)
        self.assertTrue(result.generation_completed)
        self.assertEqual(result.member_results[0].progress, 1.0)
        self.assertTrue(trainer._pending_curriculum_unlock)
        self.assertFalse(trainer.env.curriculum_ready)
        self.assertEqual(trainer.env.telemetry()["spawn_mode"], "random_track")
        self.assertFalse(result.evolved)
        self.assertFalse(trainer.env.curriculum_ready)

        trainer.evolve()

        self.assertTrue(trainer.env.curriculum_ready)
        self.assertFalse(trainer._pending_curriculum_unlock)

    def test_target_completion_promotes_once_and_checkpoint_preserves_latch(self):
        config = EvolutionConfig(
            algorithm="genetic",
            population_size=2,
            elite_count=1,
            tournament_size=2,
            evaluation_steps=3,
            initial_lap_target=2,
            max_lap_target=3,
            seed=67,
        )
        trainer = PopulationTrainer(config, _dqn(67), auto_evolve=False)
        self.addCleanup(trainer.close)

        for index, env in enumerate(trainer.member_environments):
            observation = trainer.member_observations[index]
            env.step = lambda _action, obs=observation: StepResult(
                observation=obs,
                reward=40.0,
                terminated=True,
                truncated=False,
                info={
                    "curriculum_lap_completed": True,
                    "lap_target": 2,
                    "lap_target_completed": True,
                    "laps": 2,
                    "episode_lap_progress": 1.0,
                    "episode_target_progress": 1.0,
                    "max_episode_target_progress": 1.0,
                },
            )

        completed = trainer.step()

        self.assertTrue(completed.generation_completed)
        self.assertEqual(trainer.lap_target, 2)
        self.assertTrue(trainer._pending_lap_target_increase)
        checkpoint = trainer.state_dict()

        restored = PopulationTrainer(config, _dqn(67), auto_evolve=False)
        self.addCleanup(restored.close)
        restored.load_state_dict(checkpoint)

        self.assertEqual(restored.lap_target, 2)
        self.assertEqual(restored.evaluation_step_budget, 6)
        self.assertTrue(restored._pending_lap_target_increase)

        record = restored.evolve()

        self.assertEqual(record.lap_target, 2)
        self.assertEqual(record.target_finishers, 2)
        self.assertEqual(restored.lap_target, 3)
        self.assertEqual(restored.evaluation_step_budget, 9)
        self.assertFalse(restored._pending_lap_target_increase)
        self.assertEqual(
            {env.lap_target for env in restored.member_environments},
            {3},
        )

    def test_population_checkpoint_preserves_curriculum_readiness(self):
        config = _evolution(seed=43)
        trainer = PopulationTrainer(config, _dqn(43), auto_evolve=False)
        for env in trainer.member_environments:
            env.load_curriculum_state({"unlocked": True})
        trainer._generation_curriculum_ready = True

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save(Path(directory) / "population.pth")
            restored = PopulationTrainer(config, _dqn(43), auto_evolve=False)
            restored.load(checkpoint)

        self.assertTrue(restored.env.random_start_curriculum)
        self.assertTrue(restored.env.curriculum_ready)
        self.assertEqual(
            restored.env.curriculum_state(),
            {"unlocked": True, "lap_target": 1},
        )


if __name__ == "__main__":
    unittest.main()
