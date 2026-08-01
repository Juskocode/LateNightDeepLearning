"""Isolation and sensor tests for presentation-only population rollouts."""

from __future__ import annotations

from copy import deepcopy
import math
import unittest

import numpy as np
import torch

from drivingGameRL.src.learning_runtime import (
    DrivingLearningSession,
    LearningRuntimeConfig,
)
from drivingGameRL.src.ml import DQNConfig
from drivingGameRL.src.population_rollout import PopulationRolloutManager


def _tiny_session(*, population_size: int = 4, evaluation_steps: int = 8):
    return DrivingLearningSession(
        LearningRuntimeConfig(
            algorithm="genetic_dqn",
            circuit="harbor_loop",
            seed=41,
            evaluation_steps=evaluation_steps,
            population_size=population_size,
            elite_count=1,
            tournament_size=2,
        ),
        dqn_config=DQNConfig(
            hidden_sizes=(8,),
            replay_capacity=16,
            batch_size=2,
            warmup_steps=0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            seed=41,
        ),
    )


def _assert_nested_equal(test: unittest.TestCase, before, after) -> None:
    if isinstance(before, torch.Tensor):
        test.assertTrue(torch.equal(before, after))
    elif isinstance(before, np.ndarray):
        np.testing.assert_array_equal(before, after)
    elif isinstance(before, dict):
        test.assertEqual(before.keys(), after.keys())
        for key in before:
            _assert_nested_equal(test, before[key], after[key])
    elif isinstance(before, (tuple, list)):
        test.assertEqual(len(before), len(after))
        for first, second in zip(before, after):
            _assert_nested_equal(test, first, second)
    else:
        test.assertEqual(before, after)


class PopulationRolloutManagerTests(unittest.TestCase):
    def test_clones_current_member_ids_with_a_hard_display_bound(self):
        session = _tiny_session(population_size=4)
        manager = PopulationRolloutManager(session, max_cars=3)

        snapshots = manager.telemetry(include_rays=False)

        self.assertEqual(manager.count, 3)
        self.assertEqual(manager.generation, session.current_generation)
        self.assertEqual([item["index"] for item in snapshots], [0, 1, 2])
        self.assertEqual(
            [item["member_id"] for item in snapshots],
            [member.member_id for member in session._population_trainer.population[:3]],
        )
        self.assertLessEqual(
            PopulationRolloutManager(session, max_cars=100).count,
            PopulationRolloutManager.HARD_MAX_CARS,
        )

    def test_every_car_has_an_independent_environment_and_same_start(self):
        manager = PopulationRolloutManager(_tiny_session())

        snapshots = manager.telemetry()

        self.assertEqual(len({id(env) for env in manager.environments}), manager.count)
        self.assertEqual(len({item["position"] for item in snapshots}), 1)
        self.assertEqual(len({item["heading"] for item in snapshots}), 1)
        self.assertTrue(all(item["steps"] == 0 for item in snapshots))

    def test_policy_clones_do_not_share_parameter_storage(self):
        session = _tiny_session()
        manager = PopulationRolloutManager(session)
        rollout_agent = manager._rollouts[0].agent
        source_agent = session._population_trainer.population[0].agent
        source_before = [
            parameter.detach().clone()
            for parameter in source_agent.online_network.parameters()
        ]

        with torch.no_grad():
            next(rollout_agent.online_network.parameters()).add_(10.0)

        self.assertTrue(
            all(
                torch.equal(before, current)
                for before, current in zip(
                    source_before, source_agent.online_network.parameters()
                )
            )
        )

    def test_refresh_reuses_isolated_policy_shells_and_restores_live_weights(self):
        session = _tiny_session()
        manager = PopulationRolloutManager(session)
        original_agents = tuple(rollout.agent for rollout in manager._rollouts)
        source_agents = tuple(
            member.agent for member in session._population_trainer.population
        )

        with torch.no_grad():
            next(original_agents[0].online_network.parameters()).add_(10.0)
        manager.refresh(force=True)

        refreshed_agents = tuple(rollout.agent for rollout in manager._rollouts)
        self.assertEqual(
            tuple(id(agent) for agent in refreshed_agents),
            tuple(id(agent) for agent in original_agents),
        )
        for clone, source in zip(refreshed_agents, source_agents):
            for clone_parameter, source_parameter in zip(
                clone.online_network.parameters(),
                source.online_network.parameters(),
            ):
                self.assertTrue(torch.equal(clone_parameter, source_parameter))
                self.assertNotEqual(
                    clone_parameter.untyped_storage().data_ptr(),
                    source_parameter.untyped_storage().data_ptr(),
                )

    def test_training_owned_agents_cannot_be_reused_as_policy_clones(self):
        session = _tiny_session()
        training_agent = session._population_trainer.population[-1].agent

        with self.assertRaisesRegex(ValueError, "training-owned"):
            session.population_policy_clones(
                max_cars=2,
                reusable_policy_clones=(training_agent,),
            )

    def test_sensor_rays_are_the_real_observation_distances(self):
        manager = PopulationRolloutManager(_tiny_session())
        manager.step(3)

        car = manager.telemetry(include_rays=True)[0]
        rollout = manager._rollouts[0]

        self.assertEqual(len(car["sensor_rays"]), 5)
        self.assertIs(car["rays"], car["sensor_rays"])
        self.assertEqual(
            [ray["normalized_distance"] for ray in car["sensor_rays"]],
            car["observation"][-5:],
        )
        for ray in car["sensor_rays"]:
            origin_x, origin_y = ray["origin"]
            endpoint_x, endpoint_y = ray["endpoint"]
            rendered_length = math.hypot(endpoint_x - origin_x, endpoint_y - origin_y)
            self.assertAlmostEqual(rendered_length, ray["distance"])
            self.assertAlmostEqual(
                ray["distance"], ray["normalized_distance"] * 150.0
            )
        expected_q_values = rollout.agent.q_values(car["observation"])
        np.testing.assert_allclose(car["q_values"], expected_q_values)
        self.assertEqual(car["action"], int(np.argmax(expected_q_values)))

    def test_rollouts_refresh_automatically_when_generation_changes(self):
        session = _tiny_session(population_size=3, evaluation_steps=2)
        manager = PopulationRolloutManager(session)
        original_environments = manager.environments

        while session.current_generation == 0:
            session.step()
        manager.step()

        self.assertEqual(manager.generation, session.current_generation)
        self.assertEqual(manager.count, 3)
        self.assertTrue(
            all(
                previous is not current
                for previous, current in zip(
                    original_environments, manager.environments
                )
            )
        )
        self.assertTrue(
            all(
                item["generation"] == session.current_generation
                for item in manager.telemetry(include_rays=False)
            )
        )

    def test_stepping_rollouts_leaves_trainer_state_bit_identical(self):
        session = _tiny_session(evaluation_steps=3)
        trainer = session._population_trainer
        training_env_before = deepcopy(session.env.telemetry())
        trainer_before = deepcopy(trainer.state_dict())
        manager = PopulationRolloutManager(session)

        manager.step(9)
        manager.telemetry(include_rays=True)

        _assert_nested_equal(self, trainer_before, trainer.state_dict())
        self.assertEqual(training_env_before, session.env.telemetry())
        self.assertEqual(len(trainer.population[0].agent.replay), 0)
        self.assertEqual(trainer.population[0].agent.gradient_steps, 0)
        self.assertTrue(all(item["episodes"] == 3 for item in manager.telemetry()))


if __name__ == "__main__":
    unittest.main()
