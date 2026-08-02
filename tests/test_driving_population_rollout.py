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
from drivingGameRL.src.environment import DrivingEnv
from drivingGameRL.src.ml import DQNConfig
from drivingGameRL.src.population_rollout import PopulationRolloutManager


def _tiny_session(
    *,
    population_size: int = 4,
    evaluation_steps: int = 8,
    algorithm: str = "genetic_dqn",
):
    return DrivingLearningSession(
        LearningRuntimeConfig(
            algorithm=algorithm,
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
    def test_bounds_real_scored_members_in_population_order(self):
        session = _tiny_session(population_size=4)
        manager = PopulationRolloutManager(session, max_cars=3)

        snapshots = manager.telemetry(include_rays=False)

        self.assertEqual(manager.count, 3)
        self.assertTrue(manager.uses_scored_population)
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

    def test_every_scored_car_has_an_independent_environment_and_same_start(self):
        session = _tiny_session()
        manager = PopulationRolloutManager(session)

        snapshots = manager.telemetry()

        self.assertEqual(
            manager.environments,
            session._population_trainer.member_environments,
        )
        self.assertEqual(len({id(env) for env in manager.environments}), manager.count)
        self.assertEqual(len({item["position"] for item in snapshots}), 1)
        self.assertEqual(len({item["heading"] for item in snapshots}), 1)
        self.assertTrue(all(item["steps"] == 0 for item in snapshots))

    def test_scored_population_exposes_the_shared_random_origin(self):
        session = _tiny_session()
        manager = PopulationRolloutManager(session)

        snapshots = manager.telemetry(include_rays=False)

        self.assertTrue(
            all(env.random_start_curriculum for env in manager.environments)
        )
        self.assertEqual({item["spawn_mode"] for item in snapshots}, {"random_track"})
        self.assertEqual(len({item["spawn_progress"] for item in snapshots}), 1)
        self.assertEqual(len({item["lap_origin_progress"] for item in snapshots}), 1)
        self.assertEqual(
            {item["curriculum_unlocked"] for item in snapshots}, {False}
        )

    def test_manager_step_does_not_double_advance_scored_cars(self):
        session = _tiny_session(evaluation_steps=4)
        manager = PopulationRolloutManager(session)
        session.step()
        before = tuple(env.steps for env in manager.environments)

        manager.step(3)

        self.assertEqual(tuple(env.steps for env in manager.environments), before)

    def test_one_session_tick_advances_every_scored_car(self):
        session = _tiny_session(population_size=4, evaluation_steps=4)
        manager = PopulationRolloutManager(session)

        session.step()
        snapshots = manager.telemetry(include_rays=False)

        self.assertEqual({item["steps"] for item in snapshots}, {1})
        self.assertEqual({item["status"] for item in snapshots}, {"evaluating"})
        self.assertTrue(all(item["scored"] for item in snapshots))
        self.assertEqual(
            session._population_trainer.active_member_indices,
            (0, 1, 2, 3),
        )

    def test_standalone_policy_clone_does_not_share_parameter_storage(self):
        session = _tiny_session(algorithm="double_dqn")
        manager = PopulationRolloutManager(session)
        rollout_agent = manager._rollouts[0].agent
        source_agent = session.agent
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

    def test_standalone_refresh_reuses_isolated_policy_shell(self):
        session = _tiny_session(algorithm="double_dqn")
        manager = PopulationRolloutManager(session)
        original_agents = tuple(rollout.agent for rollout in manager._rollouts)
        source_agents = (session.agent,)

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
        session = _tiny_session()
        manager = PopulationRolloutManager(session)
        session.step()

        car = manager.telemetry(include_rays=True)[0]
        member = session._population_trainer.population[0]
        sensor_count = len(DrivingEnv.SENSOR_RELATIVE_ANGLES)

        self.assertEqual(len(car["sensor_rays"]), sensor_count)
        self.assertIs(car["rays"], car["sensor_rays"])
        np.testing.assert_allclose(
            [ray["normalized_distance"] for ray in car["sensor_rays"]],
            car["observation"][-sensor_count:],
            rtol=0.0,
            atol=1e-7,
        )
        for ray in car["sensor_rays"]:
            origin_x, origin_y = ray["origin"]
            endpoint_x, endpoint_y = ray["endpoint"]
            rendered_length = math.hypot(endpoint_x - origin_x, endpoint_y - origin_y)
            self.assertAlmostEqual(rendered_length, ray["distance"])
            self.assertAlmostEqual(
                ray["distance"], ray["normalized_distance"] * 150.0
            )
        expected_q_values = member.agent.q_values(car["observation"])
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
        # Environments are long-lived worker contexts; evolution resets them in
        # place instead of allocating a fresh physics world every generation.
        self.assertEqual(original_environments, manager.environments)
        self.assertTrue(
            all(
                item["generation"] == session.current_generation
                for item in manager.telemetry(include_rays=False)
            )
        )

    def test_manager_step_leaves_scored_trainer_state_bit_identical(self):
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
        self.assertTrue(all(item["steps"] == 0 for item in manager.telemetry()))


if __name__ == "__main__":
    unittest.main()
