"""Deterministic CPU coverage for the Driving Lab DQN stack."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from drivingGameRL.src.ml import (
    DQNConfig,
    DrivingDQNAgent,
    DrivingQNetwork,
    ReplayBuffer,
)


def tiny_config(**changes) -> DQNConfig:
    values = {
        "hidden_sizes": (8,),
        "replay_capacity": 16,
        "batch_size": 2,
        "warmup_steps": 0,
        "target_sync_interval": 3,
        "epsilon_decay_steps": 10,
        "seed": 17,
    }
    values.update(changes)
    return DQNConfig(**values)


class DQNConfigTests(unittest.TestCase):
    def test_defaults_match_the_driving_environment_contract(self):
        config = DQNConfig()

        self.assertEqual(config.observation_size, 16)
        self.assertEqual(config.action_size, 5)
        self.assertEqual(config.input_size, 16)
        self.assertEqual(config.output_size, 5)
        self.assertEqual(DQNConfig.from_dict(config.to_dict()), config)

    def test_invalid_hyperparameters_are_rejected_early(self):
        invalid = (
            {"observation_size": 0},
            {"action_size": True},
            {"hidden_sizes": ()},
            {"hidden_sizes": (8, 0)},
            {"algorithm": "rainbow"},
            {"replay_capacity": 2, "batch_size": 3},
            {"gamma": 1.1},
            {"learning_rate": float("nan")},
            {"epsilon_start": 0.1, "epsilon_end": 0.2},
            {"epsilon_decay_steps": 0},
            {"seed": -1},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                DQNConfig(**changes)


class DrivingQNetworkTests(unittest.TestCase):
    def test_forward_shape_and_parameter_count_are_real(self):
        network = DrivingQNetwork(12, 5, (4,))

        output = network(torch.zeros(3, 12))

        self.assertEqual(tuple(output.shape), (3, 5))
        self.assertEqual(network.architecture, (12, 4, 5))
        self.assertEqual(network.parameter_count, 12 * 4 + 4 + 4 * 5 + 5)

    def test_snapshot_contains_exact_live_activations_and_weights(self):
        network = DrivingQNetwork(2, 2, (2,))
        with torch.no_grad():
            network.layers[0].weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 0.5]]))
            network.layers[0].bias.copy_(torch.tensor([0.0, 1.0]))
            network.layers[1].weight.copy_(torch.tensor([[2.0, 0.0], [0.0, -1.0]]))
            network.layers[1].bias.zero_()

        snapshot = network.snapshot([3.0, 1.0])

        self.assertEqual(snapshot["architecture"], [2, 2, 2])
        self.assertEqual(snapshot["layers"][1]["pre_activations"], [2.0, 3.0])
        self.assertEqual(snapshot["layers"][1]["activations"], [2.0, 3.0])
        self.assertEqual(snapshot["q_values"], [4.0, -3.0])
        self.assertEqual(
            snapshot["layers"][2]["weights"],
            network.layers[1].weight.detach().tolist(),
        )


class ReplayBufferTests(unittest.TestCase):
    def test_buffer_is_bounded_and_defensively_copies_observations(self):
        replay = ReplayBuffer(2, observation_size=3, seed=2)
        state = np.zeros(3, dtype=np.float32)
        replay.append(state, 0, 0.0, state, False)
        state[:] = 99.0
        replay.append(np.ones(3), 1, 1.0, np.ones(3), False)
        replay.append(np.full(3, 2.0), 2, 2.0, np.full(3, 2.0), True)

        self.assertEqual(len(replay), 2)
        np.testing.assert_array_equal(next(iter(replay)).state, np.ones(3))
        self.assertEqual(replay.stats()["terminal"], 1)
        self.assertAlmostEqual(replay.stats()["mean_reward"], 1.5)

    def test_sampling_is_reproducible_and_independent(self):
        first = ReplayBuffer(10, observation_size=2, seed=7)
        second = ReplayBuffer(10, observation_size=2, seed=7)
        for action in range(10):
            state = np.full(2, action, dtype=np.float32)
            first.append(state, action, float(action), state, False)
            second.append(state, action, float(action), state, False)

        self.assertEqual(
            [item.action for item in first.sample(5)],
            [item.action for item in second.sample(5)],
        )
        with self.assertRaises(ValueError):
            first.sample(11)


class DrivingDQNAgentTests(unittest.TestCase):
    def test_seed_controls_initial_weights_and_exploration(self):
        first = DrivingDQNAgent(tiny_config(epsilon_start=1.0, epsilon_end=1.0))
        second = DrivingDQNAgent(tiny_config(epsilon_start=1.0, epsilon_end=1.0))
        state = np.linspace(-1.0, 1.0, 16, dtype=np.float32)

        np.testing.assert_array_equal(first.q_values(state), second.q_values(state))
        self.assertEqual(
            [first.select_action(state) for _ in range(12)],
            [second.select_action(state) for _ in range(12)],
        )

    def test_epsilon_decays_linearly_to_its_floor(self):
        agent = DrivingDQNAgent(
            tiny_config(epsilon_start=1.0, epsilon_end=0.2, epsilon_decay_steps=10)
        )

        self.assertAlmostEqual(agent.epsilon, 1.0)
        agent.environment_steps = 5
        self.assertAlmostEqual(agent.epsilon, 0.6)
        agent.environment_steps = 500
        self.assertAlmostEqual(agent.epsilon, 0.2)

    def test_observe_runs_a_finite_real_td_update(self):
        agent = DrivingDQNAgent(tiny_config(learning_rate=0.01))
        state = np.zeros(16, dtype=np.float32)
        before = agent.q_values(state)

        self.assertIsNone(agent.observe(state, 0, 3.0, state, True))
        loss = agent.observe(state, 0, 3.0, state, True)

        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.gradient_steps, 1)
        self.assertFalse(np.array_equal(before, agent.q_values(state)))

    def test_dqn_and_double_dqn_use_distinct_bootstrap_rules(self):
        dqn = DrivingDQNAgent(tiny_config(algorithm="dqn"))
        double = DrivingDQNAgent(tiny_config(algorithm="double_dqn"))
        for agent in (dqn, double):
            with torch.no_grad():
                for parameter in agent.online_network.parameters():
                    parameter.zero_()
                for parameter in agent.target_network.parameters():
                    parameter.zero_()
                agent.online_network.layers[-1].bias.copy_(
                    torch.tensor([0.0, 5.0, 1.0, 0.0, 0.0])
                )
                agent.target_network.layers[-1].bias.copy_(
                    torch.tensor([9.0, 2.0, 3.0, 4.0, 1.0])
                )
        states = torch.zeros(2, 16)

        self.assertEqual(dqn._bootstrap_values(states).tolist(), [9.0, 9.0])
        self.assertEqual(double._bootstrap_values(states).tolist(), [2.0, 2.0])

    def test_target_network_syncs_on_the_configured_gradient_step(self):
        agent = DrivingDQNAgent(tiny_config(target_sync_interval=1))
        state = np.zeros(16, dtype=np.float32)
        agent.observe(state, 0, 2.0, state, True)
        agent.observe(state, 0, 2.0, state, True)

        self.assertEqual(agent.target_syncs, 1)
        for online, target in zip(
            agent.online_network.parameters(), agent.target_network.parameters()
        ):
            self.assertTrue(torch.equal(online, target))

    def test_population_clone_is_equal_but_tensor_independent(self):
        parent = DrivingDQNAgent(tiny_config(seed=3))
        child = parent.clone(seed=9)
        state = np.zeros(16, dtype=np.float32)

        np.testing.assert_array_equal(parent.q_values(state), child.q_values(state))
        self.assertEqual(child.environment_steps, 0)
        self.assertEqual(len(child.replay), 0)
        with torch.no_grad():
            next(child.online_network.parameters()).add_(1.0)
        self.assertFalse(
            torch.equal(
                next(parent.online_network.parameters()),
                next(child.online_network.parameters()),
            )
        )

    def test_telemetry_and_network_snapshot_expose_live_learning_state(self):
        agent = DrivingDQNAgent(tiny_config())
        state = np.zeros(16, dtype=np.float32)
        action = agent.select_action(state, explore=False)

        telemetry = agent.telemetry(state)
        snapshot = agent.network_snapshot(state)

        self.assertEqual(telemetry["last_action"], action)
        self.assertEqual(len(telemetry["q_values"]), 5)
        self.assertEqual(telemetry["parameter_count"], snapshot["parameter_count"])
        self.assertEqual(snapshot["q_values"], telemetry["q_values"])
        self.assertEqual(len(snapshot["layers"]), 3)

    def test_atomic_checkpoint_round_trip_restores_policy_and_counters(self):
        agent = DrivingDQNAgent(tiny_config(seed=22))
        state = np.arange(16, dtype=np.float32) / 16.0
        agent.select_action(state)
        agent.observe(state, 2, 1.0, state, False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "driver.pth"
            self.assertEqual(agent.save(path), path.resolve())
            restored = DrivingDQNAgent.from_checkpoint(path)
            leftovers = list(path.parent.glob(f".{path.name}.*.tmp"))

        np.testing.assert_array_equal(agent.q_values(state), restored.q_values(state))
        self.assertEqual(restored.environment_steps, agent.environment_steps)
        self.assertEqual(restored.gradient_steps, agent.gradient_steps)
        self.assertEqual(restored.select_action(state), agent.select_action(state))
        self.assertEqual(leftovers, [])

    def test_checkpoint_rejects_an_incompatible_architecture(self):
        source = DrivingDQNAgent(tiny_config())
        incompatible = DrivingDQNAgent(tiny_config(hidden_sizes=(7,)))

        with self.assertRaises(ValueError):
            incompatible.load_state_dict(source.state_dict())

    def test_legacy_five_ray_checkpoint_expands_without_changing_predictions(self):
        legacy = DrivingDQNAgent(tiny_config(observation_size=12, seed=41))
        legacy_state = np.linspace(-0.8, 0.9, 12, dtype=np.float32)
        legacy.observe(legacy_state, 1, 0.7, legacy_state, False)
        legacy.observe(legacy_state, 3, -0.2, legacy_state, False)
        checkpoint = legacy.state_dict()

        expanded_state = np.full(16, 0.37, dtype=np.float32)
        expanded_state[:7] = legacy_state[:7]
        for old_ray, new_ray in enumerate((0, 2, 4, 6, 8)):
            expanded_state[7 + new_ray] = legacy_state[7 + old_ray]

        restored = DrivingDQNAgent(tiny_config(observation_size=16, seed=41))
        restored.load_state_dict(checkpoint)

        np.testing.assert_allclose(
            restored.q_values(expanded_state),
            legacy.q_values(legacy_state),
            rtol=0.0,
            atol=1e-7,
        )
        first_layer = restored.online_network.layers[0].weight.detach()
        self.assertTrue(torch.count_nonzero(first_layer[:, (8, 10, 12, 14)]) == 0)
        # Adam moment tensors are expanded too, so resumed training remains
        # valid instead of failing on its first optimizer update.
        restored.observe(expanded_state, 1, 0.3, expanded_state, False)
        loss = restored.observe(expanded_state, 1, 0.3, expanded_state, False)
        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))

    def test_from_checkpoint_automatically_upgrades_legacy_observation_contract(self):
        legacy = DrivingDQNAgent(tiny_config(observation_size=12, seed=42))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-five-rays.pth"
            legacy.save(path)
            restored = DrivingDQNAgent.from_checkpoint(path)

        self.assertEqual(restored.config.observation_size, 16)
        self.assertEqual(restored.online_network.architecture[0], 16)


if __name__ == "__main__":
    unittest.main()
