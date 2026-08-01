"""Coverage for the interchangeable Snake learning backends."""

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from snakeGameQDlearning.src.game import SnakeGameAI
from snakeGameQDlearning.src.ml.agent import Agent
from snakeGameQDlearning.src.ml.algorithms import (
    ALGORITHM_REGISTRY,
    TabularAlgorithm,
    create_algorithm,
    normalize_algorithm_name,
)
from snakeGameQDlearning.src.ml.models import DuelingQNet, LinearQNet
from snakeGameQDlearning.src.ml.replay import Experience
from snakeGameQDlearning.src.ml.trainer import QTrainer
from snakeGameQDlearning.src.utils.helpers import (
    get_best_model_info,
    save_model_metadata,
)


class SnakeAlgorithmTests(unittest.TestCase):
    def test_factory_exposes_stable_educational_algorithms(self):
        self.assertEqual(
            tuple(ALGORITHM_REGISTRY),
            (
                "dqn",
                "double_dqn",
                "dueling_dqn",
                "dueling_double_dqn",
                "q_learning",
                "sarsa",
            ),
        )
        self.assertEqual(normalize_algorithm_name("tabular_q"), "q_learning")
        self.assertEqual(normalize_algorithm_name("expected_sarsa"), "sarsa")
        with self.assertRaises(ValueError):
            create_algorithm("not_an_algorithm")

    def test_dueling_network_centers_advantages(self):
        network = DuelingQNet(11, 16, 3, dropout=0.0)
        network.eval()
        sample = torch.randn(4, 11)
        with torch.no_grad():
            features = torch.relu(network.linear1(sample))
            features = torch.relu(network.linear2(features))
            values = network.value(features).squeeze(1)
            output = network(sample)
        self.assertEqual(tuple(output.shape), (4, 3))
        self.assertTrue(torch.allclose(output.mean(dim=1), values, atol=1e-6))

    def test_double_dqn_bootstrap_disables_dropout_during_action_selection(self):
        network = LinearQNet(11, 32, 3, dropout=0.9)
        trainer = QTrainer(
            network, learning_rate=0.001, gamma=0.9, algorithm="double_dqn"
        )
        observed_modes = []
        hook = network.register_forward_pre_hook(
            lambda module, _inputs: observed_modes.append(module.training)
        )
        network.train()

        try:
            first = trainer._bootstrap_values(torch.ones(16, 11))
            second = trainer._bootstrap_values(torch.ones(16, 11))
        finally:
            hook.remove()

        self.assertEqual(observed_modes, [False, False])
        self.assertTrue(network.training)
        self.assertTrue(torch.equal(first, second))

    def test_each_algorithm_can_learn_one_transition(self):
        state = np.zeros(11, dtype=np.float32)
        next_state = state.copy()
        next_state[0] = 1.0
        for name in ALGORITHM_REGISTRY:
            with self.subTest(algorithm=name):
                agent = Agent(algorithm=name, seed=12)
                loss = agent.train_short_memory(
                    state, [1, 0, 0], 2.0, next_state, False
                )
                self.assertTrue(np.isfinite(loss))
                self.assertEqual(agent.algorithm, name)

    def test_q_learning_bootstraps_from_greedy_action(self):
        algorithm = TabularAlgorithm("q_learning", learning_rate=1.0, gamma=0.5)
        state = np.zeros(11, dtype=np.float32)
        next_state = state.copy()
        next_state[1] = 1.0
        algorithm.table[algorithm.encode_state(next_state)] = np.array(
            [1.0, 4.0, 2.0], dtype=np.float32
        )
        experience = Experience(state, [1, 0, 0], 3.0, next_state, False)

        algorithm.train_transition(experience, epsilon=0.9)

        self.assertAlmostEqual(float(algorithm.predict(state)[0]), 5.0)

    def test_sarsa_is_labeled_expected_and_does_not_replay(self):
        agent = Agent(algorithm="sarsa", seed=13)
        self.assertIn("Expected SARSA", agent.learning.info.description)
        state = np.zeros(11, dtype=np.float32)
        agent.remember(state, [1, 0, 0], 3.0, state, False)

        loss = agent.train_long_memory()

        self.assertEqual(loss, 0.0)
        self.assertEqual(agent.learning.learned_states, 0)

    def test_tabular_prediction_does_not_create_unseen_rows(self):
        algorithm = TabularAlgorithm("q_learning")
        state = np.zeros(11, dtype=np.float32)

        values = algorithm.predict(state)

        self.assertEqual(values.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(algorithm.learned_states, 0)

    def test_tabular_checkpoint_round_trip(self):
        state = np.zeros(11, dtype=np.float32)
        trained = TabularAlgorithm("q_learning", learning_rate=1.0)
        trained.train_transition(
            Experience(state, [0, 1, 0], 4.0, state, True), epsilon=0.2
        )
        restored = TabularAlgorithm("q_learning")
        with tempfile.TemporaryDirectory() as directory:
            trained.save("table.pth", Path(directory))
            restored.load("table.pth", Path(directory))

        np.testing.assert_array_equal(restored.predict(state), trained.predict(state))

    def test_best_checkpoint_prefers_held_out_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            save_model_metadata(
                directory,
                1,
                best_score=20,
                mean_score=8.0,
                games=100,
                algorithm="double_dqn",
            )
            save_model_metadata(
                directory,
                2,
                best_score=4,
                mean_score=2.0,
                games=50,
                algorithm="double_dqn",
                evaluation_mean=1.5,
            )
            save_model_metadata(
                directory,
                3,
                best_score=3,
                mean_score=1.0,
                games=60,
                algorithm="double_dqn",
                evaluation_mean=2.5,
                experiment={
                    "environment": "standard",
                    "validation_seed_root": 1001,
                    "validation_seeds": [11, 22, 33],
                    "final_test_seed_root": 2001,
                    "final_test_seeds": [44, 55],
                    "evaluation_round": 3,
                },
            )

            model_file, metadata = get_best_model_info(
                directory, algorithm="double_dqn"
            )

        self.assertEqual(model_file, "model_v003.pth")
        self.assertEqual(metadata["evaluation_mean"], 2.5)
        self.assertEqual(metadata["experiment"]["validation_seeds"], [11, 22, 33])
        self.assertEqual(metadata["experiment"]["final_test_seeds"], [44, 55])
        self.assertEqual(metadata["experiment"]["evaluation_round"], 3)

    def test_checkpoint_selection_never_compares_different_validation_suites(self):
        with tempfile.TemporaryDirectory() as directory:
            for version, validation_mean, seeds in (
                (1, 20.0, [101, 102]),
                (2, 21.0, [101, 102]),
                (3, 2.0, [201, 202]),
                (4, 3.0, [201, 202]),
            ):
                save_model_metadata(
                    directory,
                    version,
                    best_score=version,
                    mean_score=float(version),
                    games=version * 10,
                    algorithm="double_dqn",
                    evaluation_mean=validation_mean,
                    experiment={
                        "environment": "standard",
                        "validation_seeds": seeds,
                    },
                )

            latest_experiment = get_best_model_info(
                directory, algorithm="double_dqn", environment="standard"
            )
            first_suite = get_best_model_info(
                directory,
                algorithm="double_dqn",
                environment="standard",
                validation_seeds=(101, 102),
            )

        self.assertEqual(latest_experiment[0], "model_v004.pth")
        self.assertEqual(first_suite[0], "model_v002.pth")

    def test_greedy_action_is_deterministic(self):
        game = SnakeGameAI(render=False, seed=3)
        agent = Agent(algorithm="dueling_double_dqn", seed=3)
        state = agent.get_state(game)

        actions = [agent.get_action(state, explore=False) for _ in range(5)]

        self.assertTrue(all(action == actions[0] for action in actions))
        self.assertEqual(agent.last_policy_mode, "evaluate")


if __name__ == "__main__":
    unittest.main()
