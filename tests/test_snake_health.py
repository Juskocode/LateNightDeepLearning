"""Truthful health telemetry and hardened Snake learning boundaries."""

import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from snakeGameQDlearning.src.game import SnakeGameAI
from snakeGameQDlearning.src.ml.agent import Agent
from snakeGameQDlearning.src.ml.algorithms import TabularAlgorithm
from snakeGameQDlearning.src.ml.models import LinearQNet
from snakeGameQDlearning.src.ml.replay import Experience, ReplayBuffer
from snakeGameQDlearning.src.utils.helpers import _atomic_json_dump


class SnakeHealthTests(unittest.TestCase):
    def test_deep_health_reports_stable_warmup_contract(self):
        agent = Agent(algorithm="double_dqn", seed=31)
        game = SnakeGameAI(render=False, seed=31)
        state = agent.get_state(game)
        agent.get_action(state)

        health = agent.telemetry(state, game)["health"]

        self.assertEqual(health["status"], "warming_up")
        self.assertTrue(health["finite"])
        self.assertEqual(health["alerts"], ["optimizer_warming_up", "replay_warming_up"])
        self.assertTrue(health["replay"]["applicable"])
        self.assertFalse(health["replay"]["ready"])
        self.assertEqual(health["optimization"]["updates"], 0)
        self.assertEqual(health["optimization"]["decisions"], 1)
        self.assertTrue(health["optimization"]["gradient_applicable"])
        self.assertIn("q_abs_max", health["values"])
        self.assertEqual(health["generalization"]["status"], "not_evaluated")

    def test_successful_update_reports_td_and_update_ratio(self):
        agent = Agent(algorithm="double_dqn", seed=32)
        game = SnakeGameAI(render=False, seed=32)
        state = agent.get_state(game)
        action = agent.get_action(state)
        agent.train_short_memory(state, action, 2.0, state, False)

        health = agent.telemetry(state, game)["health"]

        self.assertEqual(health["optimization"]["updates"], 1)
        self.assertEqual(health["optimization"]["update_to_decision_ratio"], 1.0)
        self.assertGreaterEqual(health["values"]["td_error_abs_mean"], 0.0)
        self.assertTrue(np.isfinite(health["optimization"]["gradient_norm"]))

    def test_tabular_metrics_mark_neural_and_sarsa_replay_not_applicable(self):
        agent = Agent(algorithm="sarsa", seed=33)
        game = SnakeGameAI(render=False, seed=33)
        state = agent.get_state(game)
        action = agent.get_action(state)
        agent.train_short_memory(state, action, 1.0, state, False)
        agent.remember(state, action, 1.0, state, False)

        health = agent.telemetry(state, game)["health"]

        self.assertFalse(health["neural"]["applicable"])
        self.assertIsNone(health["neural"]["parameters_finite"])
        self.assertFalse(health["replay"]["applicable"])
        self.assertIsNone(health["replay"]["ready"])
        self.assertFalse(health["optimization"]["gradient_applicable"])
        self.assertIsNone(health["optimization"]["gradient_norm"])
        self.assertEqual(len(agent.memory), 0)

    def test_invalid_transition_cannot_mutate_replay_or_q_table(self):
        state = np.zeros(11, dtype=np.float32)
        poisoned = state.copy()
        poisoned[4] = np.nan
        replay = ReplayBuffer(4)
        with self.assertRaises(ValueError):
            replay.append(Experience(poisoned, [1, 0, 0], 1.0, state, False))
        self.assertEqual(len(replay), 0)

        agent = Agent(algorithm="q_learning", seed=34)
        with self.assertRaises(ValueError):
            agent.train_short_memory(state, [1, 0, 0], float("inf"), state, False)
        self.assertEqual(agent.learning.learned_states, 0)

    def test_float32_overflow_is_rejected_before_replay_or_table_commit(self):
        state = np.zeros(11, dtype=np.float32)
        replay = ReplayBuffer(4)
        with self.assertRaisesRegex(ValueError, "float32"):
            replay.append(Experience(state, [1, 0, 0], 1e300, state, True))
        self.assertEqual(len(replay), 0)

        algorithm = TabularAlgorithm("q_learning", learning_rate=1.0, gamma=1.0)
        next_state = state.copy()
        next_state[0] = 1.0
        next_key = algorithm.encode_state(next_state)
        algorithm.table[next_key] = np.full(
            3, np.finfo(np.float32).max, dtype=np.float32
        )
        before = {key: value.copy() for key, value in algorithm.table.items()}
        with self.assertRaisesRegex(FloatingPointError, "float32"):
            algorithm.train_transition(
                Experience(
                    state,
                    [1, 0, 0],
                    float(np.finfo(np.float32).max),
                    next_state,
                    False,
                ),
                epsilon=0.0,
            )
        self.assertEqual(algorithm.update_count, 0)
        self.assertEqual(algorithm.rejected_update_count, 1)
        self.assertEqual(set(algorithm.table), set(before))
        for key, values in before.items():
            np.testing.assert_array_equal(algorithm.table[key], values)

    def test_transition_rejections_do_not_inflate_optimizer_attempts(self):
        agent = Agent(algorithm="q_learning", seed=341)
        state = np.zeros(11, dtype=np.float32)
        with self.assertRaises(ValueError):
            agent.train_short_memory(state, [1, 0, 0], 1e300, state, False)

        health = agent.telemetry(state)["health"]["optimization"]
        self.assertEqual(health["attempted_updates"], 0)
        self.assertEqual(health["rejected_updates"], 0)
        self.assertEqual(health["rejected_transitions"], 1)
        self.assertEqual(health["rejected_total"], 1)

    def test_rejected_neural_update_is_counted_without_parameter_update(self):
        agent = Agent(algorithm="double_dqn", seed=35)
        state = np.zeros(11, dtype=np.float32)
        before = [parameter.detach().clone() for parameter in agent.model.parameters()]

        with self.assertRaises(ValueError):
            agent.trainer.train_step(state, [1, 0, 0], np.nan, state, False)

        self.assertEqual(agent.trainer.rejected_updates, 1)
        self.assertEqual(agent.trainer.update_target_counter, 0)
        for expected, actual in zip(before, agent.model.parameters()):
            self.assertTrue(np.array_equal(expected.numpy(), actual.detach().numpy()))

    def test_termination_and_evaluation_diagnostics_are_explicit(self):
        agent = Agent(algorithm="q_learning", seed=36)
        game = SnakeGameAI(render=False, seed=36)
        game.termination_reason = "wall"
        agent.calculate_reward(game, True, 0, 0)
        agent.update_evaluation_metrics(
            {"episodes": 3, "mean_score": 1.0, "std_score": 0.5},
            training_mean=2.5,
        )

        health = agent.telemetry(agent.get_state(game), game)["health"]

        self.assertEqual(health["terminations"]["counts"]["wall"], 1)
        self.assertEqual(health["terminations"]["collision_rate"], 1.0)
        self.assertEqual(health["generalization"]["status"], "evaluated")
        self.assertEqual(health["generalization"]["generalization_gap"], 1.5)

    def test_rejected_neural_checkpoint_cannot_partially_mutate_model(self):
        model = LinearQNet(11, 32, 3, dropout=0.0)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        poisoned = {name: value.detach().clone() for name, value in before.items()}
        poisoned["linear1.weight"][0, 0] = float("nan")

        with tempfile.TemporaryDirectory() as directory:
            torch.save(poisoned, Path(directory) / "poisoned.pth")
            with self.assertRaises(ValueError):
                model.load("poisoned.pth", directory)

        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_nonfinite_neural_policy_cannot_be_saved(self):
        model = LinearQNet(11, 32, 3, dropout=0.0)
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] = torch.nan

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "non-finite"):
                model.save("poisoned.pth", directory)
            self.assertFalse((Path(directory) / "poisoned.pth").exists())

    def test_rejected_tabular_checkpoint_preserves_policy_and_config(self):
        algorithm = TabularAlgorithm("q_learning", learning_rate=0.25, gamma=0.8)
        state = np.zeros(11, dtype=np.float32)
        algorithm.train_transition(
            Experience(state, [1, 0, 0], 2.0, state, True), epsilon=0.1
        )
        before = {key: value.copy() for key, value in algorithm.table.items()}
        payload = {
            "algorithm": "q_learning",
            "learning_rate": 0.5,
            "gamma": 0.9,
            "q_table": {0: torch.tensor([float("inf"), 0.0, 0.0])},
        }

        with tempfile.TemporaryDirectory() as directory:
            torch.save(payload, Path(directory) / "poisoned.pth")
            with self.assertRaises(ValueError):
                algorithm.load("poisoned.pth", Path(directory))

        self.assertEqual(algorithm.learning_rate, 0.25)
        self.assertEqual(algorithm.gamma, 0.8)
        self.assertEqual(set(algorithm.table), set(before))
        for key, value in before.items():
            np.testing.assert_array_equal(algorithm.table[key], value)

    def test_corrupt_checkpoint_is_a_recoverable_load_failure(self):
        agent = Agent(algorithm="double_dqn", seed=37)
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "bad.pth").write_bytes(b"not a torch checkpoint")
            with patch("snakeGameQDlearning.src.ml.agent.MODEL_DIR", model_dir):
                loaded = agent._load_model_info(
                    (
                        "bad.pth",
                        {
                            "version": 1,
                            "best_score": 0,
                            "games_played": 0,
                            "mean_score": 0.0,
                        },
                    )
                )
        self.assertFalse(loaded)

    def test_metadata_replace_is_atomic_on_encoding_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "metadata.json"
            destination.write_text('{"stable": true}', encoding="utf-8")
            with patch(
                "snakeGameQDlearning.src.utils.helpers.json.dump",
                side_effect=ValueError("synthetic encoding failure"),
            ):
                with self.assertRaises(ValueError):
                    _atomic_json_dump(destination, {"new": True})
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"stable": True},
            )

    def test_failed_metadata_write_does_not_advance_version_or_leave_model(self):
        agent = Agent(algorithm="q_learning", seed=38)
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            with patch("snakeGameQDlearning.src.ml.agent.MODEL_DIR", model_dir), patch(
                "snakeGameQDlearning.src.ml.agent.save_model_metadata",
                side_effect=OSError("synthetic metadata failure"),
            ):
                with self.assertRaises(OSError):
                    agent.save_model_checkpoint(0, 0.0, reason="test")
            self.assertIsNone(agent.current_version)
            self.assertEqual(list(model_dir.glob("*.pth")), [])

    def test_numpy_integer_seed_is_normalized_for_python_rng(self):
        game = SnakeGameAI(render=False, seed=np.int64(39))
        self.assertEqual(game.episode_seed, 39)
        game.reset(seed=np.int64(40))
        self.assertEqual(game.episode_seed, 40)


if __name__ == "__main__":
    unittest.main()
