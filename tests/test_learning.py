import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from snakeGameQDlearning.src.game import SnakeGameAI
from snakeGameQDlearning.src.ml.agent import Agent
from snakeGameQDlearning.src.ml.models import LinearQNet
from snakeGameQDlearning.src.ml.replay import Experience, ReplayBuffer
from snakeGameQDlearning.src.ml.trainer import QTrainer


class LearningTests(unittest.TestCase):
    def test_observation_has_eleven_binary_features(self):
        game = SnakeGameAI(render=False, seed=2)
        state = Agent(seed=2).get_state(game)
        self.assertEqual(state.shape, (11,))
        self.assertTrue(set(state).issubset({0.0, 1.0}))

    def test_replay_buffer_is_bounded(self):
        buffer = ReplayBuffer(2)
        state = np.zeros(11, dtype=np.float32)
        for reward in range(3):
            buffer.append(Experience(state, [1, 0, 0], reward, state, False))
        self.assertEqual(len(buffer), 2)
        self.assertEqual(buffer.sample(2)[-1].reward, 2)

    def test_dqn_and_double_dqn_use_different_selection_rules(self):
        state = torch.zeros((1, 11))
        values = {}
        for algorithm in ("dqn", "double_dqn"):
            model = LinearQNet(11, 8, 3)
            trainer = QTrainer(model, learning_rate=0.001, gamma=0.9, algorithm=algorithm)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                for parameter in trainer.target_model.parameters():
                    parameter.zero_()
                model.linear3.bias.copy_(torch.tensor([4.0, 0.0, 0.0]))
                trainer.target_model.linear3.bias.copy_(torch.tensor([1.0, 5.0, 2.0]))
            values[algorithm] = float(trainer._bootstrap_values(state)[0])
        self.assertEqual(values["dqn"], 5.0)
        self.assertEqual(values["double_dqn"], 1.0)

    def test_training_step_reports_finite_loss(self):
        agent = Agent(algorithm="double_dqn", seed=3)
        state = np.zeros(11, dtype=np.float32)
        loss = agent.train_short_memory(state, [1, 0, 0], 1.0, state, False)
        self.assertTrue(np.isfinite(loss))

    def test_legacy_checkpoint_migration_preserves_predictions(self):
        legacy = {
            "linear1.weight": torch.randn(256, 11),
            "linear1.bias": torch.randn(256),
            "linear2.weight": torch.randn(3, 256),
            "linear2.bias": torch.randn(3),
        }
        sample = torch.randn(11)
        expected = torch.nn.functional.linear(
            torch.relu(torch.nn.functional.linear(sample, legacy["linear1.weight"], legacy["linear1.bias"])),
            legacy["linear2.weight"], legacy["linear2.bias"],
        )
        with tempfile.TemporaryDirectory() as directory:
            torch.save(legacy, os.path.join(directory, "legacy.pth"))
            model = LinearQNet(11, 512, 3)
            model.load("legacy.pth", directory)
        self.assertTrue(torch.allclose(model(sample), expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
