import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from PIL import Image
import pygame
import torch

from pacManRf.src.game.pacman_env import ACTION_LABELS, OBSERVATION_LABELS, PacmanEnv
from pacManRf.src.ml import DQNConfig, PacmanDQNAgent, PacmanQNetwork
from pacManRf.src.ml.trainer import DQNTrainer
from pacManRf.src.observatory_capture import capture_observatory_gif
from pacManRf.src.rl_session import PacmanRLSession, SessionConfig, WINDOW_SIZE
from pacManRf.src.visualization import PacmanObservatory


def small_session(seed=9):
    return PacmanRLSession(
        SessionConfig(
            seed=seed,
            fresh=True,
            hidden_sizes=(32, 16),
            batch_size=4,
            replay_capacity=128,
            replay_warmup=4,
            target_update_interval=4,
            epsilon_decay_steps=100,
        )
    )


class PacmanEnvironmentTests(unittest.TestCase):
    def test_observation_contract_is_named_normalized_and_finite(self):
        env = PacmanEnv(seed=2)
        state = env.reset(seed=2)

        self.assertEqual(state.shape, (32,))
        self.assertEqual(len(OBSERVATION_LABELS), 32)
        self.assertEqual(len(ACTION_LABELS), 4)
        self.assertTrue(np.isfinite(state).all())
        self.assertTrue(((0.0 <= state) & (state <= 1.0)).all())
        self.assertEqual(set(env.observation_dict()), set(OBSERVATION_LABELS))

    def test_seeded_trajectories_are_reproducible(self):
        first = PacmanEnv(seed=11)
        second = PacmanEnv(seed=11)
        np.testing.assert_array_equal(first.reset(seed=11), second.reset(seed=11))

        for action in (0, 1, 0, 2, 3, 1, 0, 0):
            first_step = first.step(action)
            second_step = second.step(action)
            np.testing.assert_array_equal(first_step[0], second_step[0])
            self.assertEqual(first_step[1], second_step[1])
            self.assertEqual(first_step[2], second_step[2])
            self.assertEqual(first_step[3]["reward_components"], second_step[3]["reward_components"])
            if first_step[2]:
                break

    def test_actions_accept_strict_one_hot_only(self):
        env = PacmanEnv(seed=3)
        env.step([1, 0, 0, 0])
        with self.assertRaises(ValueError):
            env.step([1, 1, 0, 0])
        with self.assertRaises(ValueError):
            env.step(4)


class PacmanLearningTests(unittest.TestCase):
    def test_dqn_and_double_dqn_have_distinct_bootstrap_rules(self):
        next_state = torch.zeros((1, 32))
        values = {}
        for algorithm in ("dqn", "double_dqn"):
            config = DQNConfig(
                observation_size=32,
                action_size=4,
                hidden_sizes=(8,),
                action_labels=ACTION_LABELS,
                algorithm=algorithm,
                batch_size=1,
                replay_capacity=8,
                replay_warmup=0,
            )
            model = PacmanQNetwork(32, 4, (8,))
            trainer = DQNTrainer(model, config)
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.zero_()
                for parameter in trainer.target_model.parameters():
                    parameter.zero_()
                trainer.model.layers[-1].bias.copy_(torch.tensor([4.0, 0.0, 0.0, 0.0]))
                trainer.target_model.layers[-1].bias.copy_(torch.tensor([1.0, 5.0, 2.0, 0.0]))
            values[algorithm] = float(trainer._bootstrap_values(next_state)[0])

        self.assertEqual(values["dqn"], 5.0)
        self.assertEqual(values["double_dqn"], 1.0)

    def test_bootstrap_excludes_illegal_next_actions(self):
        for algorithm in ("dqn", "double_dqn"):
            with self.subTest(algorithm=algorithm):
                config = DQNConfig(
                    observation_size=32,
                    action_size=4,
                    hidden_sizes=(8,),
                    action_labels=ACTION_LABELS,
                    algorithm=algorithm,
                    batch_size=1,
                    replay_capacity=8,
                    replay_warmup=0,
                )
                model = PacmanQNetwork(32, 4, (8,))
                trainer = DQNTrainer(model, config)
                with torch.no_grad():
                    for parameter in trainer.model.parameters():
                        parameter.zero_()
                    for parameter in trainer.target_model.parameters():
                        parameter.zero_()
                    trainer.model.layers[-1].bias.copy_(torch.tensor([100.0, 4.0, 3.0, 2.0]))
                    trainer.target_model.layers[-1].bias.copy_(torch.tensor([50.0, 7.0, 6.0, 5.0]))

                bootstrap = trainer._bootstrap_values(
                    torch.zeros((1, 32)),
                    np.asarray([[False, True, True, True]]),
                )

                self.assertEqual(float(bootstrap[0]), 7.0)

    def test_session_exposes_real_network_and_learning_metrics(self):
        session = small_session()
        for _ in range(8):
            session.step()
        telemetry = session.telemetry(max_neurons_per_layer=7)

        self.assertEqual(len(session.agent.memory), 8)
        self.assertGreater(telemetry["loss"], 0.0)
        self.assertEqual(telemetry["network"]["architecture"], [32, 32, 16, 4])
        self.assertEqual(
            [layer["full_size"] for layer in telemetry["network"]["layers"]],
            telemetry["network"]["architecture"],
        )
        self.assertEqual(len(telemetry["network"]["weights"]), 3)
        self.assertEqual(telemetry["action_labels"], list(ACTION_LABELS))
        self.assertEqual(len(telemetry["observation"]), 32)
        session.close()

    def test_checkpoint_round_trip_preserves_predictions_and_replay(self):
        session = small_session(seed=13)
        for _ in range(6):
            session.step()
        expected = session.agent.trainer.predict(session.state)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pacman.pth"
            session.agent.save_checkpoint(path, include_replay=True)
            restored = PacmanDQNAgent.from_checkpoint(path)
            actual = restored.trainer.predict(session.state)

        np.testing.assert_allclose(actual, expected)
        self.assertEqual(len(restored.memory), len(session.agent.memory))
        np.testing.assert_array_equal(
            restored.memory.tail(1)[0].next_legal_action_mask,
            session.agent.memory.tail(1)[0].next_legal_action_mask,
        )
        session.close()

    def test_evaluation_counts_episodes_without_advancing_exploration(self):
        session = PacmanRLSession(
            SessionConfig(
                seed=23,
                training=False,
                fresh=True,
                hidden_sizes=(16, 8),
                max_episode_steps=1,
            )
        )
        epsilon = session.agent.epsilon

        result = session.step()

        self.assertTrue(result["episode_finished"])
        self.assertEqual(session.agent.episodes, 1)
        self.assertEqual(session.agent.env_steps, 0)
        self.assertEqual(session.agent.epsilon, epsilon)
        session.close()


class PacmanObservatoryTests(unittest.TestCase):
    def test_all_tabs_render_live_session_data(self):
        session = small_session(seed=17)
        for _ in range(5):
            session.step()
        surface = pygame.Surface(WINDOW_SIZE)
        ui = PacmanObservatory()

        for tab in ("GAME", "VISION", "METRICS", "NETWORK"):
            surface.fill((0, 0, 0))
            ui.set_tab(tab)
            ui.render(
                surface,
                session.telemetry(),
                history=session.history_snapshot(),
                game_surface=session.render_game(),
            )
            self.assertNotEqual(surface.get_at((20, 20))[:3], (0, 0, 0))
        session.close()

    def test_function_keys_and_tab_cycle_cover_all_four_views(self):
        ui = PacmanObservatory()
        expected = (
            (pygame.K_F1, "GAME"),
            (pygame.K_F2, "VISION"),
            (pygame.K_F3, "METRICS"),
            (pygame.K_F4, "NETWORK"),
        )
        for key, tab in expected:
            consumed = ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
            self.assertTrue(consumed)
            self.assertEqual(ui.active_tab.value, tab)

        ui.set_tab("GAME")
        cycled = []
        for _ in range(4):
            ui.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
            cycled.append(ui.active_tab.value)
        self.assertEqual(cycled, ["VISION", "METRICS", "NETWORK", "GAME"])

    def test_gif_capture_is_animated(self):
        session = small_session(seed=19)
        with tempfile.TemporaryDirectory() as directory:
            path = capture_observatory_gif(
                session,
                Path(directory) / "observatory.gif",
                frames=12,
                prime_steps=5,
                output_width=480,
                duration_ms=50,
            )
            with Image.open(path) as image:
                self.assertGreaterEqual(image.n_frames, 10)
                self.assertEqual(image.format, "GIF")
        session.close()


if __name__ == "__main__":
    unittest.main()
