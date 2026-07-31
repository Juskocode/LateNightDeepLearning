import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from PIL import Image
import pygame
import torch

from pacManRf.src.game.constants import Direction
from pacManRf.src.game.pacman_env import ACTION_LABELS, OBSERVATION_LABELS, PacmanEnv
from pacManRf.src.ml import DQNConfig, PacmanDQNAgent, PacmanQNetwork
from pacManRf.src.ml.trainer import DQNTrainer
from pacManRf.src.observatory_capture import capture_observatory_gif
from pacManRf.src.rl_session import (
    SPEED_PRESETS,
    DecisionScheduler,
    PacmanRLSession,
    SessionConfig,
    SpeedController,
    WINDOW_SIZE,
)
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
    @staticmethod
    def _suppress_ghosts(env):
        for ghost in env.game.ghosts:
            ghost.released = False
            ghost.release_timer = 999.0

    @staticmethod
    def _spawn_imminent_projectile(env, owner):
        env.game.level = 3
        env.game.projectile_system.reset_level(initial_cooldown_seconds=0.0)
        shot = env.game.projectile_system.try_fire(
            owner,
            env.game.level,
            (8, 15),
            (1, 0),
            env.game._projectile_cell_is_walkable,
            next_cell=env.game._projectile_next_cell,
        )
        if shot is None:
            raise AssertionError("test projectile did not spawn")
        shot.progress = 0.99
        return shot

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

    def test_level_clear_gives_huge_reward_and_preserves_run(self):
        env = PacmanEnv(seed=29, auto_advance_levels=True)
        env.game.maze = [
            [" " if cell in ".o" else cell for cell in row]
            for row in env.game.maze
        ]
        env.game.maze[15][8] = "."
        env.game.score = 1_234
        env.game.lives = 2

        _, reward, done, info = env.step(0)

        self.assertFalse(done)
        self.assertEqual(env.game.level, 2)
        self.assertEqual(env.game.score, 1_244)
        self.assertEqual(env.game.lives, 2)
        self.assertEqual(info["cleared_level"], 1)
        self.assertEqual(info["reward_components"]["level_cleared"], 500.0)
        self.assertGreater(reward, 500.0)
        self.assertEqual(env.episode_steps, 1)
        self.assertEqual(env.game._count_dots(), env.game.total_dots)

    def test_freeze_event_survives_multi_frame_step_and_penalizes_agent(self):
        env = PacmanEnv(seed=37)
        self._suppress_ghosts(env)
        self._spawn_imminent_projectile(env, "INKY")

        state, reward, done, info = env.step(0)

        self.assertEqual(state.shape, (32,))
        self.assertFalse(done)
        self.assertEqual(env.game.lives, 3)
        self.assertTrue(env.game.player_slow.active)
        self.assertAlmostEqual(env.game.player_speed_multiplier, 0.85)
        self.assertEqual(info["freeze_ball_hits"], 1)
        self.assertEqual(info["fireball_hits"], 0)
        self.assertEqual(info["reward_components"]["freeze_hit"], -5.0)
        self.assertLess(reward, 0.0)
        self.assertEqual(info["projectile_events"][0]["kind"], "freeze_ball")
        self.assertEqual(info["projectile_events"][0]["reason"], "hit_pacman")

    def test_fireball_uses_life_loss_penalty_without_double_penalty(self):
        env = PacmanEnv(seed=41)
        self._suppress_ghosts(env)
        env.game.score = 123
        self._spawn_imminent_projectile(env, "BLINKY")

        _, _, done, info = env.step(0)

        self.assertFalse(done)
        self.assertEqual(env.game.lives, 2)
        self.assertEqual(env.game.score, 123)
        self.assertTrue(info["life_lost"])
        self.assertEqual(info["fireball_hits"], 1)
        self.assertEqual(info["freeze_ball_hits"], 0)
        self.assertEqual(info["reward_components"]["life_lost"], -25.0)
        self.assertEqual(info["reward_components"]["freeze_hit"], 0.0)

    def test_projectile_rays_reuse_threat_features_without_resizing_state(self):
        env = PacmanEnv(seed=43)
        self._suppress_ghosts(env)
        env.game.level = 3
        env.game.player.reset_position((10, 3), Direction.RIGHT)
        baseline = env.game.projectile_system.active_projectiles
        self.assertFalse(baseline)
        before = env._get_observation()

        env.game.projectile_system.reset_level(initial_cooldown_seconds=0.0)
        shot = env.game.projectile_system.try_fire(
            "INKY",
            3,
            (4, 3),
            (1, 0),
            env.game._projectile_cell_is_walkable,
            next_cell=env.game._projectile_next_cell,
        )
        self.assertIsNotNone(shot)
        after = env._get_observation()

        self.assertEqual(after.shape, (32,))
        self.assertTrue(np.any(after[12:16] > before[12:16]))
        np.testing.assert_array_equal(after[:12], before[:12])
        np.testing.assert_array_equal(after[16:], before[16:])


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
        self.assertIn("projectiles", telemetry)
        self.assertIn("BLINKY", telemetry["projectiles"]["weapons"])
        self.assertIn("INKY", telemetry["projectiles"]["weapons"])
        self.assertEqual(telemetry["projectiles_active"], len(session.env.game.active_projectiles))
        session.close()

    def test_session_continues_across_level_clear_without_reset(self):
        session = small_session(seed=31)
        session.env.game.maze = [
            [" " if cell in ".o" else cell for cell in row]
            for row in session.env.game.maze
        ]
        session.env.game.maze[15][8] = "."
        session.env.game.score = 700
        session.env.game.lives = 2
        session.state = session.env._get_observation()
        session.pending_action = 0

        result = session.step()

        self.assertFalse(result["episode_finished"])
        self.assertEqual(session.env.game.level, 2)
        self.assertEqual(session.env.game.score, 710)
        self.assertEqual(session.env.game.lives, 2)
        self.assertEqual(session.agent.episodes, 0)
        self.assertFalse(session.agent.memory.tail(1)[0].done)
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


class PacmanSpeedControlTests(unittest.TestCase):
    def test_presets_cover_slow_through_max(self):
        self.assertEqual(SPEED_PRESETS[0], ("SLOW", 1))
        self.assertEqual(SPEED_PRESETS[-1], ("MAX", 240))
        values = [value for _, value in SPEED_PRESETS]
        self.assertEqual(values, sorted(values))

    def test_speed_steps_and_direct_keys_are_bounded(self):
        controller = SpeedController(30)
        self.assertEqual((controller.label, controller.value), ("FAST", 30))

        controller.handle_key(pygame.K_LEFTBRACKET)
        self.assertEqual((controller.label, controller.value), ("NORMAL", 15))
        controller.handle_key(pygame.K_RIGHTBRACKET)
        self.assertEqual(controller.value, 30)
        controller.handle_key(pygame.K_1)
        self.assertEqual(controller.value, 1)
        controller.handle_key(pygame.K_7)
        self.assertEqual(controller.value, 240)
        controller.step(1)
        self.assertEqual(controller.value, 240)
        controller.handle_key(pygame.K_HOME)
        self.assertEqual(controller.value, 1)
        controller.handle_key(pygame.K_END)
        self.assertEqual(controller.value, 240)
        self.assertFalse(controller.handle_key(pygame.K_a))

    def test_custom_speed_moves_to_adjacent_preset(self):
        slower = SpeedController(42)
        faster = SpeedController(42)
        self.assertEqual(slower.label, "CUSTOM")
        self.assertEqual(slower.step(-1), 30)
        self.assertEqual(faster.step(1), 60)
        self.assertEqual(SpeedController(0).value, 1)
        self.assertEqual(SpeedController(999).value, 240)

    def test_decision_scheduler_keeps_rendering_separate_and_bounded(self):
        scheduler = DecisionScheduler()
        total = sum(scheduler.steps_for_frame(0.1, 30) for _ in range(10))
        self.assertEqual(total, 30)

        scheduler.reset()
        self.assertEqual(scheduler.steps_for_frame(1 / 60, 240), 4)
        self.assertEqual(scheduler.steps_for_frame(0.25, 240), 16)
        self.assertEqual(scheduler.steps_for_frame(0.1, 30, paused=True), 0)
        self.assertEqual(scheduler.steps_for_frame(0.1, 30, paused=True, single_step=True), 1)


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

    def test_speed_scale_is_clickable_and_does_not_overlap_tabs(self):
        controller = SpeedController(30)
        telemetry = controller.telemetry()
        telemetry["algorithm"] = "double_dqn"
        ui = PacmanObservatory()

        for width in (620, 879, 880, 980, 1120):
            with self.subTest(width=width):
                layout = ui.render(pygame.Surface((width, 720)), telemetry)
                for rect in (*layout.tabs.values(), *layout.speed_presets):
                    self.assertTrue(layout.header.contains(rect))
                for tab_rect in layout.tabs.values():
                    overlaps_speed = any(
                        tab_rect.colliderect(rect) for rect in layout.speed_presets
                    )
                    self.assertFalse(overlaps_speed)

        layout = ui.render(pygame.Surface(WINDOW_SIZE), telemetry)
        self.assertEqual(len(layout.speed_presets), len(SPEED_PRESETS))
        for index, rect in enumerate(layout.speed_presets):
            self.assertEqual(ui.speed_preset_at(rect.center), index)
        self.assertIsNone(ui.speed_preset_at((0, 719)))

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
