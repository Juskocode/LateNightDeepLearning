import os
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import torch

from pacManRf.src.game.pacman_env import PacmanEnv, RewardConfig
from pacManRf.src.ml import DQNConfig, PacmanDQNAgent
from pacManRf.src.ml.replay import Experience, ReplayBuffer
from pacManRf.src.rl_session import PacmanRLSession, SessionConfig, WINDOW_SIZE
from pacManRf.src.visualization import PacmanObservatory


def tiny_agent(*, gradient_clip=1.0, seed=71):
    return PacmanDQNAgent(
        DQNConfig(
            observation_size=4,
            action_size=2,
            hidden_sizes=(8,),
            action_labels=("LEFT", "RIGHT"),
            batch_size=2,
            replay_capacity=8,
            replay_warmup=2,
            target_update_interval=4,
            epsilon_decay_steps=20,
            gradient_clip=gradient_clip,
            seed=seed,
        )
    )


def append_transition(agent, reward=1.0, done=False):
    state = np.asarray([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    next_state = np.asarray([0.1, 0.3, 0.6, 0.9], dtype=np.float32)
    action = agent.select_action(
        state,
        explore=False,
        legal_action_mask=[1, 1],
    )
    return agent.observe(state, action, reward, next_state, done, [1, 1])


class PacmanHealthTelemetryTests(unittest.TestCase):
    def test_health_contract_moves_from_warmup_to_healthy(self):
        agent = tiny_agent()

        initial = agent.health_telemetry()
        self.assertEqual(initial["status"], "warming_up")
        self.assertTrue(initial["finite"])
        self.assertEqual(initial["alerts"], [])
        self.assertFalse(initial["replay"]["ready"])
        self.assertEqual(initial["replay"]["samples_until_ready"], 2)

        append_transition(agent, reward=-1.0)
        metrics = append_transition(agent, reward=2.0, done=True)
        self.assertIsNotNone(metrics)
        health = agent.health_telemetry()

        self.assertEqual(health["status"], "healthy")
        self.assertTrue(health["replay"]["ready"])
        self.assertEqual(health["optimization"]["updates"], 1)
        self.assertEqual(health["optimization"]["decisions"], 2)
        self.assertAlmostEqual(
            health["optimization"]["update_to_decision_ratio"],
            0.5,
        )
        self.assertIsInstance(health["values"]["q_abs_max"], float)
        self.assertGreaterEqual(health["values"]["td_error_abs_mean"], 0.0)
        self.assertEqual(health["recent"]["terminal_count"], 1)
        self.assertEqual(health["recent"]["window"], 2)

    def test_gradient_clipping_pressure_is_measured_not_inferred(self):
        agent = tiny_agent(gradient_clip=1e-8)
        append_transition(agent, reward=100.0)
        metrics = append_transition(agent, reward=100.0)

        self.assertIsNotNone(metrics)
        self.assertTrue(metrics.gradient_clipped)
        self.assertGreater(metrics.gradient_to_clip_ratio, 1.0)
        health = agent.health_telemetry()
        self.assertEqual(health["optimization"]["clip_events"], 1)
        self.assertTrue(health["optimization"]["clipped_last_update"])
        self.assertEqual(health["optimization"]["clip_ratio"], 1.0)
        self.assertGreater(health["optimization"]["gradient_to_clip_ratio"], 1.0)

    def test_non_finite_parameters_raise_a_critical_health_alert(self):
        agent = tiny_agent()
        with torch.no_grad():
            next(agent.model.parameters()).view(-1)[0] = torch.nan

        health = agent.health_telemetry()

        self.assertEqual(health["status"], "critical")
        self.assertFalse(health["finite"])
        self.assertIn("non_finite_learning_state", health["alerts"])
        self.assertTrue(
            any(name.startswith("online.") for name in health["numeric"]["non_finite_fields"])
        )

    def test_low_optimizer_update_coverage_is_visible_after_first_update(self):
        agent = tiny_agent()
        append_transition(agent)
        append_transition(agent)
        state = np.zeros(4, dtype=np.float32)
        for _ in range(98):
            agent.remember(state, 0, 0.0, state, False, [1, 1])

        health = agent.health_telemetry()

        self.assertEqual(health["status"], "warning")
        self.assertIn("optimizer_update_coverage_low", health["alerts"])
        self.assertLess(health["optimization"]["update_coverage"], 0.5)

    def test_live_session_adds_environment_and_termination_diagnostics(self):
        session = PacmanRLSession(
            SessionConfig(
                seed=73,
                fresh=True,
                hidden_sizes=(16, 8),
                batch_size=2,
                replay_capacity=16,
                replay_warmup=2,
                max_episode_steps=1,
            )
        )
        try:
            result = session.step()
            self.assertTrue(result["episode_finished"])
            health = session.telemetry()["health"]
            self.assertEqual(health["termination"]["last_reason"], "time_limit")
            self.assertEqual(health["termination"]["recent_counts"], {"time_limit": 1})
            self.assertIn("last_reward_components", health["environment"])
            self.assertEqual(health["recent"]["terminal_count"], 1)
        finally:
            session.close()


class PacmanBoundaryHardeningTests(unittest.TestCase):
    def test_transition_rejects_lossy_coercions_without_touching_replay(self):
        agent = tiny_agent()
        state = np.zeros(4, dtype=np.float32)
        invalid = (
            (1.75, 1.0, False, [1, 1]),
            (0, np.inf, False, [1, 1]),
            (0, 1e100, False, [1, 1]),
            (0, 1.0, 2, [1, 1]),
            (0, 1.0, False, [1, np.nan]),
        )

        for action, reward, done, mask in invalid:
            with self.subTest(action=action, reward=reward, done=done, mask=mask):
                with self.assertRaises((ValueError, FloatingPointError)):
                    agent.remember(state, action, reward, state, done, mask)
                self.assertEqual(len(agent.memory), 0)
                self.assertEqual(agent.transitions_observed, 0)

    def test_replay_rejects_bad_rng_atomically(self):
        replay = ReplayBuffer(4, observation_size=2, action_size=2, seed=5)
        replay.append(Experience(np.zeros(2), 0, 1.0, np.ones(2), False, [1, 1]))
        before = replay.tail(1)[0]
        payload = replay.state_dict()
        payload["items"][0]["reward"] = 9.0
        payload["rng_state"] = ("malformed",)

        with self.assertRaisesRegex(ValueError, "rng_state"):
            replay.load_state_dict(payload)

        after = replay.tail(1)[0]
        self.assertEqual(after.reward, before.reward)
        np.testing.assert_array_equal(after.state, before.state)

    def test_rejected_checkpoint_cannot_partially_restore_live_agent(self):
        agent = tiny_agent(seed=79)
        append_transition(agent)
        append_transition(agent)
        probe = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        prediction_before = agent.trainer.predict(probe).copy()
        steps_before = agent.env_steps
        transitions_before = agent.transitions_observed
        replay_before = agent.memory.tail(1)[0].reward

        with tempfile.TemporaryDirectory() as directory:
            good_path = Path(directory) / "good.pth"
            bad_path = Path(directory) / "bad.pth"
            agent.save_checkpoint(good_path, include_replay=True)
            payload = torch.load(good_path, map_location="cpu", weights_only=True)
            first_key = next(iter(payload["trainer"]["online_model"]))
            tensor = payload["trainer"]["online_model"][first_key]
            payload["trainer"]["online_model"][first_key] = tensor[:1]
            torch.save(payload, bad_path)

            with self.assertRaises((ValueError, RuntimeError)):
                agent.load_checkpoint(bad_path)

        np.testing.assert_allclose(agent.trainer.predict(probe), prediction_before)
        self.assertEqual(agent.env_steps, steps_before)
        self.assertEqual(agent.transitions_observed, transitions_before)
        self.assertEqual(agent.memory.tail(1)[0].reward, replay_before)

    def test_public_trainer_load_preflights_models_and_optimizer(self):
        agent = tiny_agent(seed=81)
        append_transition(agent)
        append_transition(agent)
        probe = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        expected = agent.trainer.predict(probe).copy()
        original_lr = agent.trainer.optimizer.param_groups[0]["lr"]

        malformed_target = deepcopy(agent.trainer.state_dict())
        online_key = next(iter(malformed_target["online_model"]))
        malformed_target["online_model"][online_key] = (
            malformed_target["online_model"][online_key] + 1.0
        )
        target_key = next(iter(malformed_target["target_model"]))
        malformed_target["target_model"][target_key] = malformed_target[
            "target_model"
        ][target_key][:1]
        with self.assertRaises(RuntimeError):
            agent.trainer.load_state_dict(malformed_target)
        np.testing.assert_allclose(agent.trainer.predict(probe), expected)

        bad_lr = deepcopy(agent.trainer.state_dict())
        bad_lr["optimizer"]["param_groups"][0]["lr"] = -1.0
        with self.assertRaisesRegex(ValueError, "out of range"):
            agent.trainer.load_state_dict(bad_lr)
        self.assertEqual(agent.trainer.optimizer.param_groups[0]["lr"], original_lr)

        bad_moment = deepcopy(agent.trainer.state_dict())
        first_state = next(iter(bad_moment["optimizer"]["state"].values()))
        first_state["exp_avg"] = first_state["exp_avg"].reshape(-1)[:1]
        with self.assertRaisesRegex(ValueError, "shape"):
            agent.trainer.load_state_dict(bad_moment)
        np.testing.assert_allclose(agent.trainer.predict(probe), expected)

    def test_legacy_checkpoint_without_new_health_fields_still_loads(self):
        agent = tiny_agent(seed=83)
        append_transition(agent)
        append_transition(agent, done=True)
        probe = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        expected = agent.trainer.predict(probe)

        with tempfile.TemporaryDirectory() as directory:
            modern_path = Path(directory) / "modern.pth"
            legacy_path = Path(directory) / "legacy.pth"
            agent.save_checkpoint(modern_path, include_replay=True)
            payload = torch.load(modern_path, map_location="cpu", weights_only=True)
            payload["agent"].pop("transitions_observed")
            payload["trainer"].pop("gradient_clip_events")
            payload["trainer"].pop("gradient_clip_history_complete")
            payload["trainer"].pop("recent_gradient_clips")
            for name in (
                "gradient_to_clip_ratio",
                "gradient_clipped",
                "predicted_q_abs_max",
                "target_q_abs_max",
                "td_error_abs_max",
            ):
                payload["trainer"]["last_metrics"].pop(name)
            for item in payload["replay"]["items"]:
                item.pop("next_legal_action_mask")
            torch.save(payload, legacy_path)

            restored = PacmanDQNAgent.from_checkpoint(legacy_path)

        np.testing.assert_allclose(restored.trainer.predict(probe), expected)
        self.assertEqual(restored.transitions_observed, restored.env_steps)
        self.assertEqual(restored.trainer.gradient_clip_events, 0)
        self.assertFalse(restored.trainer.gradient_clip_history_complete)
        self.assertIsNone(restored.health_telemetry()["optimization"]["clip_ratio"])
        self.assertTrue(restored.memory.tail(1)[0].next_legal_action_mask.all())

    def test_non_finite_configuration_is_rejected(self):
        for field in ("learning_rate", "gamma", "epsilon_start", "gradient_clip"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    DQNConfig(**{field: float("nan")})

    def test_environment_rejects_malformed_numeric_contracts(self):
        with self.assertRaises(ValueError):
            RewardConfig(step=float("nan"))
        with self.assertRaises(ValueError):
            PacmanEnv(seed=1.5)
        with self.assertRaises(ValueError):
            PacmanEnv(max_episode_steps=1.5)
        with self.assertRaises(ValueError):
            PacmanEnv(frame_dt=float("nan"))

        env = PacmanEnv(seed=89)
        try:
            env.game.player.grid_x = env.game.cols * 2
            with self.assertRaisesRegex(RuntimeError, "outside"):
                env._get_observation()
        finally:
            env.close()


class PacmanHealthRenderingTests(unittest.TestCase):
    def test_metrics_tab_visually_distinguishes_health_status(self):
        ui = PacmanObservatory(initial_tab="METRICS")
        surface = pygame.Surface(WINDOW_SIZE)
        base_health = {
            "finite": True,
            "alerts": [],
            "replay": {"size": 64, "capacity": 100, "fill_ratio": 0.64, "ready": True, "warmup_threshold": 64},
            "optimization": {"updates": 20, "update_to_decision_ratio": 0.25, "clip_ratio": 0.2, "gradient_to_clip_ratio": 0.2, "recent_clip_fraction": 0.1},
            "values": {"q_abs_max": 3.2, "td_error_abs_mean": 0.4},
        }
        healthy = dict(base_health, status="healthy")
        ui.render(surface, {"health": healthy})
        healthy_pixels = pygame.image.tostring(surface, "RGB")

        warning = dict(
            base_health,
            status="warning",
            alerts=["gradient_clipping_frequent"],
        )
        ui.render(surface, {"health": warning})
        warning_pixels = pygame.image.tostring(surface, "RGB")

        self.assertNotEqual(healthy_pixels, warning_pixels)

    def test_declared_minimum_surface_keeps_metrics_bottom_visible(self):
        ui = PacmanObservatory(initial_tab="METRICS")
        surface = pygame.Surface((ui.MIN_WIDTH, ui.MIN_HEIGHT))
        ui.render(surface, {"health": {"status": "warming_up", "alerts": []}})

        self.assertNotEqual(
            surface.get_at((20, ui.MIN_HEIGHT - 20)),
            pygame.Color(*ui.theme.background),
        )


if __name__ == "__main__":
    unittest.main()
