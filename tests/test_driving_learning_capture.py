"""Smoke and isolation tests for real Driving Lab documentation capture."""

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from PIL import Image, ImageChops
import pygame
import torch

from drivingGameRL.src.learning_capture import capture_learning_gif
from drivingGameRL.src.learning_runtime import (
    DrivingLearningSession,
    LearningRuntimeConfig,
)
from drivingGameRL.src.learning_visualization import LEARNING_WINDOW_SIZE
from drivingGameRL.src.ml import DQNConfig


def tiny_session(seed: int = 37) -> DrivingLearningSession:
    return DrivingLearningSession(
        LearningRuntimeConfig(
            algorithm="double_dqn",
            evaluation_steps=40,
            population_size=2,
            elite_count=1,
            seed=seed,
        ),
        dqn_config=DQNConfig(
            algorithm="double_dqn",
            hidden_sizes=(8,),
            replay_capacity=64,
            batch_size=2,
            warmup_steps=0,
            target_sync_interval=3,
            epsilon_decay_steps=40,
            seed=seed,
        ),
    )


def tiny_population_session(seed: int = 43) -> DrivingLearningSession:
    return DrivingLearningSession(
        LearningRuntimeConfig(
            algorithm="genetic_dqn",
            evaluation_steps=20,
            population_size=3,
            elite_count=1,
            tournament_size=2,
            seed=seed,
        ),
        dqn_config=DQNConfig(
            algorithm="double_dqn",
            hidden_sizes=(8,),
            replay_capacity=64,
            batch_size=2,
            warmup_steps=0,
            seed=seed,
        ),
    )


class DrivingLearningCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.font.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_capture_is_full_size_animated_and_race_is_isolated(self):
        frames_per_tab = 2
        training_steps_per_frame = 1
        expected_training_steps = 3 * frames_per_tab * training_steps_per_frame
        session = tiny_session()
        reference = tiny_session()
        for _ in range(expected_training_steps):
            reference.step()

        with tempfile.TemporaryDirectory() as directory:
            output = capture_learning_gif(
                Path(directory) / "driving-learning.gif",
                session,
                frames_per_tab=frames_per_tab,
                training_steps_per_frame=training_steps_per_frame,
                race_frames=3,
                duration_ms=40,
                palette_colors=48,
            )

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1_000)
            with Image.open(output) as image:
                self.assertEqual(image.format, "GIF")
                self.assertEqual(image.size, LEARNING_WINDOW_SIZE)
                self.assertTrue(image.is_animated)
                # Optimization can coalesce visually identical frames, but the
                # three tab layouts and moving race remain distinct.
                self.assertGreaterEqual(image.n_frames, 4)

        # A reference advanced only by the documented training-frame budget must
        # exactly match. The additional race frames therefore touched neither
        # the trainer's environment/replay nor its learned parameters.
        self.assertEqual(session.env.steps, expected_training_steps)
        self.assertEqual(session.env.telemetry(), reference.env.telemetry())
        self.assertEqual(session.observation, reference.observation)
        self.assertEqual(len(session.agent.replay), expected_training_steps)
        self.assertEqual(len(session.agent.replay), len(reference.agent.replay))
        self.assertEqual(session.agent.gradient_steps, reference.agent.gradient_steps)
        for actual, expected in zip(
            session.agent.network.parameters(), reference.agent.network.parameters()
        ):
            self.assertTrue(torch.equal(actual, expected))

    def test_capture_rejects_unbounded_or_non_gif_requests(self):
        session = tiny_session(seed=41)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                capture_learning_gif(
                    Path(directory) / "capture.gif",
                    session,
                    frames_per_tab=100,
                )
            with self.assertRaises(ValueError):
                capture_learning_gif(Path(directory) / "capture.png", session)
            with self.assertRaises(ValueError):
                capture_learning_gif(
                    Path(directory) / "capture.gif",
                    session,
                    population_car_limit=0,
                )

    def test_capture_flags_render_population_cars_and_real_rays(self):
        hidden_session = tiny_population_session()
        visible_session = tiny_population_session()
        with tempfile.TemporaryDirectory() as directory:
            hidden_path = capture_learning_gif(
                Path(directory) / "hidden.gif",
                hidden_session,
                frames_per_tab=1,
                training_steps_per_frame=3,
                race_frames=1,
                palette_colors=64,
                show_sensor_rays=False,
                show_population_cars=False,
            )
            visible_path = capture_learning_gif(
                Path(directory) / "visible.gif",
                visible_session,
                frames_per_tab=1,
                training_steps_per_frame=3,
                race_frames=1,
                palette_colors=64,
                show_sensor_rays=True,
                show_population_cars=True,
            )
            with Image.open(hidden_path) as hidden, Image.open(visible_path) as visible:
                hidden.seek(0)
                visible.seek(0)
                difference = ImageChops.difference(
                    hidden.convert("RGB"), visible.convert("RGB")
                )
                self.assertIsNotNone(difference.getbbox())

        self.assertEqual(hidden_session.env.telemetry(), visible_session.env.telemetry())
        for hidden, visible in zip(
            hidden_session.agent.network.parameters(),
            visible_session.agent.network.parameters(),
        ):
            self.assertTrue(torch.equal(hidden, visible))


if __name__ == "__main__":
    unittest.main()
