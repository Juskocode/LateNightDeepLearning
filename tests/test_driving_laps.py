import math
import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from drivingGameRL.src.environment import (
    DrivingAction,
    DrivingEnv,
    LapPose,
    LapRecord,
)
from drivingGameRL.src.game import DrivingGame
from drivingGameRL.src.math2d import Vec2
from drivingGameRL.src.rendering import format_lap_time


def drive_to_progress(env: DrivingEnv, progress: float):
    point, tangent = env.circuit.point_tangent_at(progress)
    env.vehicle.state.position = point
    env.vehicle.state.velocity = Vec2()
    env.vehicle.state.heading = math.atan2(tangent.y, tangent.x)
    return env.step(DrivingAction.COAST)


def complete_lap(env: DrivingEnv, *, dwell_steps: int = 0):
    for _ in range(dwell_steps):
        env.step(DrivingAction.COAST)
    result = None
    for index in range(1, 21):
        result = drive_to_progress(env, (index % 20) / 20.0)
    assert result is not None
    return result


class DrivingLapTests(unittest.TestCase):
    def test_lap_time_formatting_carries_rounded_minutes(self):
        self.assertEqual(format_lap_time(None), "--:--.---")
        self.assertEqual(format_lap_time(3.25), "00:03.250")
        self.assertEqual(format_lap_time(59.9999), "01:00.000")
        self.assertEqual(format_lap_time(61.0049), "01:01.005")

    def test_lap_times_use_fixed_simulation_time_and_strict_best(self):
        env = DrivingEnv("harbor_loop", fixed_dt=1.0 / 60.0)

        slower = complete_lap(env, dwell_steps=8)
        self.assertTrue(slower.info["lap_completed"])
        self.assertEqual(env.laps, 1)
        self.assertAlmostEqual(slower.info["last_lap_time"], 28.0 / 60.0)
        self.assertAlmostEqual(slower.info["best_lap_time"], 28.0 / 60.0)
        self.assertEqual(slower.info["current_lap_time"], 0.0)

        first_record = env.best_lap_record
        faster = complete_lap(env)
        self.assertTrue(faster.info["lap_completed"])
        self.assertAlmostEqual(faster.info["last_lap_time"], 20.0 / 60.0)
        self.assertAlmostEqual(faster.info["best_lap_time"], 20.0 / 60.0)
        self.assertIsNot(env.best_lap_record, first_record)

        tied_record = env.best_lap_record
        complete_lap(env)
        self.assertIs(env.best_lap_record, tied_record)
        snapshot = env.telemetry()
        self.assertEqual(snapshot["laps"], 3)
        self.assertEqual(snapshot["last_lap_time"], snapshot["best_lap_time"])

    def test_ordered_gates_reject_shortcuts_and_reverse_crossings(self):
        env = DrivingEnv("harbor_loop")
        for progress in (0.02, 0.51, 0.74, 0.99, 0.01):
            result = drive_to_progress(env, progress)
        self.assertFalse(result.info["lap_completed"])
        self.assertEqual(result.info["reward_terms"]["lap"], 0.0)
        self.assertEqual(env.laps, 0)

        env.reset()
        drive_to_progress(env, 0.01)
        reverse = drive_to_progress(env, 0.99)
        self.assertFalse(reverse.info["lap_completed"])
        self.assertEqual(env.laps, 0)

        env.reset()
        for index in range(1, 16):
            drive_to_progress(env, index / 20.0)
        for index in range(14, -1, -1):
            drive_to_progress(env, index / 20.0)
        drive_to_progress(env, 0.95)
        reversed_shortcut = drive_to_progress(env, 0.0)
        self.assertFalse(reversed_shortcut.info["lap_completed"])
        self.assertEqual(env.laps, 0)

        env.reset()
        valid = complete_lap(env)
        self.assertTrue(valid.info["lap_completed"])
        self.assertEqual(env.laps, 1)

    def test_best_records_survive_reset_and_are_isolated_per_circuit(self):
        env = DrivingEnv("harbor_loop")
        complete_lap(env)
        harbor_record = env.best_lap_record

        env.reset()
        self.assertIs(env.best_lap_record, harbor_record)
        self.assertEqual(env.current_lap_time, 0.0)
        self.assertIsNone(env.last_lap_time)
        self.assertEqual(env.laps, 0)

        env.change_circuit("pine_sprint")
        self.assertIsNone(env.best_lap_record)
        complete_lap(env, dwell_steps=4)
        pine_record = env.best_lap_record
        self.assertIsNotNone(pine_record)
        self.assertEqual(pine_record.circuit, "pine_sprint")

        env.change_circuit("harbor_loop")
        self.assertIs(env.best_lap_record, harbor_record)
        env.change_circuit("pine_sprint")
        self.assertIs(env.best_lap_record, pine_record)

    def test_ghost_interpolates_wrapped_heading_and_hides_after_finish(self):
        env = DrivingEnv("harbor_loop")
        record = LapRecord(
            circuit="harbor_loop",
            duration=1.0,
            trajectory=(
                LapPose(0.0, Vec2(0.0, 0.0), math.radians(179.0)),
                LapPose(1.0, Vec2(10.0, 4.0), math.radians(-179.0)),
            ),
        )
        env._best_laps[env.circuit.slug] = record

        midpoint = env.ghost_pose_at(0.5)
        self.assertIsNotNone(midpoint)
        self.assertAlmostEqual(midpoint.position.x, 5.0)
        self.assertAlmostEqual(midpoint.position.y, 2.0)
        self.assertAlmostEqual(abs(math.degrees(midpoint.heading)), 180.0)
        self.assertIsNotNone(env.ghost_pose_at(1.0))
        self.assertIsNone(env.ghost_pose_at(1.0 + 1e-12))
        with self.assertRaises(ValueError):
            env.ghost_pose_at(-0.1)

    def test_trajectory_sampling_is_approximately_30hz_and_bounded(self):
        env = DrivingEnv("harbor_loop")
        for _ in range(60):
            env.step(DrivingAction.COAST)
        self.assertGreaterEqual(env.current_trajectory_samples, 30)
        self.assertLessEqual(env.current_trajectory_samples, 32)

        env.MAX_GHOST_SAMPLES = 16
        for _ in range(180):
            env.step(DrivingAction.COAST)
        snapshot = env.telemetry()
        self.assertLessEqual(snapshot["ghost_recording_samples"], 16)

    def test_upgrade_invalidates_partial_candidate_but_preserves_best(self):
        game = DrivingGame("harbor_loop", render=False)
        try:
            game.cycle_upgrade("wheels")
            self.assertTrue(game.env.telemetry()["lap_candidate_valid"])
            complete_lap(game.env)
            best = game.env.best_lap_record
            for progress in (0.05, 0.10, 0.15, 0.20):
                drive_to_progress(game.env, progress)
            self.assertGreater(game.env.current_lap_time, 0.0)

            game.cycle_upgrade("motor")
            self.assertEqual(game.env.current_lap_time, 0.0)
            self.assertFalse(game.env.telemetry()["lap_candidate_valid"])
            self.assertIs(game.env.best_lap_record, best)
            self.assertIsNone(game.env.ghost_pose_at())
        finally:
            game.close()

    def test_g_toggles_presentation_without_mutating_physics(self):
        game = DrivingGame("harbor_loop", render=False)
        try:
            complete_lap(game.env)
            before = game.env.telemetry()
            game.draw()
            after = game.env.telemetry()
            self.assertEqual(after["position"], before["position"])
            self.assertEqual(after["steps"], before["steps"])
            self.assertEqual(after["best_lap_time"], before["best_lap_time"])
            self.assertTrue(game.telemetry()["ghost_enabled"])

            pygame.event.clear()
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g))
            game.handle_events()
            self.assertFalse(game.telemetry()["ghost_enabled"])
            game.draw()
        finally:
            game.close()


if __name__ == "__main__":
    unittest.main()
