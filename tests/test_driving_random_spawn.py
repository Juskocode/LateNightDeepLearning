import math
import unittest

from drivingGameRL.src.environment import DrivingAction, DrivingEnv
from drivingGameRL.src.math2d import Vec2, wrap_angle


def drive_relative_to_origin(env: DrivingEnv, offset: float):
    progress = (env.lap_origin_progress + offset) % 1.0
    point, tangent = env.circuit.point_tangent_at(progress)
    env.vehicle.state.position = point
    env.vehicle.state.velocity = Vec2()
    env.vehicle.state.heading = math.atan2(tangent.y, tangent.x)
    return env.step(DrivingAction.COAST)


def complete_origin_relative_lap(env: DrivingEnv):
    result = None
    for index in range(1, 21):
        result = drive_relative_to_origin(env, index / 20.0)
    assert result is not None
    return result


class DrivingRandomSpawnTests(unittest.TestCase):
    def test_default_environment_keeps_standard_nonterminal_laps(self):
        env = DrivingEnv("harbor_loop", seed=19)
        start, heading = env.circuit.start_pose()

        self.assertEqual(env.spawn_mode, "start_line")
        self.assertEqual(env.vehicle.state.position, start)
        self.assertAlmostEqual(env.vehicle.state.heading, heading)
        self.assertFalse(env.random_start_curriculum)

        result = complete_origin_relative_lap(env)
        self.assertTrue(result.info["lap_completed"])
        self.assertFalse(result.info["curriculum_lap_completed"])
        self.assertFalse(result.terminated)
        self.assertIsNotNone(env.best_lap_record)

    def test_first_curriculum_spawn_is_seeded_centerline_and_forward(self):
        first = DrivingEnv(
            "pine_sprint", seed=83, random_start_curriculum=True
        )
        second = DrivingEnv(
            "pine_sprint", seed=83, random_start_curriculum=True
        )

        self.assertEqual(first.spawn_mode, "random_track")
        self.assertEqual(first.spawn_progress, second.spawn_progress)
        self.assertEqual(first.vehicle.state.position, second.vehicle.state.position)
        self.assertEqual(first.vehicle.state.heading, second.vehicle.state.heading)
        self.assertNotAlmostEqual(first.spawn_progress, 0.0)

        projection = first.circuit.project(first.vehicle.state.position)
        expected_heading = math.atan2(projection.tangent.y, projection.tangent.x)
        self.assertAlmostEqual(projection.distance, 0.0)
        self.assertAlmostEqual(
            wrap_angle(first.vehicle.state.heading - expected_heading), 0.0
        )
        self.assertEqual(first.vehicle.state.velocity, Vec2())

        first.reset(seed=241)
        second.reset(seed=241)
        self.assertEqual(first.spawn_progress, second.spawn_progress)
        self.assertEqual(first.vehicle.state.position, second.vehicle.state.position)

    def test_full_loop_is_measured_from_random_origin_and_terminates(self):
        env = DrivingEnv(
            "harbor_loop", seed=17, random_start_curriculum=True
        )
        origin = env.spawn_progress

        for checkpoint_step in range(1, 20):
            result = drive_relative_to_origin(env, checkpoint_step / 20.0)
            self.assertFalse(result.terminated)
            self.assertFalse(result.info["curriculum_lap_completed"])

        result = drive_relative_to_origin(env, 1.0)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertTrue(result.info["lap_completed"])
        self.assertTrue(result.info["curriculum_lap_completed"])
        self.assertTrue(result.info["curriculum_unlocked"])
        self.assertEqual(result.info["spawn_mode"], "random_track")
        self.assertAlmostEqual(result.info["spawn_progress"], origin)
        self.assertAlmostEqual(result.info["lap_origin_progress"], origin)
        self.assertAlmostEqual(result.info["episode_lap_progress"], 1.0)
        self.assertEqual(env.laps, 1)
        self.assertTrue(env.curriculum_unlocked)
        self.assertIsNone(
            env.best_lap_record,
            "random-origin timing must not replace the start-line ghost",
        )

        snapshot = env.telemetry()
        self.assertTrue(snapshot["curriculum_lap_completed"])
        self.assertAlmostEqual(snapshot["episode_lap_progress"], 1.0)

    def test_shortcuts_and_reverse_origin_crossings_do_not_unlock(self):
        env = DrivingEnv(
            "desert_switchback", seed=12, random_start_curriculum=True
        )

        result = None
        for offset in (0.02, 0.51, 0.74, 0.99, 0.01):
            result = drive_relative_to_origin(env, offset)
        assert result is not None
        self.assertFalse(result.terminated)
        self.assertFalse(result.info["lap_completed"])
        self.assertFalse(env.curriculum_unlocked)

        env.reset(seed=12)
        drive_relative_to_origin(env, -0.01)
        reverse_crossing = drive_relative_to_origin(env, 0.01)
        self.assertFalse(reverse_crossing.terminated)
        self.assertFalse(reverse_crossing.info["lap_completed"])
        self.assertFalse(env.curriculum_unlocked)

    def test_unlocked_resets_use_seeded_eighty_twenty_mix(self):
        env = DrivingEnv(
            "harbor_loop", seed=33, random_start_curriculum=True
        )
        env.load_curriculum_state({"unlocked": True})

        env.reset(seed=1)  # First random draw is below 0.80.
        self.assertEqual(env.spawn_mode, "start_line")
        self.assertEqual(env.spawn_progress, 0.0)
        self.assertTrue(env.curriculum_unlocked)

        env.reset(seed=0)  # First random draw is above 0.80.
        self.assertEqual(env.spawn_mode, "random_track")
        self.assertNotAlmostEqual(env.spawn_progress, 0.0)
        self.assertTrue(env.curriculum_unlocked)
        self.assertEqual(env.normal_start_probability, 0.80)

    def test_curriculum_state_is_checkpoint_and_clone_safe(self):
        source = DrivingEnv(seed=9, random_start_curriculum=True)
        source.load_curriculum_state({"ready": True})
        state = source.curriculum_state()
        self.assertEqual(state, {"unlocked": True})

        clone = DrivingEnv(seed=9, random_start_curriculum=True)
        clone.load_curriculum_state(state)
        self.assertTrue(clone.curriculum_ready)
        clone.reset(seed=1)
        self.assertEqual(clone.spawn_mode, "start_line")

        clone.load_curriculum_state({})
        self.assertFalse(clone.curriculum_unlocked)
        clone.reset(seed=1)
        self.assertEqual(clone.spawn_mode, "random_track")

        with self.assertRaises(ValueError):
            clone.load_curriculum_state({"unlocked": 1})
        with self.assertRaises(ValueError):
            DrivingEnv(random_start_curriculum=1)

    def test_changing_circuit_requires_a_new_curriculum_unlock(self):
        env = DrivingEnv(seed=5, random_start_curriculum=True)
        env.load_curriculum_state({"unlocked": True})
        env.change_circuit("pine_sprint")

        self.assertFalse(env.curriculum_unlocked)
        self.assertEqual(env.spawn_mode, "random_track")


if __name__ == "__main__":
    unittest.main()
