"""Regression coverage for progressive multi-lap learning evaluations."""

from __future__ import annotations

import math
import unittest

from drivingGameRL.src.environment import DrivingAction, DrivingEnv, StepResult


def drive_to_relative_progress(
    env: DrivingEnv,
    relative_progress: float,
) -> StepResult:
    """Advance one test step at a coordinate relative to the episode origin."""

    absolute_progress = (env.lap_origin_progress + relative_progress) % 1.0
    point, tangent = env.circuit.point_tangent_at(absolute_progress)
    env.vehicle.state.position = point
    env.vehicle.state.velocity = tangent * 1.0
    env.vehicle.state.heading = math.atan2(tangent.y, tangent.x)
    return env.step(DrivingAction.COAST)


def complete_lap_in_steps(env: DrivingEnv, steps: int) -> StepResult:
    """Complete one ordered lap in an exact deterministic tick count."""

    if steps < 20:
        raise ValueError("test laps require at least 20 ordered steps")
    result: StepResult | None = None
    for index in range(1, steps + 1):
        result = drive_to_relative_progress(env, index / steps)
    assert result is not None
    return result


class DrivingMultiLapTests(unittest.TestCase):
    def test_three_lap_target_terminates_only_after_the_third_valid_lap(self):
        env = DrivingEnv(
            "harbor_loop",
            seed=101,
            max_steps=5_000,
            random_start_curriculum=True,
            lap_target=3,
        )

        first = complete_lap_in_steps(env, 20)
        self.assertTrue(first.info["lap_completed"])
        self.assertTrue(first.info["curriculum_lap_completed"])
        self.assertFalse(first.info["lap_target_completed"])
        self.assertFalse(first.terminated)
        self.assertEqual(first.info["laps"], 1)
        self.assertEqual(first.info["laps_remaining"], 2)

        second = complete_lap_in_steps(env, 20)
        self.assertTrue(second.info["lap_completed"])
        self.assertFalse(second.info["lap_target_completed"])
        self.assertFalse(second.terminated)
        self.assertEqual(second.info["laps"], 2)
        self.assertEqual(second.info["laps_remaining"], 1)

        third = complete_lap_in_steps(env, 20)
        self.assertTrue(third.info["lap_completed"])
        self.assertTrue(third.info["lap_target_completed"])
        self.assertTrue(third.terminated)
        self.assertFalse(third.truncated)
        self.assertEqual(third.info["laps"], 3)
        self.assertEqual(third.info["laps_remaining"], 0)

    def test_target_progress_aggregates_completed_and_current_laps(self):
        env = DrivingEnv(
            "harbor_loop",
            seed=103,
            max_steps=5_000,
            random_start_curriculum=True,
            lap_target=3,
        )

        first = complete_lap_in_steps(env, 20)
        self.assertAlmostEqual(first.info["episode_target_progress"], 1.0 / 3.0)
        self.assertAlmostEqual(
            first.info["max_episode_target_progress"],
            1.0 / 3.0,
        )

        halfway = None
        for index in range(1, 11):
            halfway = drive_to_relative_progress(env, index / 20.0)
        assert halfway is not None
        self.assertAlmostEqual(halfway.info["episode_lap_progress"], 0.5)
        self.assertAlmostEqual(halfway.info["episode_target_progress"], 0.5)
        self.assertAlmostEqual(
            halfway.info["max_episode_target_progress"],
            0.5,
        )
        self.assertEqual(halfway.info["laps"], 1)
        self.assertFalse(halfway.info["lap_target_completed"])
        self.assertFalse(halfway.terminated)

    def test_faster_plausible_lap_earns_larger_bounded_time_reward(self):
        fast_env = DrivingEnv(
            "harbor_loop",
            seed=107,
            max_steps=2_000,
            random_start_curriculum=True,
        )
        slow_env = DrivingEnv(
            "harbor_loop",
            seed=107,
            max_steps=2_000,
            random_start_curriculum=True,
        )

        fast = complete_lap_in_steps(fast_env, 700)
        slow = complete_lap_in_steps(slow_env, 1_100)
        fast_bonus = float(fast.info["reward_terms"]["lap_time"])
        slow_bonus = float(slow.info["reward_terms"]["lap_time"])

        self.assertTrue(fast.info["lap_time_bonus_valid"])
        self.assertTrue(slow.info["lap_time_bonus_valid"])
        self.assertGreater(fast_bonus, slow_bonus)
        self.assertGreater(slow_bonus, 0.0)
        self.assertLessEqual(fast_bonus, fast_env.LAP_TIME_BONUS_MAX)
        self.assertAlmostEqual(fast.info["lap_time_bonus"], fast_bonus)
        self.assertAlmostEqual(slow.info["lap_time_bonus"], slow_bonus)

    def test_physically_impossible_lap_receives_no_time_bonus(self):
        env = DrivingEnv(
            "harbor_loop",
            seed=109,
            max_steps=500,
            random_start_curriculum=True,
        )

        completed = complete_lap_in_steps(env, 20)

        self.assertTrue(completed.info["lap_completed"])
        self.assertFalse(completed.info["lap_time_bonus_valid"])
        self.assertEqual(completed.info["lap_time_bonus"], 0.0)
        self.assertEqual(completed.info["reward_terms"]["lap_time"], 0.0)
        self.assertLess(
            completed.info["last_lap_time"],
            completed.info["lap_time_reference"]
            * env.LAP_TIME_MINIMUM_RATIO,
        )


if __name__ == "__main__":
    unittest.main()
