"""Reward-shaping tests for useful, collision-aware driving fitness."""

import math
import unittest

from drivingGameRL.src.environment import DrivingAction, DrivingEnv
from drivingGameRL.src.math2d import Vec2


def place_on_centerline(
    env: DrivingEnv, progress: float, *, longitudinal_speed: float = 0.0
) -> None:
    point, tangent = env.circuit.point_tangent_at(progress)
    env.vehicle.state.position = point
    env.vehicle.state.heading = math.atan2(tangent.y, tangent.x)
    env.vehicle.state.velocity = tangent * longitudinal_speed


class DrivingRewardShapingTests(unittest.TestCase):
    def test_stationary_policy_cannot_farm_survival_reward_and_ends_early(self):
        env = DrivingEnv("harbor_loop", seed=5, max_steps=2_000)
        rewards = []

        for _ in range(env.STAGNATION_LIMIT_STEPS):
            result = env.step(DrivingAction.COAST)
            rewards.append(result.reward)

        self.assertTrue(result.truncated)
        self.assertFalse(result.terminated)
        self.assertTrue(result.info["stagnated"])
        self.assertEqual(result.info["truncation_reason"], "stagnation")
        self.assertEqual(result.info["stagnation_steps"], env.STAGNATION_LIMIT_STEPS)
        self.assertLess(sum(rewards), 0.0)
        self.assertFalse(any(reward > 0.0 for reward in rewards))

    def test_meaningful_forward_progress_resets_stagnation(self):
        env = DrivingEnv("harbor_loop", seed=7, max_steps=2_000)
        for _ in range(env.STAGNATION_GRACE_STEPS + 12):
            env.step(DrivingAction.COAST)
        self.assertGreater(env.telemetry()["stagnation_steps"], 0)

        place_on_centerline(env, 0.03, longitudinal_speed=60.0)
        result = env.step(DrivingAction.COAST)

        self.assertGreater(result.info["reward_terms"]["progress"], 0.0)
        self.assertEqual(result.info["stagnation_steps"], 0)
        self.assertFalse(result.truncated)

    def test_aligned_forward_motion_beats_reverse_motion(self):
        forward = DrivingEnv("pine_sprint", seed=11)
        reverse = DrivingEnv("pine_sprint", seed=11)
        place_on_centerline(forward, 0.12, longitudinal_speed=70.0)
        place_on_centerline(reverse, 0.12, longitudinal_speed=-70.0)
        forward.previous_progress = 0.12
        reverse.previous_progress = 0.12

        forward_result = forward.step(DrivingAction.COAST)
        reverse_result = reverse.step(DrivingAction.COAST)

        self.assertGreater(forward_result.info["reward_terms"]["progress"], 0.0)
        self.assertGreater(forward_result.info["reward_terms"]["pace"], 0.0)
        self.assertLess(reverse_result.info["reward_terms"]["progress"], 0.0)
        self.assertLess(reverse_result.info["reward_terms"]["reverse"], 0.0)
        self.assertGreater(forward_result.reward, reverse_result.reward)

    def test_imminent_forward_barrier_reduces_fitness_at_identical_speed(self):
        clear = DrivingEnv("canyon_maze", seed=12)
        corner = DrivingEnv("canyon_maze", seed=12)
        place_on_centerline(clear, 0.0, longitudinal_speed=70.0)
        place_on_centerline(corner, 0.60, longitudinal_speed=70.0)
        clear.previous_progress = 0.0
        corner.previous_progress = 0.60

        clear_result = clear.step(DrivingAction.COAST)
        corner_result = corner.step(DrivingAction.COAST)

        self.assertGreater(
            clear_result.info["forward_clearance"],
            corner_result.info["forward_clearance"],
        )
        self.assertAlmostEqual(
            clear_result.info["reward_terms"]["progress"],
            corner_result.info["reward_terms"]["progress"],
            places=9,
        )
        self.assertGreater(
            clear_result.info["reward_terms"]["pace"],
            corner_result.info["reward_terms"]["pace"],
        )
        self.assertGreater(clear_result.reward, corner_result.reward)

    def test_forward_backward_oscillation_has_no_positive_progress_net(self):
        env = DrivingEnv("harbor_loop", seed=13)
        progress_rewards = []
        rewards = []
        for progress in (0.01, 0.0) * 8:
            place_on_centerline(env, progress)
            result = env.step(DrivingAction.COAST)
            progress_rewards.append(result.info["reward_terms"]["progress"])
            rewards.append(result.reward)

        self.assertAlmostEqual(sum(progress_rewards), 0.0, places=9)
        self.assertLessEqual(sum(rewards), 0.0)

    def test_collision_has_entry_and_continuous_contact_costs(self):
        env = DrivingEnv("harbor_loop", seed=17)
        point, tangent = env.circuit.point_tangent_at(0.2)
        outward = tangent.perpendicular()
        env.vehicle.state.position = point + outward * (
            env.circuit.collision_radius + 3.0
        )
        env.vehicle.state.velocity = outward * 100.0

        result = env.step(DrivingAction.ACCELERATE)

        terms = result.info["reward_terms"]
        self.assertTrue(result.info["collision_started"])
        self.assertLessEqual(terms["collision"], -2.0)
        self.assertEqual(terms["barrier_contact"], -0.10)
        self.assertLess(result.reward, 0.0)


if __name__ == "__main__":
    unittest.main()
