"""Reward-shaping tests for useful, collision-aware driving fitness."""

import math
import unittest

from drivingGameRL.src.environment import DrivingAction, DrivingEnv, SensorRay
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

    def test_increasing_clearance_beats_closing_for_equivalent_motion(self):
        escaping = DrivingEnv("harbor_loop", seed=23)
        closing = DrivingEnv("harbor_loop", seed=23)
        for env in (escaping, closing):
            place_on_centerline(env, 0.04, longitudinal_speed=65.0)
            env.previous_progress = 0.04

        baseline, _ = escaping._clearance_snapshot(escaping.sensor_rays())
        margin = min(0.08, baseline * 0.4, (1.0 - baseline) * 0.4)
        self.assertGreater(margin, 0.01)
        escaping._usable_clearance = baseline - margin
        closing._usable_clearance = baseline + margin

        escaping_result = escaping.step(DrivingAction.COAST)
        closing_result = closing.step(DrivingAction.COAST)

        self.assertGreater(escaping_result.info["clearance_delta"], 0.0)
        self.assertLess(closing_result.info["clearance_delta"], 0.0)
        self.assertGreater(
            escaping_result.info["reward_terms"]["clearance_gain"], 0.0
        )
        self.assertEqual(
            escaping_result.info["reward_terms"]["wall_closing"], 0.0
        )
        self.assertLess(
            closing_result.info["reward_terms"]["wall_closing"], 0.0
        )
        self.assertEqual(
            closing_result.info["reward_terms"]["clearance_gain"], 0.0
        )
        self.assertAlmostEqual(
            escaping_result.info["reward_terms"]["progress"],
            closing_result.info["reward_terms"]["progress"],
            places=12,
        )
        self.assertGreater(escaping_result.reward, closing_result.reward)

        telemetry = escaping.telemetry()
        for key in (
            "usable_clearance",
            "previous_usable_clearance",
            "clearance_delta",
            "green_ray_fraction",
            "wall_closing",
            "clearance_motion_ratio",
            "wall_contact_steps",
            "recent_collision_entries",
            "collision_looped",
        ):
            self.assertEqual(telemetry[key], escaping_result.info[key])
        self.assertEqual(
            telemetry["clearance_objective"],
            escaping_result.info["clearance_objective"],
        )
        self.assertEqual(
            telemetry["clearance_objective"]["ray_weights"],
            escaping.CLEARANCE_RAY_WEIGHTS,
        )

    def test_stationary_car_cannot_farm_increasing_green_clearance(self):
        env = DrivingEnv("harbor_loop", seed=29)
        current, _ = env._clearance_snapshot(env.sensor_rays())
        env._usable_clearance = max(0.0, current - 0.20)

        result = env.step(DrivingAction.COAST)

        self.assertGreater(result.info["clearance_delta"], 0.0)
        self.assertEqual(result.info["clearance_motion_ratio"], 0.0)
        self.assertEqual(result.info["reward_terms"]["clearance_gain"], 0.0)
        self.assertEqual(result.info["reward_terms"]["green_clearance"], 0.0)
        self.assertEqual(
            result.info["clearance_objective"]["green_density_scale"],
            env.GREEN_DENSITY_REWARD_SCALE,
        )
        self.assertLessEqual(result.reward, 0.0)

    def test_green_sensor_density_rewards_open_sides_at_equivalent_motion(self):
        high_green = DrivingEnv("harbor_loop", seed=30)
        low_green = DrivingEnv("harbor_loop", seed=30)
        for env in (high_green, low_green):
            place_on_centerline(env, 0.04, longitudinal_speed=65.0)
            env.previous_progress = 0.04

        def install_fan(env, readings):
            def sensor_rays(max_distance=env.SENSOR_MAX_DISTANCE):
                origin = env.vehicle.state.position
                return tuple(
                    SensorRay(
                        angle=env.vehicle.state.heading + relative_angle,
                        max_distance=max_distance,
                        distance=reading * max_distance,
                        normalized_distance=reading,
                        origin=origin,
                        endpoint=(
                            origin
                            + Vec2.from_angle(
                                env.vehicle.state.heading + relative_angle
                            )
                            * (reading * max_distance)
                        ),
                        hit=reading < 1.0,
                    )
                    for relative_angle, reading in zip(
                        env.SENSOR_RELATIVE_ANGLES, readings
                    )
                )

            env.sensor_rays = sensor_rays
            env._usable_clearance, env._green_ray_fraction = (
                env._clearance_snapshot(env.sensor_rays())
            )
            env._previous_usable_clearance = env._usable_clearance

        install_fan(high_green, (0.8,) * 9)
        install_fan(
            low_green,
            (0.3, 0.3, 0.3, 0.8, 0.8, 0.8, 0.3, 0.3, 0.3),
        )

        high_result = high_green.step(DrivingAction.COAST)
        low_result = low_green.step(DrivingAction.COAST)

        self.assertEqual(high_result.info["forward_clearance"], 0.8)
        self.assertEqual(low_result.info["forward_clearance"], 0.8)
        self.assertEqual(high_result.info["green_ray_fraction"], 1.0)
        self.assertEqual(low_result.info["green_ray_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(
            high_result.info["reward_terms"]["progress"],
            low_result.info["reward_terms"]["progress"],
            places=12,
        )
        self.assertAlmostEqual(
            high_result.info["reward_terms"]["pace"],
            low_result.info["reward_terms"]["pace"],
            places=12,
        )
        self.assertEqual(high_result.info["reward_terms"]["clearance_gain"], 0.0)
        self.assertEqual(low_result.info["reward_terms"]["clearance_gain"], 0.0)
        self.assertGreater(
            high_result.info["reward_terms"]["green_clearance"],
            low_result.info["reward_terms"]["green_clearance"],
        )
        self.assertGreater(high_result.reward, low_result.reward)
        self.assertLessEqual(
            high_result.info["reward_terms"]["green_clearance"],
            high_green.GREEN_DENSITY_REWARD_SCALE,
        )

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
        self.assertLessEqual(terms["collision"], -env.COLLISION_ENTRY_PENALTY)
        self.assertEqual(terms["barrier_contact"], -env.WALL_CONTACT_PENALTY)
        self.assertLess(result.reward, 0.0)

    def test_persistent_wall_contact_is_decisive_and_truncates_early(self):
        env = DrivingEnv("harbor_loop", seed=31, max_steps=2_000)
        point, tangent = env.circuit.point_tangent_at(0.2)
        outward = tangent.perpendicular()
        rewards = []

        for _ in range(env.WALL_CONTACT_TRUNCATION_STEPS):
            env.vehicle.state.position = point + outward * (
                env.circuit.collision_radius + 2.0
            )
            env.vehicle.state.velocity = outward * 55.0
            result = env.step(DrivingAction.COAST)
            rewards.append(result.reward)

        self.assertTrue(result.truncated)
        self.assertTrue(result.info["collision_looped"])
        self.assertEqual(result.info["truncation_reason"], "collision_loop")
        self.assertEqual(
            result.info["wall_contact_steps"],
            env.WALL_CONTACT_TRUNCATION_STEPS,
        )
        self.assertLess(sum(rewards), -50.0)
        self.assertEqual(
            env.telemetry()["wall_contact_steps"],
            result.info["wall_contact_steps"],
        )

    def test_repeated_collision_entries_within_window_truncate_loop(self):
        env = DrivingEnv("harbor_loop", seed=37, max_steps=2_000)
        point, tangent = env.circuit.point_tangent_at(0.2)
        outward = tangent.perpendicular()

        for entry in range(env.COLLISION_LOOP_ENTRY_LIMIT):
            env.vehicle.state.position = point + outward * (
                env.circuit.collision_radius + 2.0
            )
            env.vehicle.state.velocity = outward * 65.0
            result = env.step(DrivingAction.COAST)
            if entry + 1 < env.COLLISION_LOOP_ENTRY_LIMIT:
                self.assertFalse(result.info["collision_looped"])
                env.vehicle.state.position = point
                env.vehicle.state.velocity = Vec2()
                env.step(DrivingAction.COAST)

        self.assertTrue(result.truncated)
        self.assertEqual(result.info["truncation_reason"], "collision_loop")
        self.assertEqual(
            result.info["recent_collision_entries"],
            env.COLLISION_LOOP_ENTRY_LIMIT,
        )

    def test_collision_entries_spaced_at_window_do_not_accumulate(self):
        env = DrivingEnv("harbor_loop", seed=38, max_steps=3_000)
        point, tangent = env.circuit.point_tangent_at(0.2)
        outward = tangent.perpendicular()
        clean_progress = 0.2

        for entry in range(env.COLLISION_LOOP_ENTRY_LIMIT):
            env.vehicle.state.position = point + outward * (
                env.circuit.collision_radius + 2.0
            )
            env.vehicle.state.velocity = outward * 65.0
            result = env.step(DrivingAction.COAST)
            self.assertTrue(result.info["collision_started"])
            self.assertEqual(result.info["recent_collision_entries"], 1)
            self.assertFalse(result.info["collision_looped"])
            if entry + 1 == env.COLLISION_LOOP_ENTRY_LIMIT:
                break

            for clean_tick in range(env.COLLISION_LOOP_WINDOW_STEPS):
                clean_progress += 0.001
                place_on_centerline(env, clean_progress)
                clean_result = env.step(DrivingAction.COAST)
                if clean_tick == env.COLLISION_LOOP_WINDOW_STEPS - 2:
                    self.assertEqual(
                        clean_result.info["recent_collision_entries"], 1
                    )
            self.assertEqual(clean_result.info["recent_collision_entries"], 0)
            self.assertEqual(
                clean_result.info["steps_since_collision"],
                env.COLLISION_LOOP_WINDOW_STEPS,
            )
            self.assertFalse(clean_result.info["collision_looped"])

    def test_reset_clears_contact_and_clearance_transition_diagnostics(self):
        env = DrivingEnv("harbor_loop", seed=39)
        point, tangent = env.circuit.point_tangent_at(0.2)
        outward = tangent.perpendicular()
        env.vehicle.state.position = point + outward * (
            env.circuit.collision_radius + 2.0
        )
        env.vehicle.state.velocity = outward * 60.0
        env.step(DrivingAction.COAST)
        self.assertGreater(env.telemetry()["wall_contact_steps"], 0)

        env.reset(seed=39)
        telemetry = env.telemetry()

        self.assertFalse(telemetry["wall_contact_active"])
        self.assertEqual(telemetry["wall_contact_steps"], 0)
        self.assertEqual(telemetry["recent_collision_entries"], 0)
        self.assertFalse(telemetry["collision_looped"])
        self.assertIsNone(telemetry["truncation_reason"])
        self.assertEqual(telemetry["clearance_delta"], 0.0)
        self.assertEqual(
            telemetry["usable_clearance"],
            telemetry["previous_usable_clearance"],
        )

    def test_clearance_objective_preserves_ordered_progress_and_lap_reward(self):
        env = DrivingEnv("harbor_loop", seed=41)
        result = None
        for index in range(1, 21):
            place_on_centerline(env, index / 20.0)
            result = env.step(DrivingAction.COAST)

        assert result is not None
        self.assertTrue(result.info["lap_completed"])
        self.assertEqual(result.info["laps"], 1)
        self.assertEqual(result.info["reward_terms"]["lap"], 75.0)
        self.assertGreater(result.info["reward_terms"]["progress"], 0.0)
        self.assertFalse(result.info["collision_looped"])
        self.assertFalse(result.truncated)


if __name__ == "__main__":
    unittest.main()
