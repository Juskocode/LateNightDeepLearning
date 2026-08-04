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

    def test_max_episode_progress_is_monotonic_and_resettable(self):
        env = DrivingEnv("harbor_loop", seed=8, max_steps=2_000)
        for progress in (0.05, 0.10):
            place_on_centerline(env, progress)
            forward = env.step(DrivingAction.COAST)

        peak = forward.info["episode_lap_progress"]
        self.assertGreater(peak, 0.0)
        self.assertEqual(forward.info["max_episode_lap_progress"], peak)

        place_on_centerline(env, 0.07)
        regressed = env.step(DrivingAction.COAST)
        self.assertLess(regressed.info["episode_lap_progress"], peak)
        self.assertEqual(regressed.info["max_episode_lap_progress"], peak)
        self.assertEqual(env.telemetry()["max_episode_lap_progress"], peak)

        env.reset(seed=8)
        telemetry = env.telemetry()
        self.assertEqual(telemetry["episode_lap_progress"], 0.0)
        self.assertEqual(telemetry["max_episode_lap_progress"], 0.0)

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
        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)
        self.assertTrue(result.info["collision_recovery_active"])
        self.assertGreater(result.info["collision_pressure"], 0.0)

    def test_slow_inward_escape_does_not_accrue_stale_contact_pressure(self):
        env = DrivingEnv("harbor_loop", seed=19, max_steps=2_000)
        point, tangent = env.circuit.point_tangent_at(0.2)
        outward = tangent.perpendicular()
        radius = env.circuit.collision_radius
        env.vehicle.state.heading = math.atan2(tangent.y, tangent.x)
        env.vehicle.state.position = point + outward * (radius + 1.0)
        env.vehicle.state.velocity = outward * 20.0

        impact = env.step(DrivingAction.COAST)
        self.assertTrue(impact.info["collided"])
        initial_pressure = impact.info["collision_pressure"]

        # This is the formerly fatal case: collision-entry hysteresis remains
        # latched while a low-speed car is physically clear and moving inward.
        env.vehicle.state.position = point + outward * (radius - 0.1)
        env.vehicle.state.velocity = outward * -3.0
        for _ in range(env.COLLISION_RECOVERY_TIMEOUT_STEPS):
            recovering = env.step(DrivingAction.COAST)
            self.assertFalse(recovering.info["collided"])
            self.assertFalse(recovering.info["wall_contact_active"])
            self.assertEqual(recovering.info["reward_terms"]["barrier_contact"], 0.0)
            self.assertFalse(recovering.truncated)

        self.assertEqual(recovering.info["wall_contact_steps"], 1)
        self.assertEqual(recovering.info["collision_pressure"], initial_pressure)
        self.assertTrue(recovering.info["collision_recovery_active"])

    def test_clean_forward_recovery_resets_collision_pressure(self):
        env = DrivingEnv("harbor_loop", seed=18, max_steps=2_000)
        progress = 0.20

        # Three quick glancing entries create pressure without trapping the car.
        # A single clear tick between them is intentionally too short to count as
        # a stable recovery.
        for _ in range(3):
            point, tangent = env.circuit.point_tangent_at(progress)
            outward = tangent.perpendicular()
            env.vehicle.state.position = point + outward * (
                env.circuit.collision_radius + 2.0
            )
            env.vehicle.state.heading = math.atan2(tangent.y, tangent.x)
            env.vehicle.state.velocity = outward * 45.0 + tangent * 35.0
            impact = env.step(DrivingAction.COAST)
            self.assertTrue(impact.info["collision_started"])
            self.assertFalse(impact.truncated)

            progress += 0.001
            place_on_centerline(env, progress, longitudinal_speed=45.0)
            clear_tick = env.step(DrivingAction.COAST)
            self.assertFalse(clear_tick.info["wall_contact_active"])
            self.assertFalse(clear_tick.truncated)

        self.assertTrue(env.telemetry()["collision_recovery_active"])
        self.assertGreater(env.telemetry()["collision_pressure"], 0.0)

        # Sustained, meaningful forward movement proves that this car recovered.
        # Recovery must clear accumulated kill pressure rather than leaving an
        # otherwise healthy car one glancing impact away from a reset.
        confirmation_steps = env.COLLISION_RECOVERY_CONFIRM_STEPS
        for _ in range(confirmation_steps):
            progress += 0.002
            place_on_centerline(env, progress, longitudinal_speed=55.0)
            recovered = env.step(DrivingAction.COAST)
            self.assertFalse(recovered.truncated)

        telemetry = env.telemetry()
        self.assertFalse(telemetry["collision_recovery_active"])
        self.assertEqual(telemetry["collision_pressure"], 0.0)
        self.assertGreaterEqual(telemetry["collision_recoveries"], 1)
        self.assertEqual(telemetry["wall_contact_steps"], 0)

        # A later isolated impact begins a fresh recovery window; it is not the
        # fourth strike of the already recovered incident.
        point, tangent = env.circuit.point_tangent_at(progress)
        outward = tangent.perpendicular()
        env.vehicle.state.position = point + outward * (
            env.circuit.collision_radius + 2.0
        )
        env.vehicle.state.heading = math.atan2(tangent.y, tangent.x)
        env.vehicle.state.velocity = outward * 45.0 + tangent * 35.0
        next_impact = env.step(DrivingAction.COAST)
        self.assertTrue(next_impact.info["collision_started"])
        self.assertFalse(next_impact.terminated)
        self.assertFalse(next_impact.truncated)
        self.assertEqual(next_impact.info["recent_collision_entries"], 1)

    def test_persistent_wall_contact_is_decisive_and_truncates_early(self):
        env = DrivingEnv("harbor_loop", seed=31, max_steps=2_000)
        point, tangent = env.circuit.point_tangent_at(0.2)
        outward = tangent.perpendicular()
        rewards = []

        for _ in range(env.COLLISION_RECOVERY_TIMEOUT_STEPS):
            env.vehicle.state.position = point + outward * (
                env.circuit.collision_radius + 2.0
            )
            env.vehicle.state.velocity = outward * 55.0
            result = env.step(DrivingAction.COAST)
            rewards.append(result.reward)

        self.assertTrue(result.truncated)
        self.assertTrue(result.info["collision_looped"])
        self.assertEqual(result.info["truncation_reason"], "collision_loop")
        self.assertEqual(result.info["collision_recoveries"], 0)
        self.assertEqual(result.info["collision_pressure"], 1.0)
        self.assertGreaterEqual(
            result.info["collision_recovery_steps"],
            env.COLLISION_RECOVERY_TIMEOUT_STEPS,
        )
        self.assertEqual(
            result.info["wall_contact_steps"],
            env.COLLISION_RECOVERY_TIMEOUT_STEPS,
        )
        self.assertLess(sum(rewards), -50.0)
        self.assertEqual(
            env.telemetry()["wall_contact_steps"],
            result.info["wall_contact_steps"],
        )

    def test_repeated_collision_entries_without_forward_recovery_truncate_loop(self):
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

    def test_collision_entries_do_not_accumulate_after_clean_forward_recovery(self):
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

            for _ in range(env.COLLISION_RECOVERY_CONFIRM_STEPS):
                clean_progress += 0.002
                place_on_centerline(
                    env,
                    clean_progress,
                    longitudinal_speed=50.0,
                )
                clean_result = env.step(DrivingAction.COAST)
                self.assertFalse(clean_result.truncated)
            self.assertEqual(clean_result.info["recent_collision_entries"], 0)
            self.assertFalse(clean_result.info["collision_looped"])
            self.assertFalse(clean_result.info["collision_recovery_active"])
            self.assertEqual(clean_result.info["collision_pressure"], 0.0)

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
        self.assertFalse(telemetry["collision_recovery_active"])
        self.assertEqual(telemetry["collision_recovery_steps"], 0)
        self.assertEqual(telemetry["collision_recovery_clean_steps"], 0)
        self.assertEqual(telemetry["collision_recoveries"], 0)
        self.assertEqual(telemetry["collision_pressure"], 0.0)
        self.assertIsNone(telemetry["truncation_reason"])
        self.assertEqual(telemetry["clearance_delta"], 0.0)
        self.assertEqual(
            telemetry["usable_clearance"],
            telemetry["previous_usable_clearance"],
        )

    def test_recovered_contact_preserves_the_ordered_lap_candidate(self):
        env = DrivingEnv("harbor_loop", seed=40, max_steps=2_000)

        # Establish ordinary ordered progress before one glancing impact.
        for progress in (0.05, 0.10):
            place_on_centerline(env, progress, longitudinal_speed=50.0)
            result = env.step(DrivingAction.COAST)
            self.assertTrue(result.info["lap_candidate_valid"])

        point, tangent = env.circuit.point_tangent_at(0.10)
        outward = tangent.perpendicular()
        env.vehicle.state.position = point + outward * (
            env.circuit.collision_radius + 3.0
        )
        env.vehicle.state.heading = math.atan2(tangent.y, tangent.x)
        env.vehicle.state.velocity = outward * 80.0 + tangent * 45.0
        impact = env.step(DrivingAction.COAST)
        self.assertTrue(impact.info["collision_started"])
        self.assertTrue(impact.info["lap_candidate_valid"])
        self.assertFalse(impact.truncated)

        progress = 0.105
        for _ in range(env.COLLISION_RECOVERY_CONFIRM_STEPS):
            progress += 0.002
            place_on_centerline(env, progress, longitudinal_speed=50.0)
            recovered = env.step(DrivingAction.COAST)
        self.assertFalse(recovered.info["collision_recovery_active"])
        self.assertTrue(recovered.info["lap_candidate_valid"])

        # Continue through all ordered gates and the episode origin. Collision
        # recovery must not silently throw away an otherwise valid near-lap.
        while progress < 0.95:
            progress = min(0.95, progress + 0.04)
            place_on_centerline(env, progress, longitudinal_speed=50.0)
            result = env.step(DrivingAction.COAST)
            self.assertFalse(result.truncated)
        place_on_centerline(env, 0.999, longitudinal_speed=50.0)
        env.step(DrivingAction.COAST)
        place_on_centerline(env, 0.001, longitudinal_speed=50.0)
        completed = env.step(DrivingAction.COAST)

        self.assertTrue(completed.info["lap_completed"])
        self.assertFalse(completed.truncated)
        self.assertEqual(completed.info["laps"], 1)
        self.assertGreaterEqual(completed.info["collision_recoveries"], 1)

    def test_clearance_objective_preserves_ordered_progress_and_lap_reward(self):
        env = DrivingEnv("harbor_loop", seed=41)
        result = None
        for index in range(1, 21):
            place_on_centerline(env, index / 20.0)
            result = env.step(DrivingAction.COAST)

        assert result is not None
        self.assertTrue(result.info["lap_completed"])
        self.assertEqual(result.info["laps"], 1)
        self.assertEqual(result.info["max_episode_lap_progress"], 1.0)
        self.assertEqual(
            result.info["reward_terms"]["lap"], env.LAP_COMPLETION_REWARD
        )
        self.assertGreater(result.info["reward_terms"]["progress"], 0.0)
        self.assertFalse(result.info["collision_looped"])
        self.assertFalse(result.truncated)

    def test_ordered_checkpoint_reward_is_one_time_and_non_farmable(self):
        env = DrivingEnv("harbor_loop", seed=42)
        for progress in (0.04, 0.08, 0.12, 0.16, 0.20, 0.24):
            place_on_centerline(env, progress, longitudinal_speed=45.0)
            before_gate = env.step(DrivingAction.COAST)
            self.assertEqual(before_gate.info["reward_terms"]["checkpoint"], 0.0)

        place_on_centerline(env, 0.26, longitudinal_speed=45.0)
        milestone = env.step(DrivingAction.COAST)
        self.assertTrue(milestone.info["checkpoint_advanced"])
        self.assertEqual(
            milestone.info["reward_terms"]["checkpoint"],
            env.ORDERED_CHECKPOINT_REWARD,
        )

        for progress in (0.24, 0.26) * 4:
            place_on_centerline(env, progress, longitudinal_speed=45.0)
            oscillation = env.step(DrivingAction.COAST)
            self.assertFalse(oscillation.info["checkpoint_advanced"])
            self.assertEqual(oscillation.info["reward_terms"]["checkpoint"], 0.0)


if __name__ == "__main__":
    unittest.main()
