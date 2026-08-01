import math
import unittest
from dataclasses import FrozenInstanceError

from drivingGameRL.src.environment import DrivingAction, DrivingEnv, SensorRay


class DrivingSensorRayTests(unittest.TestCase):
    def test_five_rays_follow_observation_label_order(self):
        env = DrivingEnv("harbor_loop", seed=17)
        rays = env.sensor_rays()

        self.assertIsInstance(rays, tuple)
        self.assertEqual(len(rays), 5)
        self.assertTrue(all(isinstance(ray, SensorRay) for ray in rays))
        self.assertEqual(
            env.OBSERVATION_LABELS[-5:],
            (
                "ray_left",
                "ray_left_forward",
                "ray_forward",
                "ray_right_forward",
                "ray_right",
            ),
        )
        expected_angles = tuple(
            env.vehicle.state.heading + relative
            for relative in env.SENSOR_RELATIVE_ANGLES
        )
        self.assertEqual(tuple(ray.angle for ray in rays), expected_angles)

    def test_observation_uses_the_exact_public_ray_readings(self):
        env = DrivingEnv("harbor_loop", seed=8)
        actions = (
            DrivingAction.COAST,
            DrivingAction.ACCELERATE,
            DrivingAction.STEER_LEFT,
            DrivingAction.STEER_RIGHT,
        )
        for action in actions:
            observation = env.observation()
            rays = env.sensor_rays()
            self.assertEqual(
                observation[-5:],
                tuple(ray.normalized_distance for ray in rays),
            )
            env.step(action)

    def test_endpoints_match_angle_and_reported_distance(self):
        env = DrivingEnv("pine_sprint", seed=4)
        origin = env.vehicle.state.position

        for ray in env.sensor_rays(123.0):
            displacement = ray.endpoint - ray.origin
            self.assertEqual(ray.origin, origin)
            self.assertAlmostEqual(displacement.length(), ray.distance, places=12)
            self.assertAlmostEqual(
                displacement.x, math.cos(ray.angle) * ray.distance, places=12
            )
            self.assertAlmostEqual(
                displacement.y, math.sin(ray.angle) * ray.distance, places=12
            )
            self.assertAlmostEqual(
                ray.normalized_distance, ray.distance / ray.max_distance, places=12
            )

    def test_hits_and_full_range_rays_are_distinguished(self):
        env = DrivingEnv("harbor_loop")
        rays = env.sensor_rays()
        hit_rays = tuple(ray for ray in rays if ray.hit)
        clear_rays = tuple(ray for ray in rays if not ray.hit)

        self.assertTrue(hit_rays)
        self.assertTrue(clear_rays)
        for ray in hit_rays:
            self.assertLessEqual(ray.distance, ray.max_distance)
            self.assertGreaterEqual(
                env.circuit.project(ray.endpoint).distance,
                env.circuit.collision_radius,
            )
        for ray in clear_rays:
            self.assertEqual(ray.distance, ray.max_distance)
            self.assertEqual(ray.normalized_distance, 1.0)

    def test_invalid_max_distances_are_rejected(self):
        env = DrivingEnv()
        for value in (0.0, -1.0, math.inf, -math.inf, math.nan, True, "150", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    env.sensor_rays(value)  # type: ignore[arg-type]

    def test_snapshots_are_deterministic_and_read_only(self):
        env = DrivingEnv("canyon_maze", seed=31)
        first = env.sensor_rays()
        second = env.sensor_rays()

        self.assertEqual(first, second)
        with self.assertRaises(FrozenInstanceError):
            first[0].distance = 0.0  # type: ignore[misc]
        with self.assertRaises(TypeError):
            first[0] = second[0]  # type: ignore[index]
        self.assertEqual(env.sensor_rays(), second)

    def test_same_pose_reuses_rays_and_pose_changes_invalidate_the_cache(self):
        env = DrivingEnv("harbor_loop", seed=9)

        first = env.sensor_rays()
        observation = env.observation()
        second = env.sensor_rays()

        self.assertIs(second, first)
        self.assertEqual(
            tuple(ray.normalized_distance for ray in second), observation[-5:]
        )

        env.vehicle.state.heading += 0.125
        moved = env.sensor_rays()
        self.assertIsNot(moved, first)
        self.assertTrue(all(ray.origin == env.vehicle.state.position for ray in moved))


if __name__ == "__main__":
    unittest.main()
