import math
import unittest
from dataclasses import FrozenInstanceError

from drivingGameRL.src.circuits import all_circuits
from drivingGameRL.src.environment import DrivingAction, DrivingEnv, SensorRay
from drivingGameRL.src.math2d import Vec2


class DrivingSensorRayTests(unittest.TestCase):
    def test_nine_rays_follow_observation_label_order(self):
        env = DrivingEnv("harbor_loop", seed=17)
        rays = env.sensor_rays()

        self.assertIsInstance(rays, tuple)
        self.assertEqual(len(rays), 9)
        self.assertTrue(all(isinstance(ray, SensorRay) for ray in rays))
        self.assertEqual(
            env.OBSERVATION_LABELS[-9:],
            (
                "ray_left",
                "ray_left_wide",
                "ray_left_forward",
                "ray_left_near",
                "ray_forward",
                "ray_right_near",
                "ray_right_forward",
                "ray_right_wide",
                "ray_right",
            ),
        )
        expected_angles = tuple(
            env.vehicle.state.heading + relative
            for relative in env.SENSOR_RELATIVE_ANGLES
        )
        self.assertEqual(tuple(ray.angle for ray in rays), expected_angles)
        self.assertEqual(
            env.SENSOR_RELATIVE_ANGLES[::2],
            (-math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2),
        )

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
                observation[-9:],
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
            tuple(ray.normalized_distance for ray in second), observation[-9:]
        )

        env.vehicle.state.heading += 0.125
        moved = env.sensor_rays()
        self.assertIsNot(moved, first)
        self.assertTrue(all(ray.origin == env.vehicle.state.position for ray in moved))

    def test_coarse_to_fine_sampling_bounds_projection_work_for_dense_fan(self):
        class CountingCircuit:
            def __init__(self, source):
                self.source = source
                self.projections = 0
                self.batch_queries = 0
                self.batch_samples = 0

            def project(self, position):
                self.projections += 1
                return self.source.project(position)

            def distances_to_centerline(self, positions):
                self.batch_queries += 1
                self.batch_samples += len(positions)
                return self.source.distances_to_centerline(positions)

            def __getattr__(self, name):
                return getattr(self.source, name)

        source = DrivingEnv("harbor_loop", seed=3).circuit
        circuit = CountingCircuit(source)
        env = DrivingEnv(circuit, seed=3)
        circuit.projections = 0
        circuit.batch_queries = 0
        circuit.batch_samples = 0
        env.vehicle.state.heading += 0.001

        rays = env.sensor_rays()

        maximum_batched_samples = 9 * (
            math.ceil(env.SENSOR_MAX_DISTANCE / env.SENSOR_SAMPLE_STEP)
            + env.SENSOR_REFINEMENT_STEPS
        )
        self.assertEqual(len(rays), 9)
        self.assertEqual(circuit.projections, 0)
        self.assertLessEqual(circuit.batch_queries, 1 + env.SENSOR_REFINEMENT_STEPS)
        self.assertLessEqual(circuit.batch_samples, maximum_batched_samples)

    def test_batched_centerline_distances_match_scalar_projection(self):
        for circuit in all_circuits():
            points = []
            for index in range(17):
                point, tangent = circuit.point_tangent_at(index / 17.0)
                normal = tangent.perpendicular()
                points.extend(
                    (
                        point,
                        point + normal * (circuit.track_width * 0.37),
                        point - normal * (circuit.collision_radius + 7.0),
                        Vec2(point.x + 13.25, point.y - 9.75),
                    )
                )

            batched = circuit.distances_to_centerline(points)
            scalar = tuple(circuit.project(point).distance for point in points)

            with self.subTest(circuit=circuit.slug):
                self.assertEqual(len(batched), len(scalar))
                for actual, expected in zip(batched, scalar):
                    self.assertAlmostEqual(actual, expected, places=11)

    def test_batched_fan_matches_scalar_full_resolution_on_hairpins(self):
        env = DrivingEnv("canyon_maze", seed=14)

        def scalar_ray(angle):
            origin = env.vehicle.state.position
            direction = Vec2.from_angle(angle)
            low = 0.0
            distance = env.SENSOR_SAMPLE_STEP
            while distance <= env.SENSOR_MAX_DISTANCE:
                sample = origin + direction * distance
                if (
                    env.circuit.project(sample).distance
                    >= env.circuit.collision_radius
                ):
                    high = distance
                    for _ in range(env.SENSOR_REFINEMENT_STEPS):
                        midpoint = (low + high) * 0.5
                        if (
                            env.circuit.project(origin + direction * midpoint).distance
                            >= env.circuit.collision_radius
                        ):
                            high = midpoint
                        else:
                            low = midpoint
                    return high
                low = distance
                distance += env.SENSOR_SAMPLE_STEP
            return env.SENSOR_MAX_DISTANCE

        for pose_index in range(24):
            point, tangent = env.circuit.point_tangent_at(pose_index / 24.0)
            env.vehicle.state.position = point
            env.vehicle.state.heading = math.atan2(tangent.y, tangent.x) + (
                pose_index % 3 - 1
            ) * 0.12
            rays = env.sensor_rays()
            expected = tuple(scalar_ray(ray.angle) for ray in rays)
            with self.subTest(pose=pose_index):
                for ray, distance in zip(rays, expected):
                    self.assertAlmostEqual(ray.distance, distance, places=12)


if __name__ == "__main__":
    unittest.main()
