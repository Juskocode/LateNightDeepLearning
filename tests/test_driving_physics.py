import math
import unittest

from drivingGameRL.src.circuits import Circuit, all_circuits, circuit_names, get_circuit
from drivingGameRL.src.environment import DrivingAction, DrivingEnv
from drivingGameRL.src.math2d import Vec2
from drivingGameRL.src.terrain import TerrainKind, terrain
from drivingGameRL.src.vehicle import CarBuild, DriverControls, Vehicle


class DrivingPhysicsTests(unittest.TestCase):
    def test_circuit_registry_is_deterministic_and_contains_three_tracks(self):
        self.assertEqual(circuit_names(), circuit_names())
        self.assertGreaterEqual(len(all_circuits()), 3)
        self.assertEqual(len(circuit_names()), len(set(circuit_names())))
        for name in circuit_names():
            circuit = get_circuit(name)
            self.assertGreater(circuit.length, 1_000)
            self.assertEqual(circuit.slug, name)

    def test_builtin_centerlines_do_not_self_intersect(self):
        def orientation(first, second, third):
            return (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (
                third.x - first.x
            )

        def intersects(first, second, third, fourth):
            return (
                orientation(first, second, third) * orientation(first, second, fourth)
                < 0.0
                and orientation(third, fourth, first)
                * orientation(third, fourth, second)
                < 0.0
            )

        def point_segment_distance(point, start, end):
            segment = end - start
            along = max(
                0.0,
                min(1.0, (point - start).dot(segment) / segment.length_squared()),
            )
            return (point - (start + segment * along)).length()

        def segment_distance(first, second, third, fourth):
            return min(
                point_segment_distance(first, third, fourth),
                point_segment_distance(second, third, fourth),
                point_segment_distance(third, first, second),
                point_segment_distance(fourth, first, second),
            )

        for circuit in all_circuits():
            count = len(circuit.points)
            for first_index in range(count):
                for second_index in range(first_index + 1, count):
                    if (
                        second_index == (first_index + 1) % count
                        or first_index == (second_index + 1) % count
                    ):
                        continue
                    self.assertFalse(
                        intersects(
                            circuit.points[first_index],
                            circuit.points[(first_index + 1) % count],
                            circuit.points[second_index],
                            circuit.points[(second_index + 1) % count],
                        ),
                        f"{circuit.slug} segments {first_index} and {second_index} cross",
                    )
                    self.assertGreater(
                        segment_distance(
                            circuit.points[first_index],
                            circuit.points[(first_index + 1) % count],
                            circuit.points[second_index],
                            circuit.points[(second_index + 1) % count],
                        ),
                        circuit.track_width,
                        f"{circuit.slug} has overlapping non-adjacent road segments",
                    )

    def test_invalid_circuit_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            Circuit(
                "duplicate",
                "Duplicate",
                (Vec2(), Vec2(), Vec2(10, 10)),
                20,
                10,
                TerrainKind.GRASS,
            )
        with self.assertRaises(ValueError):
            Circuit(
                "infinite-width",
                "Infinite width",
                (Vec2(), Vec2(10, 0), Vec2(10, 10)),
                math.inf,
                10,
                TerrainKind.GRASS,
            )
        with self.assertRaises(ValueError):
            Circuit(
                "nonfinite",
                "Nonfinite",
                (Vec2(), Vec2(math.inf, 1), Vec2(10, 10)),
                20,
                10,
                TerrainKind.GRASS,
            )

    def test_surface_sectors_and_runoff_have_different_grip(self):
        circuit = get_circuit("harbor_loop")
        wet_point, normal = circuit.point_tangent_at(0.23)
        runoff_point = wet_point + normal.perpendicular() * circuit.track_width
        self.assertEqual(circuit.terrain_at(wet_point).kind, TerrainKind.WET_ASPHALT)
        self.assertEqual(circuit.terrain_at(runoff_point).kind, TerrainKind.GRASS)
        self.assertGreater(
            circuit.terrain_at(wet_point).grip,
            circuit.terrain_at(runoff_point).grip,
        )

    def test_motor_upgrade_improves_acceleration_and_max_speed(self):
        base = Vehicle(CarBuild(motor=0))
        upgraded = Vehicle(CarBuild(motor=5))
        for vehicle in (base, upgraded):
            vehicle.reset(Vec2(), 0.0)
            for _ in range(180):
                vehicle.step(
                    DriverControls(throttle=1.0),
                    terrain(TerrainKind.ASPHALT),
                    1.0 / 60.0,
                )
        self.assertGreater(upgraded.state.speed, base.state.speed * 1.25)
        self.assertGreater(upgraded.build.max_speed, base.build.max_speed)

    def test_every_component_capability_improves_at_each_upgrade_level(self):
        capability_by_component = {
            "motor": ("acceleration", "max_speed"),
            "wheels": ("steering_rate", "steering_response"),
            "suspension": ("stability",),
            "grip": ("grip_multiplier",),
        }
        for component, capabilities in capability_by_component.items():
            builds = [CarBuild(**{component: level}) for level in range(6)]
            for capability in capabilities:
                values = [getattr(build, capability) for build in builds]
                self.assertTrue(
                    all(first < second for first, second in zip(values, values[1:])),
                    f"{component}.{capability} must improve at every level",
                )

    def test_steering_cannot_rotate_a_stationary_car(self):
        vehicle = Vehicle(CarBuild(wheels=5))
        vehicle.reset(Vec2(), 0.4)
        for _ in range(60):
            vehicle.step(
                DriverControls(steering=1.0),
                terrain(TerrainKind.ASPHALT),
                1.0 / 60.0,
            )
        self.assertAlmostEqual(vehicle.state.heading, 0.4)
        self.assertEqual(vehicle.state.position, Vec2())

    def test_wheels_suspension_and_grip_change_handling(self):
        def corner(build):
            vehicle = Vehicle(build)
            vehicle.reset(Vec2(), 0.0)
            for _ in range(150):
                vehicle.step(
                    DriverControls(throttle=0.85, steering=0.7),
                    terrain(TerrainKind.ASPHALT),
                    1.0 / 60.0,
                )
            return abs(vehicle.state.heading), abs(vehicle.last_telemetry.lateral_speed)

        base_heading, _ = corner(CarBuild())
        wheel_heading, _ = corner(CarBuild(wheels=5))
        suspension_heading, _ = corner(CarBuild(suspension=5))
        grip_heading, _ = corner(CarBuild(grip=5))
        self.assertGreater(wheel_heading, base_heading)
        self.assertGreater(suspension_heading, base_heading)
        self.assertGreater(grip_heading, base_heading)

        def residual_lateral(build):
            vehicle = Vehicle(build)
            vehicle.reset(Vec2(), 0.0)
            vehicle.state.velocity = Vec2(100.0, 40.0)
            for _ in range(30):
                vehicle.step(DriverControls(), terrain(TerrainKind.ASPHALT), 1.0 / 60.0)
            return abs(vehicle.last_telemetry.lateral_speed)

        base_lateral = residual_lateral(CarBuild())
        suspension_lateral = residual_lateral(CarBuild(suspension=5))
        grip_lateral = residual_lateral(CarBuild(grip=5))
        self.assertLess(suspension_lateral, base_lateral)
        self.assertLess(grip_lateral, base_lateral)

    def test_environment_is_reproducible_and_observation_is_finite(self):
        first = DrivingEnv("pine_sprint", seed=91)
        second = DrivingEnv("pine_sprint", seed=91)
        actions = [DrivingAction.ACCELERATE] * 80 + [DrivingAction.STEER_RIGHT] * 50
        for action in actions:
            first_result = first.step(action)
            second_result = second.step(action)
            self.assertEqual(first_result.observation, second_result.observation)
            self.assertEqual(first_result.reward, second_result.reward)
        self.assertEqual(len(first_result.observation), len(first.OBSERVATION_LABELS))
        self.assertTrue(all(math.isfinite(value) for value in first_result.observation))

    def test_reset_restores_initial_state_and_trajectory(self):
        env = DrivingEnv("pine_sprint", seed=31)
        initial = env.reset(seed=31)
        actions = [DrivingAction.ACCELERATE] * 45 + [DrivingAction.STEER_LEFT] * 20
        first = [env.step(action).observation for action in actions]
        reset_observation = env.reset(seed=31)
        second = [env.step(action).observation for action in actions]
        self.assertEqual(reset_observation, initial)
        self.assertEqual(second, first)

    def test_start_line_oscillation_cannot_farm_lap_rewards(self):
        env = DrivingEnv("harbor_loop")
        for _ in range(6):
            for progress in (0.999, 0.001):
                point, tangent = env.circuit.point_tangent_at(progress)
                env.vehicle.state.position = point
                env.vehicle.state.velocity = Vec2()
                env.vehicle.state.heading = math.atan2(tangent.y, tangent.x)
                result = env.step(DrivingAction.COAST)
                self.assertFalse(result.info["lap_completed"])
                self.assertEqual(result.info["reward_terms"]["lap"], 0.0)
        self.assertEqual(env.laps, 0)

        env.reset()
        for index in range(1, 21):
            progress = (index % 20) / 20.0
            point, tangent = env.circuit.point_tangent_at(progress)
            env.vehicle.state.position = point
            env.vehicle.state.velocity = Vec2()
            env.vehicle.state.heading = math.atan2(tangent.y, tangent.x)
            result = env.step(DrivingAction.COAST)
        self.assertTrue(result.info["lap_completed"])
        self.assertEqual(env.laps, 1)
        self.assertEqual(result.info["reward_terms"]["lap"], 20.0)

    def test_collision_recovers_to_track_and_stays_finite_under_stress(self):
        env = DrivingEnv("harbor_loop", seed=5)
        circuit = env.circuit
        track_point, _ = circuit.point_tangent_at(0.0)
        center = Vec2(
            sum(point.x for point in circuit.points) / len(circuit.points),
            sum(point.y for point in circuit.points) / len(circuit.points),
        )
        outward = (track_point - center).normalized()
        env.vehicle.state.position = track_point + outward * (
            circuit.collision_radius + 45.0
        )
        env.vehicle.state.velocity = outward * 130.0
        result = env.step(DrivingAction.ACCELERATE)
        self.assertTrue(result.info["collided"])
        self.assertLessEqual(
            circuit.project(env.vehicle.state.position).distance,
            circuit.collision_radius + 1e-7,
        )

        for index in range(2_000):
            action = (
                DrivingAction.STEER_LEFT
                if index % 37 < 19
                else DrivingAction.STEER_RIGHT
            )
            result = env.step(action)
            state = env.vehicle.state
            values = (
                state.position.x,
                state.position.y,
                state.velocity.x,
                state.velocity.y,
                state.heading,
                state.damage,
                result.reward,
            )
            self.assertTrue(all(math.isfinite(value) for value in values))

    def test_continuous_barrier_contact_counts_as_one_collision(self):
        env = DrivingEnv("harbor_loop")
        circuit = env.circuit
        track_point, _ = circuit.point_tangent_at(0.0)
        center = Vec2(
            sum(point.x for point in circuit.points) / len(circuit.points),
            sum(point.y for point in circuit.points) / len(circuit.points),
        )
        outward = (track_point - center).normalized()
        for _ in range(12):
            env.vehicle.state.position = track_point + outward * (
                circuit.collision_radius + 2.0
            )
            env.vehicle.state.velocity = outward * 80.0
            result = env.step(DrivingAction.ACCELERATE)
            self.assertTrue(result.info["collided"])
        self.assertEqual(env.collisions, 1)

        env.vehicle.state.position = track_point
        env.vehicle.state.velocity = Vec2()
        env.step(DrivingAction.COAST)
        env.vehicle.state.position = track_point + outward * (
            circuit.collision_radius + 2.0
        )
        env.vehicle.state.velocity = outward * 80.0
        result = env.step(DrivingAction.ACCELERATE)
        self.assertTrue(result.info["collision_started"])
        self.assertEqual(env.collisions, 2)

    def test_inward_penetration_is_contact_and_impact_telemetry_is_current(self):
        env = DrivingEnv("harbor_loop")
        circuit = env.circuit
        point, tangent = circuit.point_tangent_at(0.25)
        outward = tangent.perpendicular() * -1.0
        env.vehicle.state.position = point + outward * (circuit.collision_radius + 2.0)
        env.vehicle.state.velocity = outward * -12.0
        result = env.step(DrivingAction.COAST)
        self.assertTrue(result.info["collided"])
        self.assertTrue(result.info["collision_started"])
        self.assertEqual(result.info["impact_speed"], 0.0)
        self.assertAlmostEqual(
            circuit.project(env.vehicle.state.position).distance,
            circuit.collision_radius,
        )

        env.reset()
        point, tangent = circuit.point_tangent_at(0.25)
        outward = tangent.perpendicular() * -1.0
        env.vehicle.state.position = point + outward * (circuit.collision_radius + 2.0)
        env.vehicle.state.velocity = outward * 130.0
        result = env.step(DrivingAction.ACCELERATE)
        telemetry = result.info["telemetry"]
        self.assertAlmostEqual(telemetry.speed, env.vehicle.state.speed)
        self.assertAlmostEqual(
            result.observation[0],
            env.vehicle.state.speed / env.vehicle.build.max_speed,
        )

    def test_terrain_speed_and_traction_follow_surface_quality(self):
        speeds = []
        grips = []
        kinds = (
            TerrainKind.ASPHALT,
            TerrainKind.WET_ASPHALT,
            TerrainKind.GRAVEL,
            TerrainKind.GRASS,
            TerrainKind.MUD,
        )
        for kind in kinds:
            vehicle = Vehicle()
            vehicle.reset(Vec2(), 0.0)
            for _ in range(180):
                vehicle.step(DriverControls(throttle=1.0), terrain(kind), 1.0 / 60.0)
            speeds.append(vehicle.state.speed)
            grips.append(vehicle.last_telemetry.effective_grip)
        self.assertEqual(speeds, sorted(speeds, reverse=True))
        self.assertEqual(grips, sorted(grips, reverse=True))

    def test_telemetry_exposes_terrain_components_and_capabilities(self):
        build = CarBuild(motor=1, wheels=2, suspension=3, grip=4)
        env = DrivingEnv(build=build)
        env.step(DrivingAction.ACCELERATE)
        snapshot = env.telemetry()
        self.assertEqual(
            snapshot["components"],
            {"motor": 1, "wheels": 2, "suspension": 3, "grip": 4},
        )
        self.assertIn(snapshot["terrain"], {kind.value for kind in TerrainKind})
        self.assertGreater(snapshot["capabilities"]["max_speed"], 0.0)

    def test_invalid_build_and_action_are_rejected(self):
        with self.assertRaises(ValueError):
            CarBuild(grip=6)
        with self.assertRaises(ValueError):
            CarBuild(motor=True)
        env = DrivingEnv()
        with self.assertRaises(ValueError):
            env.step(999)
        with self.assertRaises(ValueError):
            env.step(DriverControls(throttle=math.nan))
        with self.assertRaises(ValueError):
            env.step(DriverControls(steering=math.inf))
        with self.assertRaises(ValueError):
            DrivingEnv(max_steps=0)


if __name__ == "__main__":
    unittest.main()
