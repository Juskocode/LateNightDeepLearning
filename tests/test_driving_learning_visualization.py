import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from drivingGameRL.src.environment import DrivingAction, DrivingEnv
from drivingGameRL.src.learning_visualization import (
    CAR_ROTATION_STEP_DEGREES,
    COLORS,
    DrivingLearningVisualization,
    LEARNING_WINDOW_SIZE,
    POPULATION_CAR_COLORS,
    TEXT_SURFACE_CACHE_LIMIT,
)
from drivingGameRL.src.rendering import TRACK_VIEW_WIDTH, WINDOW_HEIGHT as TRACK_HEIGHT


def learning_telemetry(env: DrivingEnv) -> dict[str, object]:
    observation = list(env.observation())
    first_weights = [
        [((output + 1) * (source + 2) % 9 - 4) / 10 for source in range(12)]
        for output in range(8)
    ]
    second_weights = [
        [((output + 3) * (source + 1) % 7 - 3) / 8 for source in range(8)]
        for output in range(5)
    ]
    q_values = [0.12, 0.74, -0.31, 0.28, 0.09]
    return {
        "algorithm": "genetic_double_dqn",
        "generation": 12,
        "member_index": 3,
        "population_size": 8,
        "current_fitness": 18.42,
        "best_fitness": 27.8,
        "mean_fitness": 13.9,
        "population": [
            {
                "index": index,
                "member_id": 100 + index,
                "fitness": 7.0 + index * 2.1,
                "status": "evaluating" if index == 3 else "evaluated",
            }
            for index in range(8)
        ],
        "generation_history": [
            {"generation": index, "best": 8.0 + index * 1.4, "mean": 4.0 + index}
            for index in range(12)
        ],
        "observation": observation,
        "q_values": q_values,
        "selected_action": 1,
        "action_counts": [120, 640, 84, 251, 205],
        "epsilon": 0.18,
        "loss": 0.024,
        "td_error": 0.11,
        "gradient_steps": 2_840,
        "target_syncs": 14,
        "replay_size": 3_200,
        "replay_capacity": 10_000,
        "batch_size": 64,
        "loss_history": [0.9 / (index + 1) for index in range(30)],
        "epsilon_history": [max(0.05, 1.0 - index * 0.03) for index in range(30)],
        "memory_samples": [
            {
                "action": index % 5,
                "reward": (index - 5) / 8,
                "done": index in (0, 11),
            }
            for index in range(12)
        ],
        "network": {
            "architecture": [12, 8, 5],
            "parameter_count": 149,
            "q_values": q_values,
            "layers": [
                {
                    "name": "observation",
                    "kind": "input",
                    "activations": observation,
                },
                {
                    "name": "hidden_1",
                    "kind": "hidden",
                    "activations": [0.0, 0.31, 0.82, 0.14, 0.0, 1.2, 0.46, 0.07],
                    "weights": first_weights,
                },
                {
                    "name": "q_values",
                    "kind": "output",
                    "activations": q_values,
                    "weights": second_weights,
                },
            ],
        },
    }


def rollout_rays(
    position: tuple[float, float], heading: float
) -> list[dict[str, object]]:
    distances = (48.0, 72.0, 108.0, 84.0, 60.0)
    angles = (-math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2)
    return [
        {
            "origin": position,
            "endpoint": (
                position[0] + math.cos(heading + angle) * distance,
                position[1] + math.sin(heading + angle) * distance,
            ),
            "distance": distance,
            "normalized_distance": distance / 150.0,
            "hit": distance < 150.0,
        }
        for angle, distance in zip(angles, distances)
    ]


def population_rollouts(generation: int = 12) -> list[dict[str, object]]:
    poses = (
        ((205.0, 190.0), 0.15),
        ((425.0, 345.0), 1.25),
        ((665.0, 485.0), -0.8),
    )
    return [
        {
            "member_id": member_id,
            "index": member_id,
            "generation": generation,
            "position": position,
            "heading": heading,
            "speed": 42.0 + member_id,
            "progress": 0.2 + member_id * 0.1,
            "action": member_id % 5,
            "rays": rollout_rays(position, heading),
        }
        for member_id, (position, heading) in enumerate(poses)
    ]


class DrivingLearningVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.font.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.env = DrivingEnv("canyon_maze", seed=19)
        for _ in range(18):
            self.env.step(DrivingAction.ACCELERATE)
        self.telemetry = learning_telemetry(self.env)
        self.visualization = DrivingLearningVisualization(self.env, self.telemetry)

    def test_training_render_is_fixed_size_deterministic_and_read_only(self):
        before = self.env.telemetry()
        first = self.visualization.draw().copy()
        first_pixels = pygame.image.tostring(first, "RGB")
        second = self.visualization.render_training().copy()

        self.assertEqual(first.get_size(), LEARNING_WINDOW_SIZE)
        self.assertEqual(
            first.get_bounding_rect(), pygame.Rect(0, 0, *LEARNING_WINDOW_SIZE)
        )
        self.assertEqual(pygame.image.tostring(second, "RGB"), first_pixels)
        after = self.env.telemetry()
        self.assertEqual(after["steps"], before["steps"])
        self.assertEqual(after["position"], before["position"])
        self.assertEqual(after["current_lap_time"], before["current_lap_time"])

    def test_parallel_generation_status_is_visible_and_read_only(self):
        before = self.env.telemetry()
        sequential = self.visualization.draw(
            telemetry={
                **self.telemetry,
                "parallel_workers": 1,
                "active_member_indices": [0],
                "last_tick_member_count": 1,
            }
        ).copy()
        parallel = self.visualization.draw(
            telemetry={
                **self.telemetry,
                "parallel_workers": 6,
                "active_member_indices": list(range(8)),
                "last_tick_member_count": 8,
            }
        ).copy()

        self.assertNotEqual(
            pygame.image.tostring(parallel, "RGB"),
            pygame.image.tostring(sequential, "RGB"),
        )
        self.assertEqual(self.env.telemetry(), before)

    def test_curriculum_view_marks_the_random_episode_origin_read_only(self):
        env = DrivingEnv(
            "canyon_maze", seed=29, random_start_curriculum=True
        )
        before = env.telemetry()
        visualization = DrivingLearningVisualization(env, learning_telemetry(env))

        surface = visualization.draw().copy()
        viewport = pygame.Rect(30, 132, 732, 598)
        point, _ = env.circuit.point_tangent_at(env.lap_origin_progress)
        center = (
            round(viewport.x + point.x * viewport.width / TRACK_VIEW_WIDTH),
            round(viewport.y + point.y * viewport.height / TRACK_HEIGHT),
        )
        # The policy car begins directly on the marker and covers its center;
        # include the gate endpoints and label around that car.
        neighborhood = pygame.Rect(center[0] - 55, center[1] - 45, 110, 80)
        neighborhood.clamp_ip(viewport)
        yellow_pixels = sum(
            surface.get_at((x, y))[:3] == COLORS["yellow"]
            for x in range(neighborhood.left, neighborhood.right)
            for y in range(neighborhood.top, neighborhood.bottom)
        )

        self.assertGreater(yellow_pixels, 2)
        self.assertEqual(env.telemetry(), before)

    def test_ray_and_generation_car_toggles_change_only_opted_in_track_layers(self):
        rollouts = population_rollouts()
        disabled = {
            **self.telemetry,
            "show_sensor_rays": False,
            "show_population_cars": False,
            "population_rollouts": rollouts,
        }
        no_optional_payload = {
            key: value
            for key, value in disabled.items()
            if key != "population_rollouts"
        }
        off_with_payload = self.visualization.draw(telemetry=disabled).copy()
        off_without_payload = self.visualization.draw(
            telemetry=no_optional_payload
        ).copy()
        self.assertEqual(
            pygame.image.tostring(off_with_payload, "RGB"),
            pygame.image.tostring(off_without_payload, "RGB"),
        )

        track = pygame.Rect(30, 132, 732, 598)
        off_track = off_with_payload.subsurface(track).copy()
        rays = (
            self.visualization.draw(
                telemetry={
                    **disabled,
                    "show_sensor_rays": True,
                }
            )
            .subsurface(track)
            .copy()
        )
        cars = (
            self.visualization.draw(
                telemetry={
                    **disabled,
                    "show_population_cars": True,
                }
            )
            .subsurface(track)
            .copy()
        )
        self.assertNotEqual(
            pygame.image.tostring(rays, "RGB"),
            pygame.image.tostring(off_track, "RGB"),
        )
        self.assertNotEqual(
            pygame.image.tostring(cars, "RGB"),
            pygame.image.tostring(off_track, "RGB"),
        )

    def test_same_generation_rollout_cars_have_stable_distinct_colors_and_positions(
        self,
    ):
        rollouts = population_rollouts()
        # This brightly colored stale-generation pose must not be drawn.
        rollouts.append(
            {
                "member_id": 8,
                "index": 8,
                "generation": 11,
                "position": (770.0, 120.0),
                "heading": 0.0,
                "rays": rollout_rays((770.0, 120.0), 0.0),
            }
        )
        telemetry = {
            **self.telemetry,
            "show_sensor_rays": True,
            "show_population_cars": True,
            "population_rollout_generation": 12,
            "population_rollouts": rollouts,
        }
        before = self.env.telemetry()
        first = self.visualization.draw(telemetry=telemetry).copy()
        second = self.visualization.draw(telemetry=telemetry).copy()
        self.assertEqual(
            pygame.image.tostring(first, "RGB"), pygame.image.tostring(second, "RGB")
        )

        viewport = pygame.Rect(30, 132, 732, 598)
        scale_x = viewport.width / TRACK_VIEW_WIDTH
        scale_y = viewport.height / TRACK_HEIGHT
        for rollout in rollouts[:3]:
            member_id = int(rollout["member_id"])
            color = POPULATION_CAR_COLORS[member_id % len(POPULATION_CAR_COLORS)]
            x, y = rollout["position"]
            center = (
                round(viewport.x + x * scale_x),
                round(viewport.y + y * scale_y),
            )
            neighborhood = pygame.Rect(center[0] - 24, center[1] - 24, 48, 48)
            matching = sum(
                first.get_at((pixel_x, pixel_y))[:3] == color
                for pixel_x in range(neighborhood.left, neighborhood.right)
                for pixel_y in range(neighborhood.top, neighborhood.bottom)
            )
            self.assertGreater(matching, 8, f"member {member_id} color was not drawn")

        after = self.env.telemetry()
        self.assertEqual(after["steps"], before["steps"])
        self.assertEqual(after["position"], before["position"])

    def test_keyboard_and_click_navigation_select_every_tab(self):
        self.assertEqual(self.visualization.active_tab, "OVERVIEW")
        self.assertTrue(
            self.visualization.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2)
            )
        )
        self.assertEqual(self.visualization.active_tab, "NETWORK")
        self.visualization.draw()

        self.assertTrue(
            self.visualization.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB)
            )
        )
        self.assertEqual(self.visualization.active_tab, "MEMORY")
        self.visualization.draw()

        overview = self.visualization.tab_rects["OVERVIEW"]
        self.assertTrue(
            self.visualization.handle_event(
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, button=1, pos=overview.center
                )
            )
        )
        self.assertEqual(self.visualization.active_tab, "OVERVIEW")
        self.assertFalse(
            self.visualization.handle_event(
                pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE)
            )
        )

    def test_all_tabs_save_substantial_distinct_screenshots(self):
        fingerprints = set()
        with tempfile.TemporaryDirectory() as directory:
            for tab in self.visualization.TABS:
                self.visualization.set_tab(tab)
                surface = self.visualization.draw()
                output = Path(directory) / f"{tab.lower()}.png"
                pygame.image.save(surface, output)
                self.assertGreater(output.stat().st_size, 30_000)
                fingerprints.add(hash(pygame.image.tostring(surface, "RGB")))
        self.assertEqual(len(fingerprints), len(self.visualization.TABS))

    def test_incomplete_and_malformed_optional_telemetry_is_safe(self):
        sparse = {
            "algorithm": "dqn",
            "epsilon": float("nan"),
            "q_values": "not a sequence",
            "population": None,
            "network": {"architecture": [12, 5], "layers": None},
            "replay_capacity": 0,
            "show_sensor_rays": "definitely",
            "show_population_cars": True,
            "population_rollouts": [
                None,
                {"position": (float("inf"), 4.0), "rays": "broken"},
                {"position": "nowhere", "rays": [{"endpoint": None}]},
            ],
        }
        for tab in self.visualization.TABS:
            self.visualization.set_tab(tab)
            result = self.visualization.draw(telemetry=sparse)
            self.assertIs(result, self.visualization.surface)
            self.assertEqual(result.get_size(), LEARNING_WINDOW_SIZE)

        custom = pygame.Surface(LEARNING_WINDOW_SIZE)
        self.assertEqual(
            DrivingLearningVisualization(self.env, surface=custom).draw().get_size(),
            LEARNING_WINDOW_SIZE,
        )
        with self.assertRaises(ValueError):
            DrivingLearningVisualization(self.env, surface=pygame.Surface((80, 80)))

    def test_race_view_derives_poses_and_exposes_prominent_return_hint(self):
        champion = DrivingEnv("canyon_maze", seed=20)
        for _ in range(11):
            champion.step(DrivingAction.STEER_LEFT)
        telemetry = {
            "generation": 12,
            "best_fitness": 27.8,
            "best_member": 5,
            "human_progress": 0.41,
            "champion_progress": 0.36,
            "human_speed": 78.0,
            "champion_speed": 74.0,
        }

        race = self.visualization.render_race(self.env, champion, telemetry)
        running_pixels = pygame.image.tostring(race, "RGB")

        self.assertEqual(race.get_size(), LEARNING_WINDOW_SIZE)
        cyan_pixels = sum(
            race.get_at((x, y))[:3] == COLORS["cyan"]
            for x in range(1_030, 1_380, 4)
            for y in range(620, 740, 4)
        )
        self.assertGreater(cyan_pixels, 500)
        finished = self.visualization.render_race(
            self.env,
            champion,
            {
                **telemetry,
                "winner": "human",
                "elapsed": 42.75,
                "human_finish_time": 42.75,
            },
        )
        self.assertNotEqual(pygame.image.tostring(finished, "RGB"), running_pixels)
        self.assertTrue(
            self.visualization.handle_event(
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p)
            )
        )
        self.assertTrue(self.visualization.return_to_training_requested)

    def test_race_accepts_explicit_mapping_and_tuple_poses(self):
        champion = DrivingEnv("canyon_maze", seed=21)
        human_pose = {"position": (180.0, 220.0), "heading": 0.4}
        champion_pose = (260.0, 250.0, -0.2)

        result = self.visualization.draw_race(
            self.env,
            champion,
            {"human": {"progress": 0.2}, "champion": {"progress": 0.3}},
            human_pose,
            champion_pose,
        )

        self.assertIs(result, self.visualization.surface)
        self.assertEqual(
            result.get_bounding_rect(), pygame.Rect(0, 0, *LEARNING_WINDOW_SIZE)
        )

    def test_explicit_race_pose_anchors_fallback_rays_to_the_drawn_car(self):
        explicit_pose = {"position": (400.0, 400.0), "heading": 0.25}

        rays = self.visualization._environment_rays(self.env, explicit_pose)

        self.assertEqual(len(rays), 5)
        for ray in rays:
            self.assertEqual(ray["origin"], explicit_pose["position"])

    def test_exact_sensor_api_takes_precedence_over_observation_fallback(self):
        expected = self.env.sensor_rays()
        observation = list(self.env.observation())
        observation[-5:] = [0.0] * 5
        telemetry = {
            **self.telemetry,
            "observation": observation,
            "show_sensor_rays": True,
            "show_population_cars": False,
        }

        rays = self.visualization._environment_rays(self.env, observation=observation)
        self.visualization.draw(telemetry=telemetry)

        self.assertEqual(len(rays), len(expected))
        for rendered, snapshot in zip(rays, expected):
            self.assertAlmostEqual(
                rendered["normalized_distance"], snapshot.normalized_distance
            )
            self.assertAlmostEqual(rendered["endpoint"][0], snapshot.endpoint.x)
            self.assertAlmostEqual(rendered["endpoint"][1], snapshot.endpoint.y)

    def test_dynamic_visual_caches_are_reused_and_bounded(self):
        size = (180, 120)
        first_layer = self.visualization._ray_layer(size)
        pygame.draw.circle(first_layer, (255, 255, 255, 255), (12, 12), 3)
        second_layer = self.visualization._ray_layer(size)
        self.assertIs(second_layer, first_layer)
        self.assertEqual(second_layer.get_at((12, 12)).a, 0)

        for index in range(TEXT_SURFACE_CACHE_LIMIT + 30):
            self.visualization._render_text(
                f"frame {index}", size=10, color=COLORS["text"]
            )
        self.assertEqual(
            len(self.visualization._text_surfaces), TEXT_SURFACE_CACHE_LIMIT
        )

        color = POPULATION_CAR_COLORS[0]
        for degrees in range(360):
            self.visualization._draw_car((40, 40), math.radians(degrees), color)
        variants = [key for key in self.visualization._car_rotations if key[0] == color]
        self.assertLessEqual(
            len(variants), math.ceil(360 / CAR_ROTATION_STEP_DEGREES) + 1
        )

    def test_race_sensor_toggle_draws_both_ray_fans_without_moving_either_env(self):
        champion = DrivingEnv("canyon_maze", seed=22)
        for _ in range(7):
            champion.step(DrivingAction.ACCELERATE)
        human_before = self.env.telemetry()
        champion_before = champion.telemetry()
        off = self.visualization.draw_race(
            self.env, champion, {"show_sensor_rays": False}
        ).copy()
        on = self.visualization.draw_race(
            self.env, champion, {"show_sensor_rays": True}
        ).copy()
        track = pygame.Rect(30, 132, 970, 598)
        self.assertNotEqual(
            pygame.image.tostring(off.subsurface(track), "RGB"),
            pygame.image.tostring(on.subsurface(track), "RGB"),
        )
        self.assertEqual(self.env.telemetry()["position"], human_before["position"])
        self.assertEqual(champion.telemetry()["position"], champion_before["position"])


if __name__ == "__main__":
    unittest.main()
