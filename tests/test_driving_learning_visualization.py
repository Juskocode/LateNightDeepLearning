import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from drivingGameRL.src.environment import DrivingAction, DrivingEnv
from drivingGameRL.src.learning_visualization import (
    COLORS,
    DrivingLearningVisualization,
    LEARNING_WINDOW_SIZE,
)


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


if __name__ == "__main__":
    unittest.main()
