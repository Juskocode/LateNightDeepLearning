import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from drivingGameRL.main import main
from drivingGameRL.src.game import DrivingGame
from drivingGameRL.src.math2d import Vec2
from drivingGameRL.src.sprites import CarSprite, ParticleSystem
from drivingGameRL.src.vehicle import CarBuild, DriverControls, Vehicle


class DrivingRenderingTests(unittest.TestCase):
    def test_default_sprite_loads_and_missing_asset_has_fallback(self):
        pygame.init()
        loaded = CarSprite(CarBuild(grip=3))
        fallback = CarSprite(CarBuild(), "/definitely/missing/car.png")
        self.assertTrue(loaded.using_external_image)
        self.assertFalse(fallback.using_external_image)
        self.assertGreater(loaded.image.get_bounding_rect().width, 20)
        self.assertGreater(fallback.image.get_bounding_rect().height, 40)

    def test_particles_are_seeded_and_bounded(self):
        first = ParticleSystem(seed=8)
        second = ParticleSystem(seed=8)
        for system in (first, second):
            for _ in range(20):
                system.emit_collision(Vec2(50.0, 50.0), 80.0)
            system.update(1.0 / 60.0)
        self.assertLessEqual(len(first), first.MAX_PARTICLES)
        self.assertEqual(len(first), len(second))
        first_state = [
            (sprite.position, sprite.velocity, sprite.remaining)
            for sprite in first.sprites.sprites()
        ]
        second_state = [
            (sprite.position, sprite.velocity, sprite.remaining)
            for sprite in second.sprites.sprites()
        ]
        self.assertEqual(first_state, second_state)

    def test_particle_reset_replays_seeded_collision_effect(self):
        system = ParticleSystem(seed=17)
        system.emit_collision(Vec2(40.0, 60.0), 90.0)
        first = [
            (sprite.position, sprite.velocity, sprite.lifetime)
            for sprite in system.sprites.sprites()
        ]
        system.reset()
        system.emit_collision(Vec2(40.0, 60.0), 90.0)
        second = [
            (sprite.position, sprite.velocity, sprite.lifetime)
            for sprite in system.sprites.sprites()
        ]
        self.assertEqual(second, first)
        system.emit_collision(Vec2(), 0.0)
        self.assertEqual(len(system), len(second))

    def test_game_saves_full_hud_screenshot_and_reports_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "driving.png"
            game = DrivingGame(
                "harbor_loop",
                build=CarBuild(2, 2, 2, 2),
                seed=12,
                render=False,
            )
            for _ in range(90):
                game.step(game.autopilot_controls())
            game.save_screenshot(output)
            snapshot = game.telemetry()
            self.assertTrue(snapshot["sprite_asset_loaded"])
            self.assertGreater(output.stat().st_size, 20_000)
            self.assertEqual(pygame.image.load(output).get_size(), (1100, 700))

    def test_screenshot_cli_is_finite_without_explicit_headless_or_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.png"
            exit_code = main(
                [
                    "--circuit",
                    "desert_switchback",
                    "--screenshot",
                    str(output),
                    "--seed",
                    "3",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertGreater(output.stat().st_size, 20_000)


if __name__ == "__main__":
    unittest.main()
