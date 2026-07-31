import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from pacManRf.src.game.constants import (
    Direction,
    GamePhase,
    PELLET,
    PLAYER_SPEED,
    TILE_SIZE,
)
from pacManRf.src.game.pacmanGame import PacmanGame
from pacManRf.src.game.projectiles import (
    DespawnReason,
    GhostProjectileSystem,
    PacmanSlowState,
    ProjectileKind,
)


class SequenceRng:
    def __init__(self, *values):
        self.values = iter(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self.values)


OPEN_GRID = lambda _cell: True


class GhostProjectileSystemTests(unittest.TestCase):
    def test_early_unlocks_roll_once_per_ability_and_persist_across_levels(self):
        rng = SequenceRng(0.19, 0.20, 0.90, 0.05)
        system = GhostProjectileSystem(rng)

        self.assertEqual(rng.calls, 2)
        self.assertTrue(system.is_unlocked("BLINKY", 1))
        self.assertFalse(system.is_unlocked("INKY", 1))
        self.assertTrue(system.is_unlocked("INKY", 3))

        original_unlocks = system.unlocks
        system.reset_level()
        self.assertIs(system.unlocks, original_unlocks)
        self.assertEqual(rng.calls, 2)

        system.start_new_run()
        self.assertEqual(rng.calls, 4)
        self.assertFalse(system.is_unlocked("BLINKY", 1))
        self.assertTrue(system.is_unlocked("INKY", 1))

    def test_blinky_fireball_has_five_tile_range_and_real_cooldown(self):
        system = GhostProjectileSystem(SequenceRng(0.9, 0.9))
        shot = system.try_fire("BLINKY", 3, (0, 0), (1, 0), OPEN_GRID)

        self.assertIsNotNone(shot)
        self.assertEqual(shot.spec.range_tiles, 5)
        self.assertFalse(system.can_fire("BLINKY", 3))
        self.assertIsNone(system.try_fire("BLINKY", 3, (0, 0), (1, 0), OPEN_GRID))

        system.update_cooldowns(shot.spec.cooldown_seconds)
        self.assertTrue(system.can_fire("BLINKY", 3))

    def test_projectile_disappears_at_wall(self):
        system = GhostProjectileSystem(SequenceRng(0.0, 0.0))
        system.try_fire(
            "BLINKY",
            1,
            (0, 0),
            (1, 0),
            lambda cell: cell[0] <= 2,
        )

        events = system.update(1.0, None, lambda cell: cell[0] <= 2)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].reason, DespawnReason.WALL)
        self.assertEqual(events[0].cell, (2, 0))
        self.assertEqual(events[0].blocked_cell, (3, 0))
        self.assertFalse(system.active_projectiles)

    def test_fireball_can_hit_on_its_final_range_tile(self):
        system = GhostProjectileSystem(SequenceRng(0.0, 0.0))
        system.try_fire("BLINKY", 1, (0, 0), (1, 0), OPEN_GRID)

        (event,) = system.update(1.0, (5, 0), OPEN_GRID)

        self.assertEqual(event.reason, DespawnReason.HIT_PACMAN)
        self.assertEqual(event.kind, ProjectileKind.FIREBALL)
        self.assertEqual(event.cell, (5, 0))
        self.assertEqual(event.damage, 1)
        self.assertEqual(event.speed_multiplier, 1.0)

    def test_freeze_ball_has_fifteen_tile_range_and_slows_by_fifteen_percent(self):
        system = GhostProjectileSystem(SequenceRng(0.0, 0.0))
        shot = system.try_fire("INKY", 1, (0, 0), (1, 0), OPEN_GRID)
        self.assertEqual(shot.spec.range_tiles, 15)

        (event,) = system.update(3.0, (15, 0), OPEN_GRID)

        self.assertEqual(event.reason, DespawnReason.HIT_PACMAN)
        self.assertEqual(event.kind, ProjectileKind.FREEZE_BALL)
        self.assertEqual(event.slow_fraction, 0.15)
        self.assertAlmostEqual(event.speed_multiplier, 0.85)
        self.assertEqual(event.damage, 0)

        slow = PacmanSlowState()
        self.assertTrue(slow.apply(event))
        self.assertAlmostEqual(slow.speed_multiplier, 0.85)
        slow.update(event.slow_duration_seconds)
        self.assertFalse(slow.active)
        self.assertEqual(slow.speed_multiplier, 1.0)

    def test_range_expiry_and_custom_collision_remove_projectiles(self):
        range_system = GhostProjectileSystem(SequenceRng(0.0, 0.0))
        range_system.try_fire("BLINKY", 1, (0, 0), (1, 0), OPEN_GRID)
        (range_event,) = range_system.update(1.0, None, OPEN_GRID)
        self.assertEqual(range_event.reason, DespawnReason.RANGE)
        self.assertEqual(range_event.cell, (5, 0))

        collision_system = GhostProjectileSystem(SequenceRng(0.0, 0.0))
        collision_system.try_fire("INKY", 1, (0, 0), (1, 0), OPEN_GRID)
        (collision_event,) = collision_system.update(
            1.0,
            None,
            OPEN_GRID,
            collision_test=lambda _shot, cell: cell == (3, 0),
        )
        self.assertEqual(collision_event.reason, DespawnReason.COLLISION)
        self.assertEqual(collision_event.cell, (3, 0))

    def test_line_of_sight_respects_corridor_walls_and_range(self):
        system = GhostProjectileSystem(SequenceRng(0.0, 0.0))
        blocked = lambda cell: cell != (3, 0)

        self.assertIsNone(
            system.try_fire_at_target("BLINKY", 1, (0, 0), (4, 0), blocked)
        )
        self.assertIsNone(
            system.try_fire_at_target("BLINKY", 1, (0, 0), (6, 0), OPEN_GRID)
        )
        shot = system.try_fire_at_target("BLINKY", 1, (0, 0), (5, 0), OPEN_GRID)
        self.assertIsNotNone(shot)
        self.assertEqual(shot.direction, (1, 0))

    def test_topology_callback_supports_tunnel_wrapping(self):
        system = GhostProjectileSystem(SequenceRng(0.0, 0.0))

        def tunnel_step(cell, direction):
            x, y = cell[0] + direction[0], cell[1] + direction[1]
            return (x % 20, y)

        shot = system.try_fire_at_target(
            "BLINKY",
            1,
            (0, 9),
            (19, 9),
            OPEN_GRID,
            next_cell=tunnel_step,
        )
        self.assertIsNotNone(shot)
        self.assertEqual(shot.direction, (-1, 0))

        (event,) = system.update(
            0.2,
            (19, 9),
            OPEN_GRID,
            next_cell=tunnel_step,
        )
        self.assertTrue(event.hit_pacman)
        self.assertEqual(event.cell, (19, 9))


class PacmanProjectileIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.game = PacmanGame(render=False, seed=1)
        self.game.phase = GamePhase.ACTIVE

    def _spawn(self, owner, origin, direction):
        self.game.level = 3
        self.game.projectile_system.reset_level(initial_cooldown_seconds=0.0)
        projectile = self.game.projectile_system.try_fire(
            owner,
            self.game.level,
            origin,
            direction,
            self.game._projectile_cell_is_walkable,
            next_cell=self.game._projectile_next_cell,
        )
        self.assertIsNotNone(projectile)
        self.game.projectile_shots_fired += 1
        return projectile

    def test_level_three_unlocks_both_weapons_and_early_roll_persists(self):
        # random.Random(1) rolls 0.134... then 0.847..., so only Blinky
        # receives the independent 20% level-one unlock.
        unlocks = self.game.projectile_system.unlocks
        self.assertTrue(self.game.projectile_system.is_unlocked("BLINKY", 1))
        self.assertFalse(self.game.projectile_system.is_unlocked("INKY", 1))

        self.game.next_level()
        self.assertIs(self.game.projectile_system.unlocks, unlocks)
        self.assertTrue(self.game.projectile_system.is_unlocked("BLINKY", 2))
        self.assertFalse(self.game.projectile_system.is_unlocked("INKY", 2))

        self.game.level = 3
        self.assertTrue(self.game.projectile_system.is_unlocked("BLINKY", 3))
        self.assertTrue(self.game.projectile_system.is_unlocked("INKY", 3))

    def test_fireball_hit_costs_exactly_one_life_and_despawns(self):
        self.game.player.reset_position((5, 3), Direction.LEFT)
        self._spawn("BLINKY", (0, 3), (1, 0))

        events = self.game._update_projectiles(1.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, ProjectileKind.FIREBALL)
        self.assertTrue(events[0].hit_pacman)
        self.assertEqual(self.game.lives, 2)
        self.assertEqual(self.game.fireball_hits, 1)
        self.assertEqual(self.game.phase, GamePhase.DYING)
        self.assertFalse(self.game.active_projectiles)

    def test_moving_away_does_not_collide_with_departed_grid_cell(self):
        self.game.player.reset_position((5, 3), Direction.RIGHT)
        self.game._start_move(self.game.player, Direction.RIGHT)
        self._spawn("BLINKY", (4, 3), (1, 0))

        for _ in range(9):
            self.game._update_projectiles(1 / 60)
            self.game._advance(self.game.player, 1 / 60)

        # The projectile has entered grid cell (5, 3), but Pacman is already
        # well ahead in pixel space and must not be hit by stale grid ownership.
        self.assertEqual(self.game.player.grid, (5, 3))
        self.assertEqual(self.game.active_projectiles[0].cell, (5, 3))
        self.assertGreater(
            self.game.player.position.distance_to(
                pygame.Vector2(
                    self.game._projectile_pixel_position(
                        self.game.active_projectiles[0]
                    )
                )
            ),
            TILE_SIZE * 0.7,
        )
        self.assertEqual(self.game.lives, 3)
        self.assertEqual(self.game.phase, GamePhase.ACTIVE)

    def test_large_time_step_cannot_tunnel_through_interpolated_pacman(self):
        self.game.player.reset_position((4, 3), Direction.LEFT)
        self.game.player.position.x += TILE_SIZE / 2
        self._spawn("BLINKY", (0, 3), (1, 0))

        events = self.game._update_projectiles(1.0)

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].hit_pacman)
        self.assertGreater(events[0].position_tiles[0], 4.0)
        self.assertLess(events[0].position_tiles[0], 4.5)
        self.assertEqual(self.game.lives, 2)
        self.assertFalse(self.game.active_projectiles)

    def test_armed_ghost_fires_only_with_clear_unsuppressed_sightline(self):
        self.game.level = 3
        self.game.player.reset_position((10, 3), Direction.LEFT)
        blinky = self.game.ghosts[0]
        blinky.reset_position((5, 3), Direction.RIGHT)
        blinky.released = True
        self.game.projectile_system.reset_level(initial_cooldown_seconds=0.0)

        self.game._try_fire_projectile(blinky)

        self.assertEqual(len(self.game.active_projectiles), 1)
        self.assertEqual(self.game.active_projectiles[0].owner, "BLINKY")
        self.assertEqual(self.game.projectile_shots_fired, 1)

        self.game.projectile_system.reset_level(initial_cooldown_seconds=0.0)
        self.game.frightened_timer = 1.0
        self.game._try_fire_projectile(blinky)
        self.assertFalse(self.game.active_projectiles)

    def test_freeze_hit_slows_pacman_fifteen_percent_without_life_loss(self):
        self.game.player.reset_position((15, 3), Direction.LEFT)
        self._spawn("INKY", (0, 3), (1, 0))

        events = self.game._update_projectiles(3.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, ProjectileKind.FREEZE_BALL)
        self.assertEqual(self.game.lives, 3)
        self.assertEqual(self.game.freeze_ball_hits, 1)
        self.assertTrue(self.game.player_slow.active)
        self.assertAlmostEqual(self.game.player_speed_multiplier, 0.85)
        self.assertAlmostEqual(self.game.player.speed, PLAYER_SPEED * 0.85)

    def test_projectile_uses_cached_sprite_frames_in_real_renderer(self):
        self._spawn("BLINKY", (3, 3), (1, 0))
        self.game._render(0.016)

        center = (round(3.5 * TILE_SIZE), round(3.5 * TILE_SIZE))
        self.assertNotEqual(self.game.display.get_at(center)[:3], PELLET)
        sprite = self.game.projectile_sprites[ProjectileKind.FIREBALL]
        self.assertIsInstance(sprite, pygame.sprite.Sprite)
        self.assertEqual(len(sprite.frames), 8)
        self.assertEqual(len(sprite.impact_frames), 8)

    def test_level_transition_clears_shots_but_preserves_run_combat_stats(self):
        unlocks = self.game.projectile_system.unlocks
        self._spawn("BLINKY", (3, 3), (1, 0))
        self.game.projectile_shots_fired = 7
        self.game.fireball_hits = 2
        self.game.freeze_ball_hits = 1

        self.game.next_level()

        self.assertFalse(self.game.active_projectiles)
        self.assertFalse(self.game.player_slow.active)
        self.assertIs(self.game.projectile_system.unlocks, unlocks)
        self.assertEqual(self.game.projectile_shots_fired, 7)
        self.assertEqual(self.game.fireball_hits, 2)
        self.assertEqual(self.game.freeze_ball_hits, 1)

    def test_new_run_resets_combat_stats_and_rerolls_from_seed(self):
        self.game.projectile_shots_fired = 7
        self.game.fireball_hits = 2
        self.game.freeze_ball_hits = 1
        self.game.rng.seed(1)

        self.game.restart()

        self.assertEqual(self.game.level, 1)
        self.assertEqual(self.game.projectile_shots_fired, 0)
        self.assertEqual(self.game.fireball_hits, 0)
        self.assertEqual(self.game.freeze_ball_hits, 0)
        self.assertTrue(self.game.projectile_system.is_unlocked("BLINKY", 1))
        self.assertFalse(self.game.projectile_system.is_unlocked("INKY", 1))


if __name__ == "__main__":
    unittest.main()
