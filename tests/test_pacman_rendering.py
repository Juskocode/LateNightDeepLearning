import unittest
from unittest.mock import patch

import pygame

from pacManRf.src.game.constants import TILE_SIZE, WALL_BLUE, WALL_GLOW
from pacManRf.src.game.pacmanGame import MAZE_TEMPLATE, PacmanGame


def _rgb(surface: pygame.Surface, point: tuple[int, int]) -> tuple[int, int, int]:
    return tuple(surface.get_at(point)[:3])


def _blue_components(
    surface: pygame.Surface,
    center: tuple[int, int],
    radius: int = 8,
) -> int:
    """Count four-connected blue paths in a small square around a vertex."""
    center_x, center_y = center
    remaining = {
        (x, y)
        for y in range(center_y - radius, center_y + radius + 1)
        for x in range(center_x - radius, center_x + radius + 1)
        if _rgb(surface, (x, y)) == WALL_BLUE
    }
    components = 0
    while remaining:
        components += 1
        pending = [remaining.pop()]
        while pending:
            x, y = pending.pop()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
    return components


class PacmanMazeRenderingTests(unittest.TestCase):
    def setUp(self):
        self.game = PacmanGame(render=False, seed=3)

    def test_concave_wall_vertex_has_one_continuous_blue_path(self):
        # This vertex has walls in every quadrant except north-east.
        vertex_x, vertex_y = 10, 2
        quadrants = (
            MAZE_TEMPLATE[vertex_y - 1][vertex_x - 1] == "#",
            MAZE_TEMPLATE[vertex_y - 1][vertex_x] == "#",
            MAZE_TEMPLATE[vertex_y][vertex_x - 1] == "#",
            MAZE_TEMPLATE[vertex_y][vertex_x] == "#",
        )
        self.assertEqual(quadrants, (True, False, True, True))

        center = (vertex_x * TILE_SIZE, vertex_y * TILE_SIZE)
        # The concave join is routed through the opposite, south-west wall.
        join = (center[0] - 2, center[1] + 2)
        self.assertEqual(_rgb(self.game.maze_surface, join), WALL_BLUE)
        self.assertEqual(_blue_components(self.game.maze_surface, center), 1)

    def test_every_concave_vertex_has_one_continuous_blue_path(self):
        checked = 0
        for vertex_y in range(1, len(MAZE_TEMPLATE)):
            for vertex_x in range(1, len(MAZE_TEMPLATE[0])):
                quadrants = (
                    MAZE_TEMPLATE[vertex_y - 1][vertex_x - 1] == "#",
                    MAZE_TEMPLATE[vertex_y - 1][vertex_x] == "#",
                    MAZE_TEMPLATE[vertex_y][vertex_x - 1] == "#",
                    MAZE_TEMPLATE[vertex_y][vertex_x] == "#",
                )
                if sum(quadrants) != 3:
                    continue
                checked += 1
                center = (vertex_x * TILE_SIZE, vertex_y * TILE_SIZE)
                self.assertEqual(
                    _blue_components(self.game.maze_surface, center, radius=4),
                    1,
                    msg=f"broken concave contour at maze vertex {(vertex_x, vertex_y)}",
                )
        self.assertEqual(checked, 42)

    def test_diagonal_walls_remain_separate_blue_paths(self):
        # North-east and south-west touch only at this grid vertex.
        vertex_x, vertex_y = 8, 2
        quadrants = (
            MAZE_TEMPLATE[vertex_y - 1][vertex_x - 1] == "#",
            MAZE_TEMPLATE[vertex_y - 1][vertex_x] == "#",
            MAZE_TEMPLATE[vertex_y][vertex_x - 1] == "#",
            MAZE_TEMPLATE[vertex_y][vertex_x] == "#",
        )
        self.assertEqual(quadrants, (False, True, True, False))

        center = (vertex_x * TILE_SIZE, vertex_y * TILE_SIZE)
        self.assertNotEqual(_rgb(self.game.maze_surface, center), WALL_BLUE)
        self.assertEqual(_blue_components(self.game.maze_surface, center), 2)

    def test_every_diagonal_contact_remains_two_separate_paths(self):
        checked = 0
        for vertex_y in range(1, len(MAZE_TEMPLATE)):
            for vertex_x in range(1, len(MAZE_TEMPLATE[0])):
                quadrants = (
                    MAZE_TEMPLATE[vertex_y - 1][vertex_x - 1] == "#",
                    MAZE_TEMPLATE[vertex_y - 1][vertex_x] == "#",
                    MAZE_TEMPLATE[vertex_y][vertex_x - 1] == "#",
                    MAZE_TEMPLATE[vertex_y][vertex_x] == "#",
                )
                if quadrants not in (
                    (True, False, False, True),
                    (False, True, True, False),
                ):
                    continue
                checked += 1
                center = (vertex_x * TILE_SIZE, vertex_y * TILE_SIZE)
                self.assertNotEqual(_rgb(self.game.maze_surface, center), WALL_BLUE)
                self.assertEqual(
                    _blue_components(self.game.maze_surface, center, radius=4),
                    2,
                    msg=f"diagonal contours bridged at maze vertex {(vertex_x, vertex_y)}",
                )
        self.assertEqual(checked, 3)

    def test_all_glow_paths_are_painted_before_blue_paths(self):
        with patch.object(pygame.draw, "line", wraps=pygame.draw.line) as draw_line:
            self.game._build_maze_surface()

        wall_colors = [
            call.args[1]
            for call in draw_line.call_args_list
            if len(call.args) > 1 and call.args[1] in (WALL_GLOW, WALL_BLUE)
        ]
        glow_indices = [
            index for index, color in enumerate(wall_colors) if color == WALL_GLOW
        ]
        blue_indices = [
            index for index, color in enumerate(wall_colors) if color == WALL_BLUE
        ]
        self.assertTrue(glow_indices)
        self.assertEqual(len(glow_indices), len(blue_indices))
        self.assertLess(max(glow_indices), min(blue_indices))


if __name__ == "__main__":
    unittest.main()
