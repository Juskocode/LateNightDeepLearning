"""Compact, inspectable observations for the Pacman environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from pacManRf.src.game.constants import Direction, FRIGHTENED_SECONDS, STARTING_LIVES

from .config import DEFAULT_OBSERVATION_SIZE


CARDINAL_DIRECTIONS = (
    Direction.UP,
    Direction.DOWN,
    Direction.LEFT,
    Direction.RIGHT,
)


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    """An NN input together with semantic data used by the observability UI."""

    values: np.ndarray
    labels: tuple[str, ...]
    vision_rays: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values.astype(float).tolist(),
            "labels": list(self.labels),
            "vision_rays": [dict(ray) for ray in self.vision_rays],
        }


class PacmanObservationEncoder:
    """Encode useful maze, ghost, and episode context into a fixed-size vector.

    Each cardinal ray reports wall, pellet, power-pellet, dangerous-ghost, and
    edible-ghost proximity.  ``1`` means one tile away, ``0.5`` means two tiles
    away, and ``0`` means that object is not visible before a wall.  Remaining
    features retain individual ghost and global state, so the vector is richer
    than a collision-only controller while remaining easy to explain live.
    """

    def __init__(self, ghost_slots: int = 4):
        if ghost_slots <= 0:
            raise ValueError("ghost_slots must be positive")
        self.ghost_slots = int(ghost_slots)
        self.labels = self._build_labels()

    @property
    def size(self) -> int:
        return len(self.labels)

    def _build_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        for direction in CARDINAL_DIRECTIONS:
            prefix = f"ray_{direction.name.lower()}"
            labels.extend(
                f"{prefix}_{kind}"
                for kind in ("wall", "pellet", "power", "danger", "edible")
            )
        labels.extend(f"direction_{direction.name.lower()}" for direction in CARDINAL_DIRECTIONS)
        for index in range(self.ghost_slots):
            labels.extend(
                (
                    f"ghost_{index}_dx",
                    f"ghost_{index}_dy",
                    f"ghost_{index}_distance",
                    f"ghost_{index}_dangerous",
                    f"ghost_{index}_edible",
                )
            )
        labels.extend(
            (
                "nearest_pellet_dx",
                "nearest_pellet_dy",
                "nearest_pellet_distance",
                "nearest_power_dx",
                "nearest_power_dy",
                "nearest_power_distance",
                "frightened_fraction",
                "remaining_pellets_fraction",
                "lives_fraction",
                "level_scaled",
                "mode_scatter",
                "mode_chase",
                "phase_ready",
                "phase_active",
                "phase_dying",
                "phase_clearing",
            )
        )
        return tuple(labels)

    @staticmethod
    def _cell(game: Any, x: int, y: int) -> str:
        cell_reader = getattr(game, "_cell", None)
        if callable(cell_reader):
            return str(cell_reader(x, y))
        rows = len(game.maze)
        cols = len(game.maze[0]) if rows else 0
        if not (0 <= x < cols and 0 <= y < rows):
            return "#"
        return str(game.maze[y][x])

    @staticmethod
    def _relative(origin: tuple[int, int], target: tuple[int, int], cols: int, rows: int) -> tuple[float, float, float]:
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        return (
            float(np.clip(dx / max(1, cols - 1), -1.0, 1.0)),
            float(np.clip(dy / max(1, rows - 1), -1.0, 1.0)),
            float(np.clip((abs(dx) + abs(dy)) / max(1, cols + rows - 2), 0.0, 1.0)),
        )

    @staticmethod
    def _ghost_flags(game: Any, ghost: Any) -> tuple[float, float]:
        released = bool(getattr(ghost, "released", True))
        eaten = bool(getattr(ghost, "eaten", False))
        frightened = float(getattr(game, "frightened_timer", 0.0)) > 0.0
        dangerous = released and not eaten and not frightened
        edible = released and not eaten and frightened
        return float(dangerous), float(edible)

    def _ray(self, game: Any, direction: Direction) -> tuple[list[float], dict[str, Any]]:
        px, py = game.player.grid
        dx, dy = direction.vector
        cols, rows = int(game.cols), int(game.rows)
        max_steps = max(cols, rows) + 1
        proximities = {"wall": 0.0, "pellet": 0.0, "power": 0.0, "danger": 0.0, "edible": 0.0}
        path: list[tuple[int, int]] = []

        ghosts_by_grid: dict[tuple[int, int], list[Any]] = {}
        for ghost in getattr(game, "ghosts", ()):
            ghosts_by_grid.setdefault(tuple(ghost.grid), []).append(ghost)

        x, y = px, py
        for step in range(1, max_steps + 1):
            x += dx
            y += dy
            if y == 9 and x < 0:
                x = cols - 1
            elif y == 9 and x >= cols:
                x = 0
            cell = self._cell(game, x, y)
            if cell == "#":
                proximities["wall"] = 1.0 / step
                break
            path.append((x, y))
            if cell == "." and proximities["pellet"] == 0.0:
                proximities["pellet"] = 1.0 / step
            elif cell == "o" and proximities["power"] == 0.0:
                proximities["power"] = 1.0 / step
            for ghost in ghosts_by_grid.get((x, y), ()):
                dangerous, edible = self._ghost_flags(game, ghost)
                if dangerous and proximities["danger"] == 0.0:
                    proximities["danger"] = 1.0 / step
                if edible and proximities["edible"] == 0.0:
                    proximities["edible"] = 1.0 / step

        values = [proximities[key] for key in ("wall", "pellet", "power", "danger", "edible")]
        ray = {
            "direction": direction.name,
            **proximities,
            "path": [list(point) for point in path],
        }
        return values, ray

    def _nearest_target(self, game: Any, cells: Sequence[str]) -> tuple[float, float, float]:
        origin = tuple(game.player.grid)
        targets = [
            (x, y)
            for y, row in enumerate(game.maze)
            for x, cell in enumerate(row)
            if cell in cells
        ]
        if not targets:
            return 0.0, 0.0, 0.0
        target = min(targets, key=lambda point: abs(point[0] - origin[0]) + abs(point[1] - origin[1]))
        return self._relative(origin, target, int(game.cols), int(game.rows))

    def observe(self, game: Any) -> ObservationFrame:
        values: list[float] = []
        rays: list[Mapping[str, Any]] = []
        for direction in CARDINAL_DIRECTIONS:
            ray_values, ray = self._ray(game, direction)
            values.extend(ray_values)
            rays.append(ray)

        player_direction = game.player.direction
        values.extend(float(player_direction == direction) for direction in CARDINAL_DIRECTIONS)

        origin = tuple(game.player.grid)
        ghosts = list(getattr(game, "ghosts", ()))
        for index in range(self.ghost_slots):
            if index >= len(ghosts):
                values.extend((0.0, 0.0, 0.0, 0.0, 0.0))
                continue
            ghost = ghosts[index]
            values.extend(self._relative(origin, tuple(ghost.grid), int(game.cols), int(game.rows)))
            values.extend(self._ghost_flags(game, ghost))

        values.extend(self._nearest_target(game, (".", "o")))
        values.extend(self._nearest_target(game, ("o",)))
        frightened = float(getattr(game, "frightened_timer", 0.0))
        remaining = sum(cell in ".o" for row in game.maze for cell in row)
        total = max(1, int(getattr(game, "total_dots", remaining)))
        values.extend(
            (
                float(np.clip(frightened / FRIGHTENED_SECONDS, 0.0, 1.0)),
                float(np.clip(remaining / total, 0.0, 1.0)),
                float(np.clip(getattr(game, "lives", 0) / STARTING_LIVES, 0.0, 1.5)),
                float(np.clip((getattr(game, "level", 1) - 1) / 20.0, 0.0, 1.0)),
            )
        )
        mode_name = getattr(getattr(game, "ghost_mode", None), "name", "").upper()
        values.extend((float(mode_name == "SCATTER"), float(mode_name == "CHASE")))
        phase_name = getattr(getattr(game, "phase", None), "name", "").upper()
        values.extend(float(phase_name == name) for name in ("READY", "ACTIVE", "DYING", "CLEARING"))

        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.size,):
            raise RuntimeError(f"encoder produced {observation.size} values; expected {self.size}")
        return ObservationFrame(observation, self.labels, tuple(rays))

    def encode(self, game: Any) -> np.ndarray:
        return self.observe(game).values


if PacmanObservationEncoder().size != DEFAULT_OBSERVATION_SIZE:
    raise RuntimeError("DEFAULT_OBSERVATION_SIZE does not match PacmanObservationEncoder")

