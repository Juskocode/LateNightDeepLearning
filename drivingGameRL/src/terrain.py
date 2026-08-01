"""Terrain coefficients for traction, drag, and visual particles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TerrainKind(str, Enum):
    ASPHALT = "asphalt"
    WET_ASPHALT = "wet asphalt"
    GRAVEL = "gravel"
    GRASS = "grass"
    MUD = "mud"


@dataclass(frozen=True, slots=True)
class Terrain:
    kind: TerrainKind
    grip: float
    rolling_resistance: float
    engine_efficiency: float
    color: tuple[int, int, int]
    particle_color: tuple[int, int, int]


TERRAINS: dict[TerrainKind, Terrain] = {
    TerrainKind.ASPHALT: Terrain(
        TerrainKind.ASPHALT, 1.00, 0.012, 1.00, (53, 58, 66), (120, 126, 136)
    ),
    TerrainKind.WET_ASPHALT: Terrain(
        TerrainKind.WET_ASPHALT, 0.88, 0.010, 0.98, (42, 54, 66), (112, 139, 160)
    ),
    TerrainKind.GRAVEL: Terrain(
        TerrainKind.GRAVEL, 0.76, 0.035, 0.91, (142, 120, 83), (189, 157, 104)
    ),
    TerrainKind.GRASS: Terrain(
        TerrainKind.GRASS, 0.62, 0.052, 0.82, (49, 101, 63), (91, 137, 72)
    ),
    TerrainKind.MUD: Terrain(
        TerrainKind.MUD, 0.48, 0.080, 0.70, (83, 61, 45), (117, 79, 51)
    ),
}


def terrain(kind: TerrainKind) -> Terrain:
    return TERRAINS[kind]
