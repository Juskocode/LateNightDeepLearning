"""Terrain coefficients for traction, drag, and visual particles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class TerrainKind(str, Enum):
    ASPHALT = "asphalt"
    WET_ASPHALT = "wet asphalt"
    GRAVEL = "gravel"
    GRASS = "grass"
    MUD = "mud"
    SAND = "sand"
    SNOW = "snow"
    ICE = "ice"


class ParticleMode(str, Enum):
    """Visual material emitted by a tire moving over a surface."""

    NONE = "none"
    DUST = "dust"
    SPRAY = "spray"
    DEBRIS = "debris"
    MUD = "mud"
    SNOW = "snow"


@dataclass(frozen=True, slots=True)
class Terrain:
    kind: TerrainKind
    grip: float
    rolling_resistance: float
    engine_efficiency: float
    color: tuple[int, int, int]
    particle_color: tuple[int, int, int]
    particle_mode: ParticleMode = ParticleMode.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TerrainKind):
            raise ValueError("Terrain kind must be a TerrainKind")
        coefficients = (self.grip, self.rolling_resistance, self.engine_efficiency)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in coefficients
        ):
            raise ValueError("Terrain coefficients must be finite numbers")
        if self.grip <= 0.0 or self.rolling_resistance < 0.0:
            raise ValueError(
                "Terrain grip must be positive and resistance non-negative"
            )
        if self.engine_efficiency <= 0.0:
            raise ValueError("Terrain engine efficiency must be positive")
        for color in (self.color, self.particle_color):
            if (
                not isinstance(color, tuple)
                or len(color) != 3
                or any(
                    type(channel) is not int or not 0 <= channel <= 255
                    for channel in color
                )
            ):
                raise ValueError("Terrain colors must be RGB integer tuples")
        if not isinstance(self.particle_mode, ParticleMode):
            raise ValueError("Terrain particle mode must be a ParticleMode")


TERRAINS: dict[TerrainKind, Terrain] = {
    TerrainKind.ASPHALT: Terrain(
        TerrainKind.ASPHALT,
        1.00,
        0.012,
        1.00,
        (53, 58, 66),
        (120, 126, 136),
        ParticleMode.NONE,
    ),
    TerrainKind.WET_ASPHALT: Terrain(
        TerrainKind.WET_ASPHALT,
        0.88,
        0.010,
        0.98,
        (42, 54, 66),
        (112, 139, 160),
        ParticleMode.SPRAY,
    ),
    TerrainKind.GRAVEL: Terrain(
        TerrainKind.GRAVEL,
        0.76,
        0.035,
        0.91,
        (142, 120, 83),
        (189, 157, 104),
        ParticleMode.DUST,
    ),
    TerrainKind.GRASS: Terrain(
        TerrainKind.GRASS,
        0.62,
        0.052,
        0.82,
        (49, 101, 63),
        (91, 137, 72),
        ParticleMode.DEBRIS,
    ),
    TerrainKind.MUD: Terrain(
        TerrainKind.MUD,
        0.48,
        0.080,
        0.70,
        (83, 61, 45),
        (117, 79, 51),
        ParticleMode.MUD,
    ),
    TerrainKind.SAND: Terrain(
        TerrainKind.SAND,
        0.54,
        0.095,
        0.66,
        (188, 154, 94),
        (225, 192, 126),
        ParticleMode.DUST,
    ),
    TerrainKind.SNOW: Terrain(
        TerrainKind.SNOW,
        0.40,
        0.064,
        0.72,
        (201, 215, 219),
        (235, 242, 243),
        ParticleMode.SNOW,
    ),
    TerrainKind.ICE: Terrain(
        TerrainKind.ICE,
        0.22,
        0.004,
        0.93,
        (137, 185, 207),
        (205, 232, 240),
        ParticleMode.NONE,
    ),
}


def terrain(kind: TerrainKind) -> Terrain:
    return TERRAINS[kind]
