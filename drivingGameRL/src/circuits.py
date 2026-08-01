"""Analytic closed circuits shared by simulation, sensors, and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .math2d import Vec2, clamp
from .terrain import Terrain, TerrainKind, terrain


@dataclass(frozen=True, slots=True)
class SurfaceSector:
    """Replace the road surface over a normalized portion of one lap."""

    start: float
    end: float
    kind: TerrainKind

    def __post_init__(self) -> None:
        for name, value in (("start", self.start), ("end", self.end)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value < 1.0
            ):
                raise ValueError(f"Sector {name} must be finite and in [0, 1)")
        if self.start == self.end:
            raise ValueError("A surface sector must cover a non-zero portion of a lap")
        if not isinstance(self.kind, TerrainKind):
            raise ValueError("Sector kind must be a TerrainKind")

    def contains(self, progress: float) -> bool:
        progress %= 1.0
        if self.start < self.end:
            return self.start <= progress < self.end
        return progress >= self.start or progress < self.end

    def intervals(self) -> tuple[tuple[float, float], ...]:
        """Return one or two half-open intervals on a linearized lap."""

        if self.start < self.end:
            return ((self.start, self.end),)
        return ((self.start, 1.0), (0.0, self.end))


@dataclass(frozen=True, slots=True)
class TrackProjection:
    point: Vec2
    tangent: Vec2
    distance: float
    signed_offset: float
    progress: float
    segment_index: int


@dataclass(frozen=True, slots=True)
class Circuit:
    slug: str
    name: str
    points: tuple[Vec2, ...]
    track_width: float
    runoff_width: float
    runoff: TerrainKind
    sectors: tuple[SurfaceSector, ...] = ()
    description: str = ""
    _segment_lengths: tuple[float, ...] = field(
        init=False, repr=False, compare=False
    )
    _projection_segments: tuple[
        tuple[Vec2, Vec2, float, Vec2, float], ...
    ] = field(init=False, repr=False, compare=False)
    _total_length: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("A closed circuit requires at least three points")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.track_width, self.runoff_width)
        ):
            raise ValueError("Track and runoff widths must be positive")
        if not all(
            math.isfinite(value)
            for point in self.points
            for value in (point.x, point.y)
        ):
            raise ValueError("Circuit points must be finite")
        lengths = tuple(
            (self.points[(index + 1) % len(self.points)] - point).length()
            for index, point in enumerate(self.points)
        )
        if any(length <= 1e-9 for length in lengths):
            raise ValueError("Consecutive circuit points must be distinct")
        traversed = 0.0
        projection_segments = []
        for index, start in enumerate(self.points):
            segment = self.points[(index + 1) % len(self.points)] - start
            length = lengths[index]
            projection_segments.append(
                (
                    start,
                    segment,
                    segment.length_squared(),
                    segment / length,
                    traversed,
                )
            )
            traversed += length
        object.__setattr__(self, "_segment_lengths", lengths)
        object.__setattr__(self, "_projection_segments", tuple(projection_segments))
        object.__setattr__(self, "_total_length", traversed)
        if not isinstance(self.runoff, TerrainKind):
            raise ValueError("Circuit runoff must be a TerrainKind")
        if any(not isinstance(sector, SurfaceSector) for sector in self.sectors):
            raise ValueError("Circuit sectors must be SurfaceSector values")
        for first_index, first in enumerate(self.sectors):
            for second in self.sectors[first_index + 1 :]:
                if any(
                    max(first_start, second_start) < min(first_end, second_end)
                    for first_start, first_end in first.intervals()
                    for second_start, second_end in second.intervals()
                ):
                    raise ValueError("Circuit surface sectors must not overlap")

    @property
    def segment_lengths(self) -> tuple[float, ...]:
        return self._segment_lengths

    @property
    def length(self) -> float:
        return self._total_length

    @property
    def collision_radius(self) -> float:
        return self.track_width * 0.5 + self.runoff_width

    def project(self, position: Vec2) -> TrackProjection:
        if not all(math.isfinite(value) for value in (position.x, position.y)):
            raise ValueError("Projected position must be finite")
        best: tuple[float, float, float, Vec2, float, int, float] | None = None
        position_x, position_y = position.x, position.y

        for index, (start, segment, length_squared, tangent, traversed) in enumerate(
            self._projection_segments
        ):
            along = 0.0
            if length_squared > 1e-12:
                along = clamp(
                    (
                        (position_x - start.x) * segment.x
                        + (position_y - start.y) * segment.y
                    )
                    / length_squared,
                    0.0,
                    1.0,
                )
            nearest_x = start.x + segment.x * along
            nearest_y = start.y + segment.y * along
            delta_x = position_x - nearest_x
            delta_y = position_y - nearest_y
            distance_squared = delta_x * delta_x + delta_y * delta_y
            if best is None or distance_squared < best[0]:
                best = (
                    distance_squared,
                    nearest_x,
                    nearest_y,
                    tangent,
                    traversed + along * self._segment_lengths[index],
                    index,
                    -tangent.y * delta_x + tangent.x * delta_y,
                )

        assert best is not None
        (
            distance_squared,
            nearest_x,
            nearest_y,
            tangent,
            progress_distance,
            index,
            signed,
        ) = best
        return TrackProjection(
            point=Vec2(nearest_x, nearest_y),
            tangent=tangent,
            distance=math.sqrt(distance_squared),
            signed_offset=signed,
            progress=(progress_distance / self._total_length) % 1.0,
            segment_index=index,
        )

    def road_kind_at_progress(self, progress: float) -> TerrainKind:
        for sector in self.sectors:
            if sector.contains(progress):
                return sector.kind
        return TerrainKind.ASPHALT

    def terrain_at(self, position: Vec2) -> Terrain:
        projection = self.project(position)
        if projection.distance <= self.track_width * 0.5:
            return terrain(self.road_kind_at_progress(projection.progress))
        return terrain(self.runoff)

    def start_pose(self) -> tuple[Vec2, float]:
        start = self.points[0]
        tangent = self._projection_segments[0][3]
        heading = math.atan2(tangent.y, tangent.x)
        return start, heading

    def point_tangent_at(self, progress: float) -> tuple[Vec2, Vec2]:
        """Interpolate the center line at normalized lap *progress*."""

        target = (progress % 1.0) * self._total_length
        for index, segment_length in enumerate(self._segment_lengths):
            if target <= segment_length:
                start, segment, _, tangent, _ = self._projection_segments[index]
                fraction = 0.0 if segment_length <= 1e-12 else target / segment_length
                return start + segment * fraction, tangent
            target -= segment_length
        start = self.points[0]
        return start, self._projection_segments[0][3]


_CIRCUITS = {
    "harbor_loop": Circuit(
        slug="harbor_loop",
        name="Harbor Loop",
        points=(
            Vec2(118, 160),
            Vec2(285, 86),
            Vec2(570, 92),
            Vec2(710, 205),
            Vec2(684, 430),
            Vec2(525, 590),
            Vec2(250, 612),
            Vec2(88, 500),
            Vec2(70, 295),
        ),
        track_width=72,
        runoff_width=34,
        runoff=TerrainKind.GRASS,
        sectors=(SurfaceSector(0.19, 0.30, TerrainKind.WET_ASPHALT),),
        description="Wide flowing corners with a short wet dockside sector.",
    ),
    "pine_sprint": Circuit(
        slug="pine_sprint",
        name="Pine Sprint",
        points=(
            Vec2(105, 120),
            Vec2(350, 78),
            Vec2(670, 145),
            Vec2(715, 315),
            Vec2(610, 525),
            Vec2(395, 612),
            Vec2(155, 550),
            Vec2(72, 365),
            Vec2(135, 235),
        ),
        track_width=64,
        runoff_width=30,
        runoff=TerrainKind.MUD,
        sectors=(SurfaceSector(0.52, 0.66, TerrainKind.GRAVEL),),
        description="Technical forest course with gravel and muddy runoff.",
    ),
    "desert_switchback": Circuit(
        slug="desert_switchback",
        name="Desert Switchback",
        points=(
            Vec2(90, 105),
            Vec2(690, 105),
            Vec2(718, 218),
            Vec2(430, 238),
            Vec2(675, 330),
            Vec2(705, 565),
            Vec2(90, 575),
            Vec2(73, 455),
            Vec2(360, 435),
            Vec2(105, 350),
            Vec2(72, 220),
        ),
        track_width=58,
        runoff_width=28,
        runoff=TerrainKind.GRAVEL,
        sectors=(
            SurfaceSector(0.31, 0.40, TerrainKind.GRAVEL),
            SurfaceSector(0.73, 0.78, TerrainKind.WET_ASPHALT),
        ),
        description="Long straights joined by tight, low-grip switchbacks.",
    ),
    "alpine_gauntlet": Circuit(
        slug="alpine_gauntlet",
        name="Alpine Gauntlet",
        points=(
            Vec2(377.5, 70),
            Vec2(520, 68),
            Vec2(680, 120),
            Vec2(728, 230),
            Vec2(650, 312),
            Vec2(720, 405),
            Vec2(680, 545),
            Vec2(545, 620),
            Vec2(400, 570),
            Vec2(270, 632),
            Vec2(120, 560),
            Vec2(66, 440),
            Vec2(145, 365),
            Vec2(72, 280),
            Vec2(104, 130),
            Vec2(235, 72),
        ),
        track_width=52,
        runoff_width=26,
        runoff=TerrainKind.SNOW,
        sectors=(
            SurfaceSector(0.13, 0.22, TerrainKind.ICE),
            SurfaceSector(0.56, 0.64, TerrainKind.WET_ASPHALT),
            SurfaceSector(0.82, 0.88, TerrainKind.SNOW),
        ),
        description=(
            "A fast sixteen-point mountain loop with an icy crest, wet valley, "
            "and snowy final sector."
        ),
    ),
    "canyon_maze": Circuit(
        slug="canyon_maze",
        name="Canyon Maze",
        points=(
            Vec2(400, 90),
            Vec2(700, 90),
            Vec2(720, 190),
            Vec2(480, 190),
            Vec2(480, 290),
            Vec2(680, 290),
            Vec2(720, 390),
            Vec2(630, 470),
            Vec2(710, 560),
            Vec2(650, 630),
            Vec2(100, 630),
            Vec2(70, 530),
            Vec2(330, 530),
            Vec2(330, 430),
            Vec2(100, 430),
            Vec2(65, 335),
            Vec2(170, 265),
            Vec2(65, 190),
            Vec2(100, 90),
        ),
        track_width=42,
        runoff_width=22,
        runoff=TerrainKind.SAND,
        sectors=(
            SurfaceSector(0.18, 0.25, TerrainKind.GRAVEL),
            SurfaceSector(0.44, 0.52, TerrainKind.SAND),
            SurfaceSector(0.76, 0.82, TerrainKind.WET_ASPHALT),
        ),
        description=(
            "An eighteen-corner precision course of opposing hairpins, dusty "
            "switchbacks, and a flash-flood crossing."
        ),
    ),
}


def circuit_names() -> tuple[str, ...]:
    return tuple(_CIRCUITS)


def get_circuit(name: str) -> Circuit:
    try:
        return _CIRCUITS[name]
    except KeyError as error:
        choices = ", ".join(circuit_names())
        raise ValueError(f"Unknown circuit {name!r}; choose from {choices}") from error


def all_circuits() -> tuple[Circuit, ...]:
    return tuple(_CIRCUITS.values())
