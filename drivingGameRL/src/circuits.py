"""Analytic closed circuits shared by simulation, sensors, and rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .math2d import Vec2, clamp
from .terrain import Terrain, TerrainKind, terrain


@dataclass(frozen=True, slots=True)
class SurfaceSector:
    """Replace the road surface over a normalized portion of one lap."""

    start: float
    end: float
    kind: TerrainKind

    def contains(self, progress: float) -> bool:
        progress %= 1.0
        if self.start <= self.end:
            return self.start <= progress <= self.end
        return progress >= self.start or progress <= self.end


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
        if any(length <= 1e-9 for length in self.segment_lengths):
            raise ValueError("Consecutive circuit points must be distinct")

    @property
    def segment_lengths(self) -> tuple[float, ...]:
        return tuple(
            (self.points[(index + 1) % len(self.points)] - point).length()
            for index, point in enumerate(self.points)
        )

    @property
    def length(self) -> float:
        return sum(self.segment_lengths)

    @property
    def collision_radius(self) -> float:
        return self.track_width * 0.5 + self.runoff_width

    def project(self, position: Vec2) -> TrackProjection:
        if not all(math.isfinite(value) for value in (position.x, position.y)):
            raise ValueError("Projected position must be finite")
        lengths = self.segment_lengths
        total = sum(lengths)
        best: tuple[float, Vec2, Vec2, float, int, float] | None = None
        traversed = 0.0

        for index, start in enumerate(self.points):
            end = self.points[(index + 1) % len(self.points)]
            segment = end - start
            length_squared = segment.length_squared()
            along = 0.0
            if length_squared > 1e-12:
                along = clamp(
                    (position - start).dot(segment) / length_squared, 0.0, 1.0
                )
            nearest = start + segment * along
            delta = position - nearest
            distance_squared = delta.length_squared()
            tangent = segment.normalized()
            if best is None or distance_squared < best[0]:
                progress_distance = traversed + along * lengths[index]
                best = (
                    distance_squared,
                    nearest,
                    tangent,
                    progress_distance,
                    index,
                    tangent.perpendicular().dot(delta),
                )
            traversed += lengths[index]

        assert best is not None
        distance_squared, point, tangent, progress_distance, index, signed = best
        return TrackProjection(
            point=point,
            tangent=tangent,
            distance=math.sqrt(distance_squared),
            signed_offset=signed,
            progress=(progress_distance / total) % 1.0,
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
        tangent = (self.points[1] - start).normalized()
        heading = math.atan2(tangent.y, tangent.x)
        return start, heading

    def point_tangent_at(self, progress: float) -> tuple[Vec2, Vec2]:
        """Interpolate the center line at normalized lap *progress*."""

        target = (progress % 1.0) * self.length
        for index, segment_length in enumerate(self.segment_lengths):
            if target <= segment_length:
                start = self.points[index]
                end = self.points[(index + 1) % len(self.points)]
                tangent = (end - start).normalized()
                fraction = 0.0 if segment_length <= 1e-12 else target / segment_length
                return start + (end - start) * fraction, tangent
            target -= segment_length
        start = self.points[0]
        return start, (self.points[1] - start).normalized()


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
