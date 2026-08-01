"""Tiny dependency-free vector helpers used by the driving physics."""

from __future__ import annotations

from dataclasses import dataclass
import math


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_angle(angle: float) -> float:
    """Return *angle* in the ``[-pi, pi)`` interval."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vec2":
        return Vec2(self.x / scalar, self.y / scalar)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def length_squared(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self, fallback: "Vec2 | None" = None) -> "Vec2":
        magnitude = self.length()
        if magnitude <= 1e-12:
            return fallback or Vec2(1.0, 0.0)
        return self / magnitude

    def perpendicular(self) -> "Vec2":
        return Vec2(-self.y, self.x)

    @classmethod
    def from_angle(cls, angle: float) -> "Vec2":
        return cls(math.cos(angle), math.sin(angle))


ZERO = Vec2()
