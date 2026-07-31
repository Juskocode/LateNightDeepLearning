"""Pure, deterministic projectile rules for the Pacman ghost abilities.

The game loop owns rendering and applies the effects returned here.  Keeping
projectiles in tile coordinates makes the rules usable by both the pygame game
and the reinforcement-learning environment without introducing frame-rate
dependent behaviour.

Typical integration::

    weapons = GhostProjectileSystem(game.rng)

    # Once per frame, before asking released ghosts to shoot:
    is_walkable = lambda cell: game._cell(*cell) != "#"
    events = weapons.update(dt, game.player.grid, is_walkable)

    # When a ghost has a clear cardinal line of sight:
    weapons.try_fire_at_target(
        ghost.name, game.level, ghost.grid, game.player.grid,
        is_walkable,
    )

``GhostProjectileSystem`` performs exactly two RNG calls when a run starts:
one for Blinky and one for Inky, in that order.  Each ability independently
has a 20% chance to unlock early.  Otherwise both become available at level 3.
Level/round resets deliberately do not reroll these per-run unlocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Protocol, Sequence, TypeAlias


GridCell: TypeAlias = tuple[int, int]
GridVector: TypeAlias = tuple[int, int]
CellIsWalkable: TypeAlias = Callable[[GridCell], bool]
NextCell: TypeAlias = Callable[[GridCell, GridVector], GridCell]

CARDINAL_DIRECTIONS: tuple[GridVector, ...] = (
    (0, -1),  # up
    (-1, 0),  # left
    (0, 1),  # down
    (1, 0),  # right
)

# Collision callbacks are sampled at most this far apart.  The combined
# Pacman/projectile hit radius is larger than this, so even deliberately large
# simulation steps cannot tunnel through an interpolated player position.
MAX_COLLISION_STEP_TILES = 0.20
_MOVEMENT_EPSILON = 1e-9


class RandomSource(Protocol):
    """The small part of :class:`random.Random` used by this module."""

    def random(self) -> float: ...


class ProjectileKind(str, Enum):
    FIREBALL = "fireball"
    FREEZE_BALL = "freeze_ball"


class DespawnReason(str, Enum):
    HIT_PACMAN = "hit_pacman"
    WALL = "wall"
    RANGE = "range"
    COLLISION = "collision"


@dataclass(frozen=True)
class ProjectileSpec:
    """Immutable balance and effect data for one ghost weapon."""

    owner: str
    kind: ProjectileKind
    range_tiles: int
    speed_tiles_per_second: float
    cooldown_seconds: float
    unlock_level: int = 3
    damage: int = 0
    slow_fraction: float = 0.0
    slow_duration_seconds: float = 0.0
    color: tuple[int, int, int] = (255, 255, 255)
    glow_color: tuple[int, int, int] = (255, 255, 255)
    radius_tiles: float = 0.17

    def __post_init__(self) -> None:
        if not self.owner.strip():
            raise ValueError("projectile owner cannot be empty")
        if self.range_tiles < 1:
            raise ValueError("range_tiles must be positive")
        if (
            not math.isfinite(self.speed_tiles_per_second)
            or self.speed_tiles_per_second <= 0
        ):
            raise ValueError("speed_tiles_per_second must be finite and positive")
        if not math.isfinite(self.cooldown_seconds) or self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be finite and non-negative")
        if self.unlock_level < 1:
            raise ValueError("unlock_level must be positive")
        if self.damage < 0:
            raise ValueError("damage cannot be negative")
        if not 0.0 <= self.slow_fraction < 1.0:
            raise ValueError("slow_fraction must be in [0, 1)")
        if (
            not math.isfinite(self.slow_duration_seconds)
            or self.slow_duration_seconds < 0
        ):
            raise ValueError("slow_duration_seconds must be finite and non-negative")
        if not 0.0 < self.radius_tiles <= 0.5:
            raise ValueError("radius_tiles must be in (0, 0.5]")

    @property
    def speed_multiplier_on_hit(self) -> float:
        return 1.0 - self.slow_fraction


FIREBALL_SPEC = ProjectileSpec(
    owner="BLINKY",
    kind=ProjectileKind.FIREBALL,
    range_tiles=5,
    speed_tiles_per_second=7.0,
    cooldown_seconds=4.25,
    damage=1,
    color=(255, 111, 45),
    glow_color=(255, 35, 20),
    radius_tiles=0.15,
)

FREEZE_BALL_SPEC = ProjectileSpec(
    owner="INKY",
    kind=ProjectileKind.FREEZE_BALL,
    range_tiles=15,
    speed_tiles_per_second=5.25,
    cooldown_seconds=5.5,
    slow_fraction=0.15,
    slow_duration_seconds=3.0,
    color=(126, 239, 255),
    glow_color=(36, 128, 255),
    radius_tiles=0.18,
)

DEFAULT_PROJECTILE_SPECS: tuple[ProjectileSpec, ...] = (
    FIREBALL_SPEC,
    FREEZE_BALL_SPEC,
)


@dataclass(frozen=True)
class AbilityUnlockRoll:
    """Observable result of one early-unlock roll."""

    owner: str
    kind: ProjectileKind
    random_value: float
    probability: float
    unlocked_early: bool


@dataclass(frozen=True)
class RunAbilityUnlocks:
    """Per-run ability decisions; retained unchanged across levels."""

    rolls: tuple[AbilityUnlockRoll, ...]

    def is_early(self, kind_or_owner: ProjectileKind | str) -> bool:
        if isinstance(kind_or_owner, ProjectileKind):
            return any(
                roll.kind == kind_or_owner and roll.unlocked_early
                for roll in self.rolls
            )
        owner = str(kind_or_owner).upper()
        return any(roll.owner == owner and roll.unlocked_early for roll in self.rolls)


def roll_run_ability_unlocks(
    rng: RandomSource,
    *,
    specs: Sequence[ProjectileSpec] = DEFAULT_PROJECTILE_SPECS,
    early_unlock_chance: float = 0.20,
) -> RunAbilityUnlocks:
    """Roll each level-3 ability once for a possible level-1 unlock.

    Specs are evaluated in the supplied order.  This stable RNG consumption is
    useful for seeded games, recordings, and reproducible RL evaluation.
    """

    if not math.isfinite(early_unlock_chance) or not 0.0 <= early_unlock_chance <= 1.0:
        raise ValueError("early_unlock_chance must be in [0, 1]")

    rolls: list[AbilityUnlockRoll] = []
    for spec in specs:
        random_value = float(rng.random())
        if not 0.0 <= random_value < 1.0:
            raise ValueError("rng.random() must return a value in [0, 1)")
        rolls.append(
            AbilityUnlockRoll(
                owner=spec.owner.upper(),
                kind=spec.kind,
                random_value=random_value,
                probability=early_unlock_chance,
                unlocked_early=random_value < early_unlock_chance,
            )
        )
    return RunAbilityUnlocks(tuple(rolls))


@dataclass
class Projectile:
    """A projectile travelling from one tile centre to the next."""

    projectile_id: int
    spec: ProjectileSpec
    cell: GridCell
    direction: GridVector
    progress: float = 0.0
    tiles_travelled: int = 0

    @property
    def owner(self) -> str:
        return self.spec.owner

    @property
    def kind(self) -> ProjectileKind:
        return self.spec.kind

    @property
    def position_tiles(self) -> tuple[float, float]:
        """Continuous tile-space centre, convenient for interpolation/rendering."""

        return (
            self.cell[0] + self.direction[0] * self.progress,
            self.cell[1] + self.direction[1] * self.progress,
        )

    @property
    def remaining_range_tiles(self) -> int:
        return max(0, self.spec.range_tiles - self.tiles_travelled)


@dataclass(frozen=True)
class ProjectileEvent:
    """A terminal projectile event and the gameplay effect it carries."""

    projectile_id: int
    owner: str
    kind: ProjectileKind
    reason: DespawnReason
    cell: GridCell
    position_tiles: tuple[float, float]
    blocked_cell: GridCell | None = None
    damage: int = 0
    slow_fraction: float = 0.0
    slow_duration_seconds: float = 0.0

    @property
    def hit_pacman(self) -> bool:
        return self.reason == DespawnReason.HIT_PACMAN

    @property
    def speed_multiplier(self) -> float:
        return 1.0 - self.slow_fraction


@dataclass
class PacmanSlowState:
    """Small optional helper for applying freeze-hit events to player speed."""

    fraction: float = 0.0
    remaining_seconds: float = 0.0

    @property
    def active(self) -> bool:
        return self.remaining_seconds > 0.0 and self.fraction > 0.0

    @property
    def speed_multiplier(self) -> float:
        return 1.0 - self.fraction if self.active else 1.0

    def apply(self, event: ProjectileEvent) -> bool:
        """Apply a freeze event, refreshing rather than stacking its slowdown."""

        if not event.hit_pacman or event.slow_fraction <= 0.0:
            return False
        self.fraction = max(self.fraction if self.active else 0.0, event.slow_fraction)
        self.remaining_seconds = max(
            self.remaining_seconds, event.slow_duration_seconds
        )
        return True

    def update(self, dt: float) -> None:
        _validate_dt(dt)
        self.remaining_seconds = max(0.0, self.remaining_seconds - dt)
        if self.remaining_seconds == 0.0:
            self.fraction = 0.0

    def clear(self) -> None:
        self.fraction = 0.0
        self.remaining_seconds = 0.0


CollisionTest: TypeAlias = Callable[[Projectile, GridCell], bool]


class GhostProjectileSystem:
    """Owns unlocks, cooldowns, active shots, and deterministic movement."""

    def __init__(
        self,
        rng: RandomSource,
        *,
        specs: Sequence[ProjectileSpec] = DEFAULT_PROJECTILE_SPECS,
        early_unlock_chance: float = 0.20,
    ) -> None:
        if not specs:
            raise ValueError("at least one projectile spec is required")
        owners = [spec.owner.upper() for spec in specs]
        if len(set(owners)) != len(owners):
            raise ValueError("each projectile spec must have a unique owner")

        self.rng = rng
        self.specs = tuple(specs)
        self.early_unlock_chance = float(early_unlock_chance)
        self._spec_by_owner = {spec.owner.upper(): spec for spec in self.specs}
        self._active: list[Projectile] = []
        self._cooldowns = {owner: 0.0 for owner in owners}
        self._next_projectile_id = 1
        self.unlocks = roll_run_ability_unlocks(
            self.rng,
            specs=self.specs,
            early_unlock_chance=self.early_unlock_chance,
        )

    @property
    def active_projectiles(self) -> tuple[Projectile, ...]:
        return tuple(self._active)

    @property
    def cooldowns(self) -> dict[str, float]:
        return dict(self._cooldowns)

    def start_new_run(self) -> RunAbilityUnlocks:
        """Clear combat state and reroll the two early-unlock decisions."""

        self._active.clear()
        self._cooldowns = {owner: 0.0 for owner in self._spec_by_owner}
        self._next_projectile_id = 1
        self.unlocks = roll_run_ability_unlocks(
            self.rng,
            specs=self.specs,
            early_unlock_chance=self.early_unlock_chance,
        )
        return self.unlocks

    def reset_level(self, *, initial_cooldown_seconds: float = 0.0) -> None:
        """Clear shots/cooldowns without rerolling per-run early unlocks."""

        _validate_non_negative_time(
            initial_cooldown_seconds, "initial_cooldown_seconds"
        )
        self._active.clear()
        for owner in self._cooldowns:
            self._cooldowns[owner] = float(initial_cooldown_seconds)

    def clear_projectiles(self) -> None:
        self._active.clear()

    def spec_for(self, owner: str) -> ProjectileSpec | None:
        return self._spec_by_owner.get(str(owner).upper())

    def is_unlocked(self, owner: str, level: int) -> bool:
        spec = self.spec_for(owner)
        if spec is None:
            return False
        return level >= spec.unlock_level or self.unlocks.is_early(spec.kind)

    def seconds_until_ready(self, owner: str) -> float:
        return self._cooldowns.get(str(owner).upper(), math.inf)

    def can_fire(self, owner: str, level: int, *, suppressed: bool = False) -> bool:
        name = str(owner).upper()
        return (
            not suppressed
            and self.is_unlocked(name, level)
            and self._cooldowns.get(name, math.inf) <= 0.0
        )

    def update_cooldowns(self, dt: float) -> None:
        _validate_dt(dt)
        for owner, remaining in self._cooldowns.items():
            self._cooldowns[owner] = max(0.0, remaining - dt)

    def aim_direction(
        self,
        origin: GridCell,
        target: GridCell,
        max_range_tiles: int,
        is_walkable: CellIsWalkable,
        *,
        next_cell: NextCell | None = None,
        directions: Iterable[GridVector] = CARDINAL_DIRECTIONS,
    ) -> GridVector | None:
        """Return a clear cardinal ray to ``target`` or ``None``.

        ``next_cell`` can implement tunnel wrapping.  Revisited cells terminate
        a ray, so a wrapping topology cannot loop forever.
        """

        if max_range_tiles < 1:
            return None
        step_cell = next_cell or _default_next_cell
        for direction_value in directions:
            direction = _coerce_direction(direction_value)
            cell = origin
            visited = {origin}
            for _ in range(max_range_tiles):
                candidate = _coerce_cell(step_cell(cell, direction))
                if candidate in visited or not is_walkable(candidate):
                    break
                if candidate == target:
                    return direction
                visited.add(candidate)
                cell = candidate
        return None

    def try_fire(
        self,
        owner: str,
        level: int,
        origin: GridCell,
        direction: GridVector | object,
        is_walkable: CellIsWalkable,
        *,
        next_cell: NextCell | None = None,
        suppressed: bool = False,
    ) -> Projectile | None:
        """Spawn a shot if its ability/cooldown and first corridor tile allow it."""

        name = str(owner).upper()
        spec = self.spec_for(name)
        if spec is None or not self.can_fire(name, level, suppressed=suppressed):
            return None
        vector = _coerce_direction(direction)
        cell = _coerce_cell(origin)
        step_cell = next_cell or _default_next_cell
        first_cell = _coerce_cell(step_cell(cell, vector))
        if not is_walkable(first_cell):
            return None

        projectile = Projectile(
            projectile_id=self._next_projectile_id,
            spec=spec,
            cell=cell,
            direction=vector,
        )
        self._next_projectile_id += 1
        self._active.append(projectile)
        self._cooldowns[name] = spec.cooldown_seconds
        return projectile

    def try_fire_at_target(
        self,
        owner: str,
        level: int,
        origin: GridCell,
        target: GridCell,
        is_walkable: CellIsWalkable,
        *,
        next_cell: NextCell | None = None,
        suppressed: bool = False,
    ) -> Projectile | None:
        """Fire only when the target is visible along a valid corridor ray."""

        name = str(owner).upper()
        spec = self.spec_for(name)
        if spec is None or not self.can_fire(name, level, suppressed=suppressed):
            return None
        direction = self.aim_direction(
            _coerce_cell(origin),
            _coerce_cell(target),
            spec.range_tiles,
            is_walkable,
            next_cell=next_cell,
        )
        if direction is None:
            return None
        return self.try_fire(
            name,
            level,
            origin,
            direction,
            is_walkable,
            next_cell=next_cell,
            suppressed=suppressed,
        )

    def update(
        self,
        dt: float,
        pacman_cell: GridCell | None,
        is_walkable: CellIsWalkable,
        *,
        next_cell: NextCell | None = None,
        pacman_collision_test: CollisionTest | None = None,
        collision_test: CollisionTest | None = None,
    ) -> tuple[ProjectileEvent, ...]:
        """Advance cooldowns and shots, returning wall/range/collision events.

        Projectiles are resolved in spawn order. ``pacman_cell`` preserves the
        grid-only API, while ``pacman_collision_test`` lets an interpolated game
        use the projectile's continuous position. A Pacman hit takes precedence
        over a custom collision, including at the final tile of a shot's range.
        """

        _validate_dt(dt)
        self.update_cooldowns(dt)
        target = _coerce_cell(pacman_cell) if pacman_cell is not None else None
        step_cell = next_cell or _default_next_cell
        survivors: list[Projectile] = []
        events: list[ProjectileEvent] = []

        for projectile in self._active:
            terminal_event: ProjectileEvent | None = None

            # Pacman may have entered a projectile's occupied tile since the
            # previous update.  Ignore the owner's spawn tile at distance zero.
            departed_spawn = projectile.tiles_travelled > 0 or projectile.progress > 0
            if departed_spawn and target == projectile.cell:
                terminal_event = _event_for(projectile, DespawnReason.HIT_PACMAN)
            elif (
                departed_spawn
                and pacman_collision_test is not None
                and pacman_collision_test(projectile, projectile.cell)
            ):
                terminal_event = _event_for(projectile, DespawnReason.HIT_PACMAN)
            elif (
                departed_spawn
                and collision_test is not None
                and collision_test(projectile, projectile.cell)
            ):
                terminal_event = _event_for(projectile, DespawnReason.COLLISION)

            distance_remaining = projectile.spec.speed_tiles_per_second * dt
            while terminal_event is None and distance_remaining > _MOVEMENT_EPSILON:
                candidate = _coerce_cell(
                    step_cell(projectile.cell, projectile.direction)
                )
                candidate_is_walkable = is_walkable(candidate)
                destination_progress = 1.0 if candidate_is_walkable else 0.5
                distance_to_destination = max(
                    0.0,
                    destination_progress - projectile.progress,
                )
                collision_step = (
                    MAX_COLLISION_STEP_TILES
                    if pacman_collision_test is not None
                    else math.inf
                )
                movement = min(
                    distance_remaining,
                    distance_to_destination,
                    collision_step,
                )
                projectile.progress += movement
                distance_remaining -= movement

                # Sample the complete travelled segment rather than only tile
                # centres. This prevents a fast shot from crossing Pacman
                # between two collision checks when a caller supplies a large
                # dt (for example, headless training or a paused debugger).
                if (
                    pacman_collision_test is not None
                    and (projectile.tiles_travelled > 0 or projectile.progress > 0)
                    and pacman_collision_test(projectile, projectile.cell)
                ):
                    terminal_event = _event_for(
                        projectile,
                        DespawnReason.HIT_PACMAN,
                    )
                    continue

                reached_destination = (
                    distance_to_destination <= _MOVEMENT_EPSILON
                    or projectile.progress >= destination_progress - _MOVEMENT_EPSILON
                )
                if not reached_destination:
                    continue

                if not candidate_is_walkable:
                    terminal_event = _event_for(
                        projectile,
                        DespawnReason.WALL,
                        blocked_cell=candidate,
                    )
                    continue

                projectile.cell = candidate
                projectile.progress = 0.0
                projectile.tiles_travelled += 1

                if target == projectile.cell:
                    terminal_event = _event_for(projectile, DespawnReason.HIT_PACMAN)
                elif collision_test is not None and collision_test(
                    projectile, projectile.cell
                ):
                    terminal_event = _event_for(projectile, DespawnReason.COLLISION)
                elif projectile.tiles_travelled >= projectile.spec.range_tiles:
                    terminal_event = _event_for(projectile, DespawnReason.RANGE)

            if terminal_event is None:
                survivors.append(projectile)
            else:
                events.append(terminal_event)

        self._active = survivors
        return tuple(events)


def _event_for(
    projectile: Projectile,
    reason: DespawnReason,
    *,
    blocked_cell: GridCell | None = None,
) -> ProjectileEvent:
    hit = reason == DespawnReason.HIT_PACMAN
    return ProjectileEvent(
        projectile_id=projectile.projectile_id,
        owner=projectile.owner,
        kind=projectile.kind,
        reason=reason,
        cell=projectile.cell,
        position_tiles=projectile.position_tiles,
        blocked_cell=blocked_cell,
        damage=projectile.spec.damage if hit else 0,
        slow_fraction=projectile.spec.slow_fraction if hit else 0.0,
        slow_duration_seconds=projectile.spec.slow_duration_seconds if hit else 0.0,
    )


def _default_next_cell(cell: GridCell, direction: GridVector) -> GridCell:
    return cell[0] + direction[0], cell[1] + direction[1]


def _coerce_direction(value: GridVector | object) -> GridVector:
    candidate = getattr(value, "vector", value)
    try:
        dx, dy = candidate  # type: ignore[misc]
    except (TypeError, ValueError) as error:
        raise ValueError("direction must be a cardinal (dx, dy) pair") from error
    direction = (int(dx), int(dy))
    if direction not in CARDINAL_DIRECTIONS or direction != (dx, dy):
        raise ValueError("direction must be one of (0,-1), (-1,0), (0,1), (1,0)")
    return direction


def _coerce_cell(value: object) -> GridCell:
    try:
        x, y = value  # type: ignore[misc]
    except (TypeError, ValueError) as error:
        raise ValueError("cell must be an (x, y) pair") from error
    cell = (int(x), int(y))
    if cell != (x, y):
        raise ValueError("cell coordinates must be integers")
    return cell


def _validate_dt(dt: float) -> None:
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")


def _validate_non_negative_time(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = (
    "AbilityUnlockRoll",
    "CARDINAL_DIRECTIONS",
    "DEFAULT_PROJECTILE_SPECS",
    "DespawnReason",
    "FIREBALL_SPEC",
    "FREEZE_BALL_SPEC",
    "GhostProjectileSystem",
    "PacmanSlowState",
    "Projectile",
    "ProjectileEvent",
    "ProjectileKind",
    "ProjectileSpec",
    "RunAbilityUnlocks",
    "roll_run_ability_unlocks",
)
