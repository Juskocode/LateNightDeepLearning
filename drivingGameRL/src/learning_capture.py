"""Deterministic GIF capture from the real Driving Lab learning runtime.

The capture intentionally receives an already configured
:class:`~drivingGameRL.src.learning_runtime.DrivingLearningSession`. This keeps
documentation images honest: every metric, replay sample, activation, and race
position comes from the same runtime used by the interactive learning mode.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pygame

from .learning_runtime import ChampionRace, DrivingLearningSession
from .learning_visualization import (
    DrivingLearningVisualization,
    LEARNING_WINDOW_SIZE,
)
from .vehicle import DriverControls


TRAINING_TABS = ("OVERVIEW", "NETWORK", "MEMORY")
MAX_FRAMES_PER_TAB = 8
MAX_RACE_FRAMES = 16
MAX_TRAINING_STEPS_PER_FRAME = 120
RACE_STEPS_PER_CAPTURED_FRAME = 3


def _bounded_integer(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in the [{minimum}, {maximum}] interval")
    return value


def _gif_frame(surface: pygame.Surface, colors: int) -> Image.Image:
    """Copy one Pygame surface into an optimized, deterministic GIF frame."""

    if surface.get_size() != LEARNING_WINDOW_SIZE:
        raise ValueError(f"capture surface must be {LEARNING_WINDOW_SIZE}")
    rgb = Image.frombytes(
        "RGB",
        surface.get_size(),
        pygame.image.tobytes(surface, "RGB"),
    )
    return rgb.quantize(
        colors=colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )


def capture_learning_gif(
    path: str | Path,
    session: DrivingLearningSession,
    *,
    frames_per_tab: int = 3,
    training_steps_per_frame: int = 2,
    race_frames: int = 8,
    duration_ms: int = 120,
    palette_colors: int = 96,
) -> Path:
    """Capture training observatory tabs followed by a champion race.

    Training advances by exactly ``3 * frames_per_tab *
    training_steps_per_frame`` fixed environment steps. The race segment then
    uses private race environments and a frozen champion clone, so it cannot add
    replay transitions, train a network, or alter the session environment.

    The frame bounds are deliberate: full-size 1400 x 760 palette images remain
    useful in project documentation without allowing an accidental unbounded
    in-memory capture.
    """

    if not isinstance(session, DrivingLearningSession):
        raise TypeError("session must be a DrivingLearningSession")
    frames_per_tab = _bounded_integer(
        "frames_per_tab",
        frames_per_tab,
        minimum=1,
        maximum=MAX_FRAMES_PER_TAB,
    )
    training_steps_per_frame = _bounded_integer(
        "training_steps_per_frame",
        training_steps_per_frame,
        minimum=1,
        maximum=MAX_TRAINING_STEPS_PER_FRAME,
    )
    race_frames = _bounded_integer(
        "race_frames",
        race_frames,
        minimum=1,
        maximum=MAX_RACE_FRAMES,
    )
    duration_ms = _bounded_integer(
        "duration_ms",
        duration_ms,
        minimum=40,
        maximum=2_000,
    )
    palette_colors = _bounded_integer(
        "palette_colors",
        palette_colors,
        minimum=16,
        maximum=256,
    )

    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".gif":
        raise ValueError("learning capture path must use the .gif suffix")
    output.parent.mkdir(parents=True, exist_ok=True)

    # No display mode is created. Font initialization and software Surfaces are
    # sufficient for both local use and SDL's dummy video driver in CI.
    pygame.font.init()
    visualization = DrivingLearningVisualization(session.env, session.telemetry())
    images: list[Image.Image] = []

    for tab in TRAINING_TABS:
        visualization.set_tab(tab)
        for _ in range(frames_per_tab):
            for _ in range(training_steps_per_frame):
                session.step()
            images.append(
                _gif_frame(
                    visualization.draw(session.env, session.telemetry()),
                    palette_colors,
                )
            )

    # ChampionRace owns both environments and a cloned policy. Human controls are
    # fixed only to make the documentation animation reproducible; they do not
    # stand in for telemetry or modify the training runtime.
    race = ChampionRace(session)
    human_controls = DriverControls(throttle=1.0, steering=-0.18)
    for _ in range(race_frames):
        for _ in range(RACE_STEPS_PER_CAPTURED_FRAME):
            if not race.finished:
                race.step(human_controls)
        race_telemetry = session.telemetry()
        race_telemetry.update(race.telemetry())
        images.append(
            _gif_frame(
                visualization.draw_race(
                    race.human_env,
                    race.champion_env,
                    race_telemetry,
                ),
                palette_colors,
            )
        )

    images[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=1,
    )
    return output


__all__ = (
    "MAX_FRAMES_PER_TAB",
    "MAX_RACE_FRAMES",
    "MAX_TRAINING_STEPS_PER_FRAME",
    "RACE_STEPS_PER_CAPTURED_FRAME",
    "TRAINING_TABS",
    "capture_learning_gif",
)
