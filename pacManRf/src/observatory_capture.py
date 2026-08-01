"""Reproducible PNG/GIF capture from the real Pacman RL observatory."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pygame

from pacManRf.src.rl_session import (
    RENDER_FPS,
    WINDOW_SIZE,
    DecisionScheduler,
    PacmanRLSession,
    SpeedController,
)
from pacManRf.src.visualization import PacmanObservatory


def _pil_frame(surface: pygame.Surface, *, output_width: int = 880) -> Image.Image:
    image = Image.frombytes("RGB", surface.get_size(), pygame.image.tobytes(surface, "RGB"))
    if output_width and image.width != output_width:
        height = round(image.height * output_width / image.width)
        image = image.resize((output_width, height), Image.Resampling.LANCZOS)
    return image


def _prime(session: PacmanRLSession, steps: int) -> None:
    for _ in range(max(0, int(steps))):
        session.step()


def _telemetry(
    session: PacmanRLSession,
    speed: SpeedController,
    *,
    simulation_fps_actual: float | None = None,
) -> dict:
    data = session.telemetry()
    data.update(speed.telemetry())
    if simulation_fps_actual is not None:
        data["simulation_fps_actual"] = simulation_fps_actual
    return data


def capture_observatory_png(
    session: PacmanRLSession,
    path: str | Path,
    *,
    tab: str = "GAME",
    prime_steps: int = 80,
    speed: int = RENDER_FPS,
) -> Path:
    pygame.init()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _prime(session, prime_steps)
    canvas = pygame.Surface(WINDOW_SIZE)
    ui = PacmanObservatory(initial_tab=tab)
    speed_controller = SpeedController(speed)
    ui.render(
        canvas,
        _telemetry(session, speed_controller),
        history=session.history_snapshot(),
        game_surface=session.render_game(),
    )
    pygame.image.save(canvas, output)
    return output


def capture_observatory_gif(
    session: PacmanRLSession,
    path: str | Path,
    *,
    frames: int = 48,
    prime_steps: int = 80,
    duration_ms: int = 140,
    output_width: int = 800,
    speed: int = RENDER_FPS,
) -> Path:
    """Capture GAME → VISION → METRICS → NETWORK with live values."""
    pygame.init()
    frame_count = max(12, int(frames))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _prime(session, prime_steps)
    canvas = pygame.Surface(WINDOW_SIZE)
    ui = PacmanObservatory(initial_tab="GAME")
    speed_controller = SpeedController(speed)
    scheduler = DecisionScheduler()
    images: list[Image.Image] = []
    captured_seconds = 0.0
    captured_simulation_frames = 0

    for index in range(frame_count):
        frame_seconds = max(40, int(duration_ms)) / 1_000.0
        simulation_frames = scheduler.frames_for_render(
            frame_seconds,
            speed_controller.value,
        )
        for _ in range(simulation_frames):
            session.advance_simulation_frame()
        captured_seconds += frame_seconds
        captured_simulation_frames += simulation_frames
        section = min(3, index * 4 // frame_count)
        ui.set_tab(("GAME", "VISION", "METRICS", "NETWORK")[section])
        ui.render(
            canvas,
            _telemetry(
                session,
                speed_controller,
                simulation_fps_actual=(
                    captured_simulation_frames / captured_seconds
                ),
            ),
            history=session.history_snapshot(),
            game_surface=session.render_game(),
        )
        images.append(
            _pil_frame(canvas, output_width=output_width).quantize(
                colors=96,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
        )

    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=max(40, int(duration_ms)),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return output
