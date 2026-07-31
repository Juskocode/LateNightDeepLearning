"""Reproducible PNG/GIF capture from the real Pacman RL observatory."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pygame

from pacManRf.src.rl_session import WINDOW_SIZE, PacmanRLSession
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


def capture_observatory_png(
    session: PacmanRLSession,
    path: str | Path,
    *,
    tab: str = "GAME",
    prime_steps: int = 80,
) -> Path:
    pygame.init()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _prime(session, prime_steps)
    canvas = pygame.Surface(WINDOW_SIZE)
    ui = PacmanObservatory(initial_tab=tab)
    ui.render(
        canvas,
        session.telemetry(),
        history=session.history_snapshot(),
        game_surface=session.render_game(),
    )
    pygame.image.save(canvas, output)
    return output


def capture_observatory_gif(
    session: PacmanRLSession,
    path: str | Path,
    *,
    frames: int = 60,
    prime_steps: int = 80,
    duration_ms: int = 140,
    output_width: int = 880,
) -> Path:
    """Capture GAME → METRICS → NETWORK using live training values."""
    pygame.init()
    frame_count = max(12, int(frames))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _prime(session, prime_steps)
    canvas = pygame.Surface(WINDOW_SIZE)
    ui = PacmanObservatory(initial_tab="GAME")
    images: list[Image.Image] = []

    for index in range(frame_count):
        session.step()
        section = min(2, index * 3 // frame_count)
        ui.set_tab(("GAME", "METRICS", "NETWORK")[section])
        ui.render(
            canvas,
            session.telemetry(),
            history=session.history_snapshot(),
            game_surface=session.render_game(),
        )
        images.append(_pil_frame(canvas, output_width=output_width))

    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=max(40, int(duration_ms)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return output
