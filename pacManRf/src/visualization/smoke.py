"""Standalone surface smoke check for all Pacman observatory tabs.

Run with ``python -m pacManRf.src.visualization.smoke``.  Passing an output
directory writes one PNG per tab; without it, verification is in-memory only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import pygame

from pacManRf.src.game.pacman_env import OBSERVATION_LABELS

from .observatory import ObservatoryTab, PacmanObservatory


def verify_surfaces(output_directory: str | Path | None = None) -> dict[str, tuple[int, int]]:
    """Render all tabs with a known fixture and assert they contain pixels."""

    pygame.init()
    canvas_size = (1120, 720)
    game = pygame.Surface((560, 650))
    game.fill((2, 5, 14))
    pygame.draw.rect(game, (31, 74, 196), game.get_rect().inflate(-30, -30), 5, border_radius=12)
    pygame.draw.circle(game, (255, 215, 59), (280, 325), 26)

    telemetry = {
        "algorithm": "double_dqn",
        "score": 740,
        "episode": 18,
        "reward": 10.0,
        "loss": 0.024,
        "epsilon": 0.17,
        "decisions_per_second": 30,
        "speed_label": "FAST",
        "speed_preset_index": 3,
        "speed_preset_count": 7,
        "replay_size": 384,
        "replay_capacity": 10_000,
        "action_labels": ["LEFT", "RIGHT", "UP", "DOWN"],
        "online_q_values": [-0.8, 1.5, 0.4, -0.1],
        "target_q_values": [-0.6, 1.2, 0.55, -0.2],
        "chosen_action": 1,
        "recent_rewards": [0.0, 0.0, 10.0, -1.0, 0.0, 10.0],
        "recent_actions": [2, 2, 1, 0, 3, 1],
        "recent_dones": [False, False, False, True, False, False],
        "observation_labels": list(OBSERVATION_LABELS),
        "observation": dict(
            zip(
                OBSERVATION_LABELS,
                (
                    1.0, 1.0, 0.0, 1.0,
                    0.92, 0.48, 0.25, 0.71,
                    0.0, 0.32, 0.0, 0.12,
                    0.0, 0.76, 0.18, 0.0,
                    0.0, 0.0, 0.58, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.46, 0.72, 0.35, 0.61,
                    0.75, 0.2, 1.0, 0.5,
                ),
            )
        ),
        "network": {
            "weight_layout": "out_in",
            "layers": [
                {"name": "OBS", "activations": [0.0, 1.0, -0.5, 0.25]},
                {"name": "HIDDEN", "activations": [0.3, -0.7, 1.1, 0.0, 0.6]},
                {"name": "Q", "activations": [-0.8, 1.5, 0.4, -0.1]},
            ],
            "weights": [
                [
                    [0.2, -0.1, 0.4, 0.0],
                    [-0.5, 0.6, 0.1, 0.3],
                    [0.9, -0.2, 0.1, -0.4],
                    [0.0, 0.2, -0.3, 0.5],
                    [0.3, 0.1, 0.7, -0.2],
                ],
                [
                    [0.2, -0.3, 0.1, 0.4, -0.2],
                    [0.8, 0.1, -0.5, 0.3, 0.6],
                    [-0.4, 0.2, 0.7, -0.1, 0.3],
                    [0.1, -0.6, 0.2, 0.5, -0.3],
                ],
            ],
        },
    }
    history = {
        "rewards": [-10, -2, 4, 8, 3, 12, 10],
        "losses": [0.8, 0.5, 0.31, 0.18, 0.09, 0.04, 0.024],
        "scores": [0, 1, 1, 3, 2, 5, 7],
        "epsilons": [1.0, 0.82, 0.64, 0.51, 0.35, 0.24, 0.17],
    }
    ui = PacmanObservatory()
    output = Path(output_directory) if output_directory is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    checks: dict[str, tuple[int, int]] = {}
    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        for tab in ObservatoryTab:
            canvas = pygame.Surface(canvas_size)
            ui.set_tab(tab)
            ui.render(canvas, telemetry, history=history, game_surface=game)
            path = (output or scratch) / f"pacman-observatory-{tab.value.lower()}.png"
            pygame.image.save(canvas, path)
            if not path.exists() or path.stat().st_size < 1_000:
                raise RuntimeError(f"{tab.value} tab did not produce a non-empty render")
            checks[tab.value] = canvas.get_size()
    pygame.quit()
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", help="optional directory for rendered PNG files")
    arguments = parser.parse_args()
    checks = verify_surfaces(arguments.output)
    summary = ", ".join(f"{name}={size[0]}x{size[1]}" for name, size in checks.items())
    print(f"Pacman observatory surface smoke passed: {summary}")


if __name__ == "__main__":
    main()
