"""Command-line entry point for the top-down driving laboratory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .src.circuits import all_circuits, circuit_names
from .src.vehicle import CarBuild, MAX_UPGRADE_LEVEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play or capture the deterministic 2D driving laboratory."
    )
    parser.add_argument("--circuit", choices=circuit_names(), default="harbor_loop")
    parser.add_argument(
        "--list-circuits", action="store_true", help="List circuits and exit"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run a deterministic autopilot without a window",
    )
    parser.add_argument(
        "--steps", type=int, help="Stop after this many fixed simulation steps"
    )
    parser.add_argument(
        "--screenshot", type=Path, help="Save the final rendered frame as PNG"
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Particle and environment seed"
    )
    parser.add_argument("--fps", type=int, default=60, help="Display frame limit")
    parser.add_argument(
        "--motor", type=int, choices=range(MAX_UPGRADE_LEVEL + 1), default=0
    )
    parser.add_argument(
        "--wheels", type=int, choices=range(MAX_UPGRADE_LEVEL + 1), default=0
    )
    parser.add_argument(
        "--suspension", type=int, choices=range(MAX_UPGRADE_LEVEL + 1), default=0
    )
    parser.add_argument(
        "--grip", type=int, choices=range(MAX_UPGRADE_LEVEL + 1), default=0
    )
    parser.add_argument(
        "--car-sprite", type=Path, help="Optional transparent top-down car image"
    )
    parser.add_argument(
        "--no-sensors", action="store_true", help="Hide the five live track rays"
    )
    parser.add_argument(
        "--no-ghost", action="store_true", help="Hide the in-session best-lap ghost"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_circuits:
        for circuit in all_circuits():
            print(f"{circuit.slug:20} {circuit.name}: {circuit.description}")
        return 0
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    capture_mode = args.headless or args.screenshot is not None
    if capture_mode:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    # Importing the Pygame view after configuring its video driver makes the
    # module safe for CI and remote training machines.
    from .src.game import DrivingGame

    game = DrivingGame(
        args.circuit,
        build=CarBuild(args.motor, args.wheels, args.suspension, args.grip),
        seed=args.seed,
        render=not capture_mode,
        car_sprite_path=args.car_sprite,
    )
    game.show_sensors = not args.no_sensors
    game.show_ghost = not args.no_ghost
    try:
        steps = args.steps
        if capture_mode and steps is None:
            steps = 240
        game.run(fps=args.fps, max_steps=steps, autopilot=capture_mode)
        if args.screenshot:
            output = game.save_screenshot(args.screenshot)
            print(f"Screenshot saved to {output}")
        snapshot = game.telemetry()
        best_lap = snapshot["best_lap_time"]
        best_text = "--" if best_lap is None else f"{float(best_lap):.3f}s"
        print(
            f"Circuit {snapshot['circuit']} | progress {snapshot['progress'] * 100:.1f}% | "
            f"lap {snapshot['laps'] + 1} | speed {snapshot['speed']:.1f} | "
            f"best {best_text} | collisions {snapshot['collisions']}"
        )
    finally:
        game.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
