"""Command-line entry point for the top-down driving laboratory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

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
        help="Run without a window (autopilot in manual mode)",
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
    learning = parser.add_argument_group("learning observatory")
    learning.add_argument(
        "--learn",
        "--rl",
        action="store_true",
        help="Train a value policy or genetic population instead of manual play",
    )
    learning.add_argument(
        "--algorithm",
        choices=("dqn", "double_dqn", "genetic", "genetic_dqn"),
        default="genetic_dqn",
        help="Driving learning backend (default: genetic_dqn)",
    )
    learning.add_argument(
        "--population",
        type=int,
        default=8,
        help="Number of policies in genetic modes",
    )
    learning.add_argument(
        "--elite-count",
        type=int,
        default=2,
        help="Exact unmutated survivors per generation",
    )
    learning.add_argument(
        "--tournament-size",
        type=int,
        default=2,
        help="Candidates sampled for parent selection",
    )
    learning.add_argument(
        "--evaluation-steps",
        type=int,
        default=900,
        help="Fixed simulation-step budget per policy",
    )
    learning.add_argument(
        "--generations",
        type=int,
        help="Stop after this many completed generations/episodes",
    )
    learning.add_argument(
        "--learning-speed",
        type=int,
        choices=(1, 4, 16, 64, 256),
        default=16,
        metavar="STEPS_PER_FRAME",
        help="Initial training speed: 1, 4, 16, 64, or 256 (MAX)",
    )
    learning.add_argument(
        "--crossover",
        choices=("uniform", "blend"),
        default="uniform",
        help="Genetic crossover operator",
    )
    learning.add_argument("--crossover-rate", type=float, default=0.65)
    learning.add_argument("--blend-alpha", type=float, default=0.20)
    learning.add_argument("--mutation-rate", type=float, default=0.08)
    learning.add_argument("--mutation-std", type=float, default=0.055)
    learning.add_argument(
        "--checkpoint",
        type=Path,
        help="Load this checkpoint when present and save it on clean exit",
    )
    learning.add_argument(
        "--fresh",
        action="store_true",
        help="Do not load an existing --checkpoint",
    )
    learning.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save --checkpoint on exit (S still saves interactively)",
    )
    learning.add_argument(
        "--gif",
        type=Path,
        help="Capture real Overview, Network, Memory, and champion-race frames",
    )
    learning.add_argument(
        "--population-cars",
        "--show-population-cars",
        action="store_true",
        help="Start with isolated same-generation rollout cars visible",
    )
    return parser


def _run_learning(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.population < 2:
        parser.error("--population must be at least 2")
    if not 1 <= args.elite_count < args.population:
        parser.error("--elite-count must be in [1, population)")
    if not 1 <= args.tournament_size <= args.population:
        parser.error("--tournament-size must be in [1, population]")
    if args.evaluation_steps <= 0:
        parser.error("--evaluation-steps must be positive")
    if args.generations is not None and args.generations <= 0:
        parser.error("--generations must be positive")

    from .src.learning_game import DrivingLearningGame
    from .src.learning_runtime import DrivingLearningSession, LearningRuntimeConfig

    runtime = LearningRuntimeConfig(
        algorithm=args.algorithm,
        circuit=args.circuit,
        seed=args.seed,
        evaluation_steps=args.evaluation_steps,
        population_size=args.population,
        elite_count=args.elite_count,
        tournament_size=args.tournament_size,
        crossover=args.crossover,
        crossover_rate=args.crossover_rate,
        blend_alpha=args.blend_alpha,
        mutation_rate=args.mutation_rate,
        mutation_std=args.mutation_std,
    )
    session = DrivingLearningSession(
        runtime,
        build=CarBuild(args.motor, args.wheels, args.suspension, args.grip),
    )
    if args.checkpoint and args.checkpoint.expanduser().is_file() and not args.fresh:
        session.load(args.checkpoint)
        print(f"Loaded driving learner from {args.checkpoint.expanduser().resolve()}")

    capture_mode = args.headless or args.screenshot is not None or args.gif is not None
    game = DrivingLearningGame(
        session,
        render=not capture_mode,
        learning_speed=args.learning_speed,
        checkpoint_path=args.checkpoint,
        show_sensor_rays=not args.no_sensors,
        show_population_cars=args.population_cars,
    )
    try:
        steps = args.steps
        generations = args.generations
        if capture_mode and steps is None and generations is None:
            if args.gif is not None:
                steps = 0
            else:
                generations = 1
        game.run(
            fps=args.fps,
            max_training_steps=steps,
            max_generations=generations,
        )
        if args.screenshot:
            output = game.save_screenshot(args.screenshot)
            print(f"Learning screenshot saved to {output}")
        if args.gif:
            from .src.learning_capture import capture_learning_gif

            output = capture_learning_gif(
                args.gif,
                session,
                frames_per_tab=3,
                training_steps_per_frame=60,
                race_frames=12,
                show_sensor_rays=not args.no_sensors,
                show_population_cars=args.population_cars,
            )
            print(f"Learning GIF saved to {output.resolve()}")
        if args.checkpoint and not args.no_save:
            output = session.save(args.checkpoint)
            print(f"Driving learner saved to {output}")
        snapshot = session.telemetry()
        best = snapshot.get("best_fitness")
        best_text = "--" if best is None else f"{float(best):.3f}"
        print(
            f"{args.algorithm} | generation {snapshot.get('generation', 0)} | "
            f"member {int(snapshot.get('member_index') or 0) + 1}/"
            f"{snapshot.get('population_size', 1)} | best fitness {best_text} | "
            f"replay {snapshot.get('replay_size', 0)}"
        )
    finally:
        game.close()
    return 0


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
    if args.gif is not None and not args.learn:
        parser.error("--gif requires --learn (or late-night-driving-rl)")
    capture_mode = (
        args.headless or args.screenshot is not None or args.gif is not None
    )
    if capture_mode:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    if args.learn:
        return _run_learning(args, parser)

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


def learning_main(argv: list[str] | None = None) -> int:
    """Console-script wrapper that enters learning mode by default."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    return main(["--learn", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
