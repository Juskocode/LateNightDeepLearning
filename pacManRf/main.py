"""Play Pacman or train its observable DQN/Double-DQN agent."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Pacman arcade game and RL observatory")
    parser.add_argument("--rl", action="store_true", help="Run the reinforcement-learning agent")
    parser.add_argument("--algorithm", choices=("dqn", "double_dqn"), default="double_dqn")
    parser.add_argument("--headless", action="store_true", help="Train without opening a window")
    parser.add_argument("--games", type=int, default=0, help="Headless episodes; 0 runs continuously")
    parser.add_argument("--steps", type=int, help="Optional maximum headless decisions")
    parser.add_argument("--speed", type=int, default=30, help="Visual decisions per second")
    parser.add_argument(
        "--tab",
        choices=("game", "vision", "metrics", "network"),
        default="game",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing Pacman checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="Custom Pacman RL checkpoint")
    parser.add_argument("--eval", action="store_true", help="Disable exploration and learning")
    parser.add_argument("--save-replay", action="store_true", help="Include replay memory in checkpoints")
    parser.add_argument("--no-save", action="store_true", help="Do not save the RL checkpoint on exit")
    parser.add_argument("--screenshot", type=Path, help="Save the current game/observatory PNG and exit")
    parser.add_argument("--gif", type=Path, help="Capture a live RL observatory GIF and exit")
    parser.add_argument("--gif-frames", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _run_manual(args) -> None:
    import pygame
    from pacManRf.src.game.pacmanGame import PacmanGame

    game = PacmanGame(render=not (args.headless or args.screenshot))
    if args.screenshot:
        game.phase = type(game.phase).ACTIVE
        game.frightened_timer = 4.2
        game.score = 460
        game.save_screenshot(args.screenshot)
        pygame.quit()
        return
    while game.running:
        game.play_step()
    pygame.quit()


def _run_rl(args) -> None:
    from pacManRf.src.observatory_capture import (
        capture_observatory_gif,
        capture_observatory_png,
    )
    from pacManRf.src.rl_session import (
        PacmanRLSession,
        SessionConfig,
        run_headless_session,
        run_visual_session,
    )

    session = PacmanRLSession(
        SessionConfig(
            algorithm=args.algorithm,
            seed=args.seed,
            training=not args.eval,
            fresh=args.fresh,
            checkpoint=args.checkpoint,
            save_replay=args.save_replay,
        )
    )
    if args.gif:
        output = capture_observatory_gif(session, args.gif, frames=args.gif_frames)
        print(f"Saved Pacman RL GIF to {output}")
        session.close()
        return
    if args.screenshot:
        output = capture_observatory_png(session, args.screenshot, tab=args.tab)
        print(f"Saved Pacman RL screenshot to {output}")
        session.close()
        return
    if args.headless:
        try:
            run_headless_session(
                session,
                episodes=args.games,
                max_steps=args.steps,
                save_on_exit=not args.no_save and not args.eval,
            )
        except KeyboardInterrupt:
            print("Pacman RL training interrupted")
        return
    run_visual_session(
        session,
        speed=args.speed,
        initial_tab=args.tab,
        save_on_exit=not args.no_save and not args.eval,
    )


def main():
    args = parse_args()
    if args.headless or args.screenshot or args.gif:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if args.rl or args.gif:
        _run_rl(args)
    else:
        _run_manual(args)


def rl_main():
    """Console-script entry point that selects RL mode automatically."""
    if "--rl" not in sys.argv:
        sys.argv.insert(1, "--rl")
    main()


if __name__ == "__main__":
    main()
