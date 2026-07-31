"""Run the Pacman game or capture a deterministic documentation frame."""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Play the Late Night Pacman demo")
    parser.add_argument("--screenshot", help="Save a PNG and exit")
    parser.add_argument("--headless", action="store_true", help="Use SDL's dummy display")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.headless or args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame
    from pacManRf.src.game.pacmanGame import PacmanGame

    game = PacmanGame(render=not (args.headless or args.screenshot))
    if args.screenshot:
        # Place the scene in an informative, reproducible state.
        game.phase = type(game.phase).ACTIVE
        game.frightened_timer = 4.2
        game.score = 460
        game.save_screenshot(args.screenshot)
        pygame.quit()
        return

    while game.running:
        game.play_step()
    pygame.quit()


if __name__ == "__main__":
    main()
