"""One entry point for the games and project tests."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Late Night Deep Learning")
    parser.add_argument("command", choices=("pacman", "snake", "tests"))
    args, remaining = parser.parse_known_args()

    if args.command == "pacman":
        from pacManRf.main import main as run
    elif args.command == "snake":
        from snakeGameQDlearning.main import main as run
    else:
        from .test_runner import main as run

    # Child CLIs read sys.argv; retain only their options.
    import sys
    sys.argv = [sys.argv[0], *remaining]
    run()


if __name__ == "__main__":
    main()
