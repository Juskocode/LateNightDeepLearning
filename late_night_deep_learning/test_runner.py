"""Run the repository test suite independent of the current directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import unittest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the project test suite")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(project_root / "tests"),
        top_level_dir=str(project_root),
    )
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
