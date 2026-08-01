import time
import os
import glob
import json
from typing import Any, Callable, Optional, Tuple
import functools
from pathlib import Path


def timing_decorator(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def print_game_stats(
    game_number: int, score: int, record: int, mean_score: float
) -> None:
    print(
        f"Game: {game_number:4d} | Score: {score:3d} | Record: {record:3d} | "
        f"Mean: {mean_score:6.2f}"
    )


def calculate_statistics(scores: list) -> dict:
    if not scores:
        return {}

    return {
        "total_games": len(scores),
        "mean_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "last_10_mean": sum(scores[-10:]) / min(len(scores), 10),
    }


def get_next_model_version(model_dir: str) -> int:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    pattern = os.path.join(model_dir, "model_v*.pth")
    existing_models = glob.glob(pattern)

    if not existing_models:
        return 1

    versions = []
    for model_path in existing_models:
        filename = os.path.basename(model_path)
        try:
            version_str = filename.split("_v")[1].split(".")[0]
            versions.append(int(version_str))
        except (IndexError, ValueError):
            continue

    return max(versions) + 1 if versions else 1


def save_model_metadata(
    model_dir: str,
    version: int,
    best_score: int,
    mean_score: float,
    games: int,
    *,
    algorithm: str | None = None,
    evaluation_mean: float | None = None,
    checkpoint_reason: str | None = None,
    experiment: dict[str, Any] | None = None,
) -> None:
    metadata_file = os.path.join(model_dir, f"model_v{version:03d}_metadata.json")
    metadata = {
        "version": version,
        "best_score": best_score,
        "mean_score": mean_score,
        "games_played": games,
        "timestamp": time.time(),
        "model_file": f"model_v{version:03d}.pth",
    }
    if algorithm is not None:
        metadata["algorithm"] = algorithm
    if evaluation_mean is not None:
        metadata["evaluation_mean"] = evaluation_mean
    if checkpoint_reason is not None:
        metadata["checkpoint_reason"] = checkpoint_reason
    if experiment is not None:
        metadata["experiment"] = dict(experiment)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def update_model_metadata(
    model_dir: str, version: int, mean_score: float, games: int
) -> None:
    """Update only mean_score and games_played, keep best_score unchanged"""
    metadata_file = os.path.join(model_dir, f"model_v{version:03d}_metadata.json")

    # Read existing metadata
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist, create with minimal data
        metadata = {
            "version": version,
            "best_score": 0,
            "model_file": f"model_v{version:03d}.pth",
        }

    # Update only specific fields
    metadata["mean_score"] = mean_score
    metadata["games_played"] = games
    metadata["timestamp"] = time.time()

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def _matches_algorithm(metadata: dict, algorithm: str | None) -> bool:
    if algorithm is None:
        return True
    stored = metadata.get("algorithm")
    if stored is None:
        # Pre-registry checkpoints used the shared MLP and are safe for these
        # two historical modes only.
        return algorithm in ("dqn", "double_dqn")
    return stored == algorithm


def _experiment_key(metadata: dict) -> tuple:
    experiment = metadata.get("experiment")
    if not isinstance(experiment, dict):
        return ("legacy",)
    seeds = experiment.get("validation_seeds")
    seed_key = tuple(seeds) if isinstance(seeds, list) else ()
    return (experiment.get("environment"), seed_key)


def _matches_experiment(
    metadata: dict,
    *,
    environment: str | None,
    validation_seeds: tuple[int, ...] | None,
) -> bool:
    if environment is None and validation_seeds is None:
        return True
    experiment = metadata.get("experiment")
    if not isinstance(experiment, dict):
        # Historical Snake checkpoints used only the standard board and did not
        # record a validation suite.
        return environment in (None, "standard") and validation_seeds is None
    if environment is not None and experiment.get("environment") != environment:
        return False
    if validation_seeds is not None:
        return experiment.get("validation_seeds") == list(validation_seeds)
    return True


def get_best_model_info(
    model_dir: str,
    *,
    algorithm: str | None = None,
    environment: str | None = None,
    validation_seeds: tuple[int, ...] | None = None,
) -> Optional[Tuple[str, dict]]:
    if not os.path.exists(model_dir):
        return None

    pattern = os.path.join(model_dir, "model_v*_metadata.json")
    metadata_files = glob.glob(pattern)

    if not metadata_files:
        return None

    candidates = []

    for metadata_file in metadata_files:
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            if not _matches_algorithm(metadata, algorithm) or not _matches_experiment(
                metadata,
                environment=environment,
                validation_seeds=validation_seeds,
            ):
                continue

            if metadata.get("model_file"):
                candidates.append(metadata)
        except (json.JSONDecodeError, KeyError):
            continue

    if not candidates:
        return None

    # Without an exact suite filter, select within the most recently created
    # experiment group. Validation means from different seed suites must never
    # compete with one another merely because they share an algorithm name.
    if validation_seeds is None:
        groups: dict[tuple, list[dict]] = {}
        for metadata in candidates:
            groups.setdefault(_experiment_key(metadata), []).append(metadata)
        candidates = max(
            groups.values(),
            key=lambda group: max(int(item.get("version", 0)) for item in group),
        )
    evaluated = [
        metadata
        for metadata in candidates
        if isinstance(metadata.get("evaluation_mean"), (int, float))
    ]
    if evaluated:
        # Once held-out results exist, checkpoint choice is validation-driven.
        best_metadata = max(
            evaluated,
            key=lambda item: (
                float(item["evaluation_mean"]),
                float(item.get("mean_score", 0.0)),
                int(item.get("version", 0)),
            ),
        )
    else:
        # Full backward compatibility for pre-evaluation metadata.
        best_metadata = max(
            candidates,
            key=lambda item: (
                int(item.get("best_score", 0)),
                float(item.get("mean_score", 0.0)),
                int(item.get("version", 0)),
            ),
        )
    return best_metadata["model_file"], best_metadata


def get_latest_model_info(
    model_dir: str,
    *,
    algorithm: str | None = None,
    environment: str | None = None,
    validation_seeds: tuple[int, ...] | None = None,
) -> Optional[Tuple[str, dict]]:
    if not os.path.exists(model_dir):
        return None

    pattern = os.path.join(model_dir, "model_v*_metadata.json")
    metadata_files = glob.glob(pattern)

    if not metadata_files:
        return None

    latest_model = None
    latest_version = -1
    latest_metadata = None

    for metadata_file in metadata_files:
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            if not _matches_algorithm(metadata, algorithm) or not _matches_experiment(
                metadata,
                environment=environment,
                validation_seeds=validation_seeds,
            ):
                continue

            version = metadata.get("version", 0)
            if version > latest_version:
                latest_version = version
                latest_model = metadata.get("model_file")
                latest_metadata = metadata
        except (json.JSONDecodeError, KeyError):
            continue

    return (latest_model, latest_metadata) if latest_model else None
