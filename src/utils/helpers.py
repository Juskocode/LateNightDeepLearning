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


def print_game_stats(game_number: int, score: int, record: int, mean_score: float) -> None:
    print(f"Game: {game_number:4d} | Score: {score:3d} | Record: {record:3d} | "
          f"Mean: {mean_score:6.2f}")


def calculate_statistics(scores: list) -> dict:
    if not scores:
        return {}

    return {
        'total_games': len(scores),
        'mean_score': sum(scores) / len(scores),
        'max_score': max(scores),
        'min_score': min(scores),
        'last_10_mean': sum(scores[-10:]) / min(len(scores), 10)
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
            version_str = filename.split('_v')[1].split('.')[0]
            versions.append(int(version_str))
        except (IndexError, ValueError):
            continue

    return max(versions) + 1 if versions else 1


def save_model_metadata(model_dir: str, version: int, score: int, games: int) -> None:
    metadata_file = os.path.join(model_dir, f"model_v{version:03d}_metadata.json")
    metadata = {
        'version': version,
        'best_score': score,
        'games_played': games,
        'timestamp': time.time(),
        'model_file': f"model_v{version:03d}.pth"
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_best_model_info(model_dir: str) -> Optional[Tuple[str, dict]]:
    if not os.path.exists(model_dir):
        return None

    pattern = os.path.join(model_dir, "model_v*_metadata.json")
    metadata_files = glob.glob(pattern)

    if not metadata_files:
        return None

    best_model = None
    best_score = -1
    best_metadata = None

    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            score = metadata.get('best_score', 0)
            if score > best_score:
                best_score = score
                best_model = metadata.get('model_file')
                best_metadata = metadata
        except (json.JSONDecodeError, KeyError):
            continue

    return (best_model, best_metadata) if best_model else None


def get_latest_model_info(model_dir: str) -> Optional[Tuple[str, dict]]:
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
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            version = metadata.get('version', 0)
            if version > latest_version:
                latest_version = version
                latest_model = metadata.get('model_file')
                latest_metadata = metadata
        except (json.JSONDecodeError, KeyError):
            continue

    return (latest_model, latest_metadata) if latest_model else None