import time
from typing import Any, Callable
import functools


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