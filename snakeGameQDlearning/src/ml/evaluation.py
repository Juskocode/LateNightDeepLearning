"""Held-out evaluation that never trains or consumes the exploration RNG."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from snakeGameQDlearning.src.game.environments import (
    EnvironmentPreset,
    EpisodeSeedStreams,
)
from snakeGameQDlearning.src.game.snake_game import SnakeGameAI


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    mean_score: float
    std_score: float
    median_score: float
    min_score: int
    max_score: int
    mean_steps: float
    win_rate: float
    termination_counts: dict[str, int]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSuiteSelection:
    """Resolved immutable scenarios for validation or final testing."""

    name: str
    seeds: tuple[int, ...]
    seed_root: int
    source: str
    evaluation_round: int | None = None


def resolve_evaluation_suite(
    streams: EpisodeSeedStreams,
    suite: str,
    count: int,
    *,
    loaded_metadata: dict | None = None,
    environment: str | None = None,
    prefer_stored: bool = False,
) -> EvaluationSuiteSelection:
    """Select a fixed suite, optionally reproducing a checkpoint's scenarios.

    Stored scenarios are used only when the checkpoint records the same board
    preset. Legacy checkpoints and deliberate seed/count overrides fall back to
    a suite derived from the current CLI roots.
    """

    normalized = suite.lower().replace("-", "_")
    if normalized not in ("validation", "final_test"):
        raise ValueError("suite must be 'validation' or 'final_test'")
    if count <= 0:
        raise ValueError("evaluation episode count must be positive")

    experiment = (
        dict((loaded_metadata or {}).get("experiment", {}))
        if isinstance((loaded_metadata or {}).get("experiment", {}), dict)
        else {}
    )
    stored_environment = experiment.get("environment")
    stored_values = experiment.get(f"{normalized}_seeds")
    compatible = (
        prefer_stored
        and environment is not None
        and stored_environment == environment
        and isinstance(stored_values, list)
        and bool(stored_values)
        and all(isinstance(seed, int) for seed in stored_values)
    )
    if compatible:
        root_key = (
            "validation_seed_root"
            if normalized == "validation"
            else "final_test_seed_root"
        )
        default_root = (
            streams.evaluation_seed
            if normalized == "validation"
            else streams.final_test_seed
        )
        round_value = experiment.get("evaluation_round")
        return EvaluationSuiteSelection(
            normalized,
            tuple(int(seed) for seed in stored_values),
            int(experiment.get(root_key, default_root)),
            "checkpoint",
            int(round_value) if isinstance(round_value, int) else None,
        )

    if normalized == "validation":
        seeds = streams.validation_seeds(count)
        root = streams.evaluation_seed
    else:
        seeds = streams.final_test_seeds(count)
        root = streams.final_test_seed
    return EvaluationSuiteSelection(normalized, seeds, root, "cli")


def evaluate_agent(
    agent,
    seeds: Iterable[int],
    preset: EnvironmentPreset,
    *,
    max_steps: int | None = None,
) -> EvaluationResult:
    """Run a greedy policy on unseen seeds without updating policy or replay."""

    evaluation_seeds = tuple(int(seed) for seed in seeds)
    if not evaluation_seeds:
        raise ValueError("at least one evaluation seed is required")

    scores: list[int] = []
    step_counts: list[int] = []
    reasons: Counter[str] = Counter()
    step_limit = max_steps or (preset.width // 20) * (preset.height // 20) * 4
    for seed in evaluation_seeds:
        game = SnakeGameAI(
            width=preset.width,
            height=preset.height,
            render=False,
            seed=seed,
            randomize_start=True,
            process_events=False,
        )
        done = False
        steps = 0
        while not done and steps < step_limit:
            state = agent.get_state(game)
            action = agent.get_action(state, explore=False)
            _, done, _ = game.play_step(action, render_frame=False)
            steps += int(game.transition_applied)
        reason = game.termination_reason or "evaluation_limit"
        scores.append(game.score)
        step_counts.append(steps)
        reasons[reason] += 1

    score_array = np.asarray(scores, dtype=np.float64)
    return EvaluationResult(
        episodes=len(scores),
        mean_score=float(score_array.mean()),
        std_score=float(score_array.std()),
        median_score=float(np.median(score_array)),
        min_score=int(score_array.min()),
        max_score=int(score_array.max()),
        mean_steps=float(np.mean(step_counts)),
        win_rate=reasons["win"] / len(scores),
        termination_counts=dict(reasons),
    )
