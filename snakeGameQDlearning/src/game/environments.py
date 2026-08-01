"""Educational Snake environments and deterministic generalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class EnvironmentPreset:
    name: str
    width: int
    height: int
    learning_goal: str


ENVIRONMENT_PRESETS = {
    "tutorial": EnvironmentPreset(
        "tutorial", 200, 160, "Fast feedback on turns, danger rays, and food direction"
    ),
    "compact": EnvironmentPreset(
        "compact", 320, 240, "Planning with tighter walls and shorter routes"
    ),
    "standard": EnvironmentPreset(
        "standard", 640, 480, "Balanced training and held-out evaluation"
    ),
    "wide": EnvironmentPreset(
        "wide",
        720,
        360,
        "Transfer to a different aspect ratio and longer horizontal routes",
    ),
}


def get_environment_preset(name: str) -> EnvironmentPreset:
    try:
        return ENVIRONMENT_PRESETS[name.lower()]
    except KeyError as error:
        choices = ", ".join(ENVIRONMENT_PRESETS)
        raise ValueError(
            f"unknown environment {name!r}; choose one of: {choices}"
        ) from error


class EpisodeSeedStreams:
    """Reproducible, independent seed streams for training and evaluation.

    Validation and final-test suites are deliberately fixed for the lifetime of
    an experiment.  Re-evaluating a later checkpoint therefore measures it on
    the same scenarios instead of comparing means from different random draws.
    """

    _TRAIN_TAG = 0x54524149
    _VALIDATION_TAG = 0x56414C49
    _FINAL_TEST_TAG = 0x54455354

    def __init__(
        self,
        training_seed: int,
        evaluation_seed: int | None = None,
        final_test_seed: int | None = None,
    ):
        self.training_seed = int(training_seed)
        self.evaluation_seed = (
            int(evaluation_seed)
            if evaluation_seed is not None
            else int(training_seed) + 1_000_003
        )
        self.final_test_seed = (
            int(final_test_seed)
            if final_test_seed is not None
            else int(training_seed) + 2_000_003
        )

    @staticmethod
    def _derive(seed: int, stream_tag: int, *indices: int) -> int:
        sequence = np.random.SeedSequence([seed, stream_tag, *indices])
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def train_seed(self, episode: int) -> int:
        return self._derive(self.training_seed, self._TRAIN_TAG, int(episode))

    def validation_seeds(self, count: int) -> tuple[int, ...]:
        if count <= 0:
            raise ValueError("validation episode count must be positive")
        return tuple(
            self._derive(self.evaluation_seed, self._VALIDATION_TAG, index)
            for index in range(count)
        )

    def final_test_seeds(self, count: int) -> tuple[int, ...]:
        if count <= 0:
            raise ValueError("final-test episode count must be positive")
        return tuple(
            self._derive(self.final_test_seed, self._FINAL_TEST_TAG, index)
            for index in range(count)
        )

    def evaluation_seeds(self, evaluation_round: int, count: int) -> tuple[int, ...]:
        """Backward-compatible alias for the fixed validation suite.

        ``evaluation_round`` is accepted so older notebooks keep running, but
        it no longer changes the scenarios.  The round belongs in checkpoint
        metadata, not in the validation data split.
        """

        int(evaluation_round)
        return self.validation_seeds(count)


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    first_episode: int
    randomized_start_probability: float


class SnakeCurriculum:
    """A gentle curriculum that removes fixed-spawn memorization over time."""

    stages = (
        CurriculumStage("orientation", 0, 0.15),
        CurriculumStage("spawn_shift", 20, 0.55),
        CurriculumStage("generalization", 75, 1.0),
    )

    def stage_for(self, episode: int) -> CurriculumStage:
        active = self.stages[0]
        for stage in self.stages:
            if episode >= stage.first_episode:
                active = stage
        return active

    def randomize_start(self, episode: int, episode_seed: int) -> bool:
        stage = self.stage_for(episode)
        chooser = random.Random((int(episode_seed) << 1) ^ 0xC0FFEE)
        return chooser.random() < stage.randomized_start_probability


def available_environments() -> tuple[str, ...]:
    return tuple(ENVIRONMENT_PRESETS)
