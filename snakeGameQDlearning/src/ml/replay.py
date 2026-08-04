"""Bounded experience replay with a deliberately small public API."""

import math
import random
from collections import deque
from numbers import Integral, Real
from typing import NamedTuple

import numpy as np


class Experience(NamedTuple):
    state: np.ndarray
    action: list[int]
    reward: float
    next_state: np.ndarray
    done: bool


def validated_experience(experience: Experience) -> Experience:
    """Return an owned, finite transition or reject it before state mutates."""

    if not isinstance(experience, Experience):
        raise TypeError("experience must be an Experience instance")
    try:
        state = np.asarray(experience.state, dtype=np.float32)
        next_state = np.asarray(experience.next_state, dtype=np.float32)
        action = np.asarray(experience.action)
    except (TypeError, ValueError) as error:
        raise ValueError("experience contains non-numeric data") from error
    if state.shape != (11,) or next_state.shape != state.shape:
        raise ValueError("experience states must be matching 11-feature vectors")
    if not np.isfinite(state).all() or not np.isfinite(next_state).all():
        raise ValueError("experience states must contain only finite values")
    try:
        finite_action = np.isfinite(action.astype(np.float64)).all()
    except (TypeError, ValueError) as error:
        raise ValueError("experience action must be numeric") from error
    if (
        action.shape != (3,)
        or not finite_action
        or not np.isin(action, (0, 1)).all()
        or int(action.sum()) != 1
    ):
        raise ValueError("experience action must be one-hot: [straight, right, left]")
    if (
        isinstance(experience.reward, bool)
        or not isinstance(experience.reward, Real)
        or not math.isfinite(float(experience.reward))
    ):
        raise ValueError("experience reward must be a finite number")
    with np.errstate(over="ignore", invalid="ignore"):
        learner_reward = np.float32(experience.reward)
    if not np.isfinite(learner_reward):
        raise ValueError("experience reward must be representable as float32")
    if not isinstance(experience.done, (bool, np.bool_)):
        raise ValueError("experience done flag must be boolean")
    return Experience(
        state.copy(),
        [int(value) for value in action],
        float(learner_reward),
        next_state.copy(),
        bool(experience.done),
    )


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None):
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self._items = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)
        self._reward_sum = 0.0
        self._positive = 0
        self._negative = 0
        self._terminal = 0

    def _count(self, experience: Experience, direction: int) -> None:
        reward = float(experience.reward)
        self._reward_sum += direction * reward
        self._positive += direction * int(reward > 0)
        self._negative += direction * int(reward < 0)
        self._terminal += direction * int(experience.done)

    def append(self, experience: Experience) -> None:
        experience = validated_experience(experience)
        if len(self._items) == self.capacity:
            self._count(self._items[0], -1)
        self._items.append(experience)
        self._count(experience, 1)

    def sample(self, size: int) -> list[Experience]:
        if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
            raise ValueError("sample size must be a positive integer")
        if len(self._items) <= size:
            return list(self._items)
        return self._rng.sample(list(self._items), int(size))

    def tail(self, size: int) -> list[Experience]:
        if isinstance(size, bool) or not isinstance(size, Integral):
            raise ValueError("tail size must be an integer")
        if size <= 0:
            return []
        return list(self._items)[-size:]

    def stats(self) -> dict[str, float | int]:
        length = len(self._items)
        return {
            "mean_reward": self._reward_sum / length if length else 0.0,
            "fill_ratio": length / self.capacity,
            "positive": self._positive,
            "negative": self._negative,
            "terminal": self._terminal,
        }

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
        self._reward_sum = 0.0
        self._positive = 0
        self._negative = 0
        self._terminal = 0
