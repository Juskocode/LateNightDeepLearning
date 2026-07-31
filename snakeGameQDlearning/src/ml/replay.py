"""Bounded experience replay with a deliberately small public API."""

import random
from collections import deque
from typing import NamedTuple

import numpy as np


class Experience(NamedTuple):
    state: np.ndarray
    action: list[int]
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items = deque(maxlen=capacity)
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
        if len(self._items) == self.capacity:
            self._count(self._items[0], -1)
        self._items.append(experience)
        self._count(experience, 1)

    def sample(self, size: int) -> list[Experience]:
        if len(self._items) <= size:
            return list(self._items)
        return self._rng.sample(list(self._items), size)

    def tail(self, size: int) -> list[Experience]:
        if size <= 0:
            return []
        return list(self._items)[-size:]

    def stats(self) -> dict[str, float | int]:
        length = len(self._items)
        return {
            "mean_reward": self._reward_sum / length if length else 0.0,
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
