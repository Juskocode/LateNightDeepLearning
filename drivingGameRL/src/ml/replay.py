"""Bounded and independently seeded experience replay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Iterator

import numpy as np


@dataclass(frozen=True, slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """FIFO replay memory whose sampling does not touch global RNG state."""

    def __init__(
        self, capacity: int, observation_size: int = 16, *, seed: int | None = None
    ):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if (
            isinstance(observation_size, bool)
            or not isinstance(observation_size, int)
            or observation_size <= 0
        ):
            raise ValueError("observation_size must be a positive integer")
        self.capacity = capacity
        self.observation_size = observation_size
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)
        self._reward_sum = 0.0
        self._terminal_count = 0

    def append(
        self,
        transition_or_state: Transition | np.ndarray | tuple[float, ...],
        action: int | None = None,
        reward: float | None = None,
        next_state: np.ndarray | tuple[float, ...] | None = None,
        done: bool | None = None,
    ) -> Transition:
        """Validate, defensively copy, and append one transition."""

        if isinstance(transition_or_state, Transition):
            if any(value is not None for value in (action, reward, next_state, done)):
                raise ValueError("a Transition cannot be combined with separate fields")
            source = transition_or_state
            action = source.action
            reward = source.reward
            next_state = source.next_state
            done = source.done
            state = source.state
        else:
            state = transition_or_state
        transition = self._validated_transition(state, action, reward, next_state, done)
        if len(self._items) == self.capacity:
            removed = self._items[0]
            self._reward_sum -= removed.reward
            self._terminal_count -= int(removed.done)
        self._items.append(transition)
        self._reward_sum += transition.reward
        self._terminal_count += int(transition.done)
        return transition

    add = append

    def sample(self, batch_size: int) -> tuple[Transition, ...]:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if batch_size > len(self._items):
            raise ValueError(
                "batch_size cannot exceed the number of stored transitions"
            )
        return tuple(self._rng.sample(tuple(self._items), batch_size))

    def clear(self) -> None:
        self._items.clear()
        self._reward_sum = 0.0
        self._terminal_count = 0

    def stats(self) -> dict[str, float | int]:
        size = len(self._items)
        return {
            "size": size,
            "capacity": self.capacity,
            "fill_ratio": size / self.capacity,
            "mean_reward": self._reward_sum / size if size else 0.0,
            "terminal": self._terminal_count,
        }

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._items)

    def _validated_transition(
        self,
        state: object,
        action: object,
        reward: object,
        next_state: object,
        done: object,
    ) -> Transition:
        state_array = self._observation("state", state)
        next_array = self._observation("next_state", next_state)
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        action_value = int(action)
        if action_value < 0:
            raise ValueError("action must be non-negative")
        if isinstance(reward, bool) or not isinstance(
            reward, (int, float, np.integer, np.floating)
        ):
            raise ValueError("reward must be finite")
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("reward must be finite")
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("done must be a boolean")
        return Transition(
            state_array,
            action_value,
            reward_value,
            next_array,
            bool(done),
        )

    def _observation(self, name: str, value: object) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.observation_size,):
            raise ValueError(
                f"{name} must have shape ({self.observation_size},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} values must be finite")
        return array.copy()
