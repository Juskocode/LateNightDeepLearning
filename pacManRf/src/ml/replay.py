"""Bounded, serializable experience replay for Pacman transitions."""

from __future__ import annotations

import random
import threading
from collections import Counter, deque
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_legal_action_mask: np.ndarray | None = None


class ReplayBatch(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_legal_action_masks: np.ndarray


class ReplayBuffer:
    """A thread-safe replay buffer with useful live statistics."""

    def __init__(
        self,
        capacity: int,
        *,
        observation_size: int | None = None,
        action_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if observation_size is not None and observation_size <= 0:
            raise ValueError("observation_size must be positive")
        if action_size is not None and action_size <= 0:
            raise ValueError("action_size must be positive")
        self.capacity = int(capacity)
        self.observation_size = observation_size
        self.action_size = action_size
        self._items: deque[Experience] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)
        self._lock = threading.RLock()

    def _validated(self, experience: Experience) -> Experience:
        state = np.asarray(experience.state, dtype=np.float32).reshape(-1).copy()
        next_state = np.asarray(experience.next_state, dtype=np.float32).reshape(-1).copy()
        if state.shape != next_state.shape:
            raise ValueError("state and next_state must have matching shapes")
        if self.observation_size is not None and state.shape != (self.observation_size,):
            raise ValueError(f"state must have shape ({self.observation_size},)")
        if not np.isfinite(state).all() or not np.isfinite(next_state).all():
            raise ValueError("states must contain finite values")
        action = int(experience.action)
        if self.action_size is not None and not 0 <= action < self.action_size:
            raise ValueError(f"action must be between 0 and {self.action_size - 1}")
        reward = float(experience.reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        raw_mask = experience.next_legal_action_mask
        if raw_mask is None:
            if self.action_size is None:
                raise ValueError("action_size is required when no next-action mask is supplied")
            next_legal_action_mask = np.ones(self.action_size, dtype=np.bool_)
        else:
            next_legal_action_mask = np.asarray(raw_mask, dtype=np.bool_).reshape(-1).copy()
            if self.action_size is not None and next_legal_action_mask.shape != (self.action_size,):
                raise ValueError(f"next_legal_action_mask must have shape ({self.action_size},)")
            if not next_legal_action_mask.any():
                raise ValueError("next_legal_action_mask must allow at least one action")
        return Experience(
            state,
            action,
            reward,
            next_state,
            bool(experience.done),
            next_legal_action_mask,
        )

    def append(self, experience: Experience) -> None:
        validated = self._validated(experience)
        with self._lock:
            self._items.append(validated)

    def push(
        self,
        state: np.ndarray | Sequence[float],
        action: int,
        reward: float,
        next_state: np.ndarray | Sequence[float],
        done: bool,
        next_legal_action_mask: np.ndarray | Sequence[bool] | None = None,
    ) -> None:
        self.append(
            Experience(
                np.asarray(state),
                int(action),
                float(reward),
                np.asarray(next_state),
                bool(done),
                None if next_legal_action_mask is None else np.asarray(next_legal_action_mask),
            )
        )

    def sample(self, size: int) -> list[Experience]:
        if size <= 0:
            raise ValueError("sample size must be positive")
        with self._lock:
            count = min(int(size), len(self._items))
            return self._rng.sample(list(self._items), count) if count else []

    def sample_batch(self, size: int) -> ReplayBatch | None:
        experiences = self.sample(size)
        if not experiences:
            return None
        states, actions, rewards, next_states, dones, next_legal_action_masks = zip(*experiences)
        return ReplayBatch(
            np.stack(states).astype(np.float32, copy=False),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32, copy=False),
            np.asarray(dones, dtype=np.bool_),
            np.stack(next_legal_action_masks).astype(np.bool_, copy=False),
        )

    def tail(self, size: int) -> list[Experience]:
        if size <= 0:
            return []
        with self._lock:
            return list(self._items)[-int(size):]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            items = list(self._items)
        rewards = np.asarray([item.reward for item in items], dtype=np.float64)
        actions = Counter(item.action for item in items)
        size = len(items)
        return {
            "size": size,
            "capacity": self.capacity,
            "fill_ratio": size / self.capacity,
            "mean_reward": float(rewards.mean()) if size else 0.0,
            "reward_std": float(rewards.std()) if size else 0.0,
            "min_reward": float(rewards.min()) if size else 0.0,
            "max_reward": float(rewards.max()) if size else 0.0,
            "positive": int(np.count_nonzero(rewards > 0)),
            "negative": int(np.count_nonzero(rewards < 0)),
            "zero": int(np.count_nonzero(rewards == 0)),
            "terminal": sum(int(item.done) for item in items),
            "action_counts": {int(action): int(count) for action, count in sorted(actions.items())},
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a weights-only-safe checkpoint representation."""

        with self._lock:
            items = [
                {
                    "state": torch.from_numpy(item.state.copy()),
                    "action": item.action,
                    "reward": item.reward,
                    "next_state": torch.from_numpy(item.next_state.copy()),
                    "done": item.done,
                    "next_legal_action_mask": torch.from_numpy(item.next_legal_action_mask.copy()),
                }
                for item in self._items
            ]
            rng_state = self._rng.getstate()
        return {
            "capacity": self.capacity,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "items": items,
            "rng_state": rng_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_observation = state.get("observation_size")
        saved_actions = state.get("action_size")
        if self.observation_size is not None and saved_observation not in (None, self.observation_size):
            raise ValueError("replay observation size does not match this agent")
        if self.action_size is not None and saved_actions not in (None, self.action_size):
            raise ValueError("replay action size does not match this agent")

        restored: list[Experience] = []
        for item in state.get("items", ()):
            state_value = item["state"]
            next_state_value = item["next_state"]
            if isinstance(state_value, torch.Tensor):
                state_value = state_value.detach().cpu().numpy()
            if isinstance(next_state_value, torch.Tensor):
                next_state_value = next_state_value.detach().cpu().numpy()
            next_legal_action_mask = item.get("next_legal_action_mask")
            if isinstance(next_legal_action_mask, torch.Tensor):
                next_legal_action_mask = next_legal_action_mask.detach().cpu().numpy()
            restored.append(
                self._validated(
                    Experience(
                        np.asarray(state_value),
                        int(item["action"]),
                        float(item["reward"]),
                        np.asarray(next_state_value),
                        bool(item["done"]),
                        None if next_legal_action_mask is None else np.asarray(next_legal_action_mask),
                    )
                )
            )
        with self._lock:
            self._items.clear()
            self._items.extend(restored[-self.capacity :])
            if "rng_state" in state:
                self._rng.setstate(state["rng_state"])

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
