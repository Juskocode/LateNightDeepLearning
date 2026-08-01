"""Reusable DQN and Double-DQN learning agent for the Driving Lab."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .config import DQNConfig
from .network import DrivingQNetwork
from .replay import ReplayBuffer


class DrivingDQNAgent:
    """CPU DQN learner with replay, target network, and observable internals."""

    CHECKPOINT_VERSION = 1

    def __init__(self, config: DQNConfig | None = None):
        self.config = config or DQNConfig()
        # Isolate initialization from the application's global Torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.online_network = DrivingQNetwork(self.config)
            self.target_network = DrivingQNetwork(self.config)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.loss_function = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(
            self.config.replay_capacity,
            self.config.observation_size,
            seed=self.config.seed,
        )
        self._rng = random.Random(self.config.seed)
        self.environment_steps = 0
        self.gradient_steps = 0
        self.target_syncs = 0
        self.last_loss = 0.0
        self.last_gradient_norm = 0.0
        self.last_predicted_mean = 0.0
        self.last_target_mean = 0.0
        self.last_td_error = 0.0
        self.last_action: int | None = None
        self.last_policy = "uninitialized"
        self.last_q_values = np.zeros(self.config.action_size, dtype=np.float32)
        self.action_counts = [0 for _ in range(self.config.action_size)]

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.environment_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    @property
    def network(self) -> DrivingQNetwork:
        """Short alias useful to generic policy renderers."""

        return self.online_network

    @property
    def memory(self) -> ReplayBuffer:
        return self.replay

    def q_values(self, observation: Sequence[float] | np.ndarray) -> np.ndarray:
        array = self._observation(observation)
        was_training = self.online_network.training
        self.online_network.eval()
        try:
            with torch.no_grad():
                values = self.online_network(torch.from_numpy(array))
        finally:
            self.online_network.train(was_training)
        return values.detach().cpu().numpy().astype(np.float32, copy=True)

    def select_action(
        self, observation: Sequence[float] | np.ndarray, *, explore: bool = True
    ) -> int:
        values = self.q_values(observation)
        if explore and self._rng.random() < self.epsilon:
            action = self._rng.randrange(self.config.action_size)
            policy = "explore"
        else:
            action = int(np.argmax(values))
            policy = "greedy"
        self.last_q_values = values
        self.last_action = action
        self.last_policy = policy
        self.action_counts[action] += 1
        return action

    act = select_action

    def observe(
        self,
        state: Sequence[float] | np.ndarray,
        action: int,
        reward: float,
        next_state: Sequence[float] | np.ndarray,
        done: bool,
    ) -> float | None:
        """Store a transition and run one scheduled replay update if ready."""

        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        action = int(action)
        if not 0 <= action < self.config.action_size:
            raise ValueError(
                f"action must be in the [0, {self.config.action_size}) interval"
            )
        self.replay.append(state, action, reward, next_state, done)
        self.environment_steps += 1
        learning_start = max(self.config.batch_size, self.config.warmup_steps)
        if len(self.replay) < learning_start:
            return None
        if self.environment_steps % self.config.train_interval:
            return None
        return self.train_step()

    remember = observe

    def train_step(self) -> float | None:
        """Fit one uniformly sampled replay batch."""

        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.replay.sample(self.config.batch_size)
        states = torch.from_numpy(np.stack([item.state for item in batch]))
        actions = torch.tensor([item.action for item in batch], dtype=torch.long)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([item.next_state for item in batch]))
        dones = torch.tensor([item.done for item in batch], dtype=torch.bool)

        self.online_network.train()
        predicted = self.online_network(states).gather(1, actions[:, None]).squeeze(1)
        bootstrap = self._bootstrap_values(next_states)
        targets = rewards + (~dones).float() * self.config.gamma * bootstrap

        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_function(predicted, targets)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_sync_interval == 0:
            self.sync_target()
        td_errors = targets.detach() - predicted.detach()
        self.last_loss = float(loss.detach())
        self.last_gradient_norm = float(gradient_norm.detach())
        self.last_predicted_mean = float(predicted.detach().mean())
        self.last_target_mean = float(targets.detach().mean())
        self.last_td_error = float(td_errors.abs().mean())
        return self.last_loss

    learn = train_step

    def _bootstrap_values(self, next_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_values = self.target_network(next_states)
            if self.config.algorithm == "dqn":
                return target_values.max(dim=1).values
            online_actions = self.online_network(next_states).argmax(
                dim=1, keepdim=True
            )
            return target_values.gather(1, online_actions).squeeze(1)

    def sync_target(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.target_syncs += 1

    def copy_weights_from(
        self,
        source: "DrivingDQNAgent | DrivingQNetwork",
        *,
        sync_target: bool = True,
    ) -> None:
        """Copy a genotype without sharing any mutable tensors."""

        network = (
            source.online_network if isinstance(source, DrivingDQNAgent) else source
        )
        if not isinstance(network, DrivingQNetwork):
            raise TypeError("source must be a DrivingDQNAgent or DrivingQNetwork")
        self.online_network.load_state_dict(network.state_dict())
        if sync_target:
            self.sync_target()

    def clone(
        self, *, seed: int | None = None, include_optimizer: bool = False
    ) -> "DrivingDQNAgent":
        """Return an independent policy clone suitable for a population child."""

        clone_seed = self.config.seed if seed is None else seed
        clone = DrivingDQNAgent(replace(self.config, seed=clone_seed))
        clone.online_network.load_state_dict(self.online_network.state_dict())
        if include_optimizer:
            clone.target_network.load_state_dict(self.target_network.state_dict())
            clone.optimizer.load_state_dict(deepcopy(self.optimizer.state_dict()))
            clone.environment_steps = self.environment_steps
            clone.gradient_steps = self.gradient_steps
            clone.target_syncs = self.target_syncs
            clone.last_loss = self.last_loss
            clone.last_gradient_norm = self.last_gradient_norm
            clone.last_predicted_mean = self.last_predicted_mean
            clone.last_target_mean = self.last_target_mean
            clone.last_td_error = self.last_td_error
            clone._rng.setstate(self._rng.getstate())
        else:
            clone.target_network.load_state_dict(clone.online_network.state_dict())
        clone.target_network.eval()
        return clone

    def network_snapshot(
        self, observation: Sequence[float] | np.ndarray
    ) -> dict[str, Any]:
        return self.online_network.snapshot(observation)

    def telemetry(
        self, observation: Sequence[float] | np.ndarray | None = None
    ) -> dict[str, Any]:
        """Return compact live metrics; full weights live in network_snapshot."""

        q_values = (
            self.q_values(observation).tolist()
            if observation is not None
            else self.last_q_values.astype(float).tolist()
        )
        parameter_norm = math.sqrt(
            sum(
                float(torch.sum(parameter.detach() ** 2))
                for parameter in self.online_network.parameters()
            )
        )
        target_gap = sum(
            float(torch.mean(torch.abs(online.detach() - target.detach())))
            for online, target in zip(
                self.online_network.parameters(), self.target_network.parameters()
            )
        )
        return {
            "algorithm": self.config.algorithm,
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "epsilon": self.epsilon,
            "last_loss": self.last_loss,
            "gradient_norm": self.last_gradient_norm,
            "mean_predicted_q": self.last_predicted_mean,
            "mean_target_q": self.last_target_mean,
            "mean_absolute_td_error": self.last_td_error,
            "last_action": self.last_action,
            "policy": self.last_policy,
            "q_values": q_values,
            "greedy_action": int(np.argmax(q_values)),
            "action_counts": self.action_counts.copy(),
            "target_syncs": self.target_syncs,
            "target_parameter_gap": target_gap,
            "parameter_count": self.online_network.parameter_count,
            "parameter_norm": parameter_norm,
            "architecture": list(self.online_network.architecture),
            "replay": self.replay.stats(),
        }

    def state_dict(self) -> dict[str, Any]:
        """Serializable training state (replay contents are intentionally omitted)."""

        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "config": self.config.to_dict(),
            "online_network": deepcopy(self.online_network.state_dict()),
            "target_network": deepcopy(self.target_network.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "target_syncs": self.target_syncs,
            "metrics": {
                "last_loss": self.last_loss,
                "last_gradient_norm": self.last_gradient_norm,
                "last_predicted_mean": self.last_predicted_mean,
                "last_target_mean": self.last_target_mean,
                "last_td_error": self.last_td_error,
                "last_action": self.last_action,
                "last_policy": self.last_policy,
                "last_q_values": self.last_q_values.tolist(),
                "action_counts": self.action_counts.copy(),
            },
            "policy_rng_state": self._rng.getstate(),
        }

    def load_state_dict(
        self, state: Mapping[str, Any], *, load_optimizer: bool = True
    ) -> None:
        if int(state.get("checkpoint_version", -1)) != self.CHECKPOINT_VERSION:
            raise ValueError("unsupported Driving DQN checkpoint version")
        saved_config = DQNConfig.from_dict(state["config"])
        current_signature = (
            self.config.observation_size,
            self.config.action_size,
            self.config.hidden_sizes,
            self.config.algorithm,
        )
        saved_signature = (
            saved_config.observation_size,
            saved_config.action_size,
            saved_config.hidden_sizes,
            saved_config.algorithm,
        )
        if current_signature != saved_signature:
            raise ValueError("checkpoint architecture or algorithm is incompatible")
        self.online_network.load_state_dict(state["online_network"])
        self.target_network.load_state_dict(state["target_network"])
        self.target_network.eval()
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
        self.environment_steps = int(state.get("environment_steps", 0))
        self.gradient_steps = int(state.get("gradient_steps", 0))
        self.target_syncs = int(state.get("target_syncs", 0))
        metrics = state.get("metrics", {})
        self.last_loss = float(metrics.get("last_loss", 0.0))
        self.last_gradient_norm = float(metrics.get("last_gradient_norm", 0.0))
        self.last_predicted_mean = float(metrics.get("last_predicted_mean", 0.0))
        self.last_target_mean = float(metrics.get("last_target_mean", 0.0))
        self.last_td_error = float(metrics.get("last_td_error", 0.0))
        self.last_action = metrics.get("last_action")
        self.last_policy = str(metrics.get("last_policy", "restored"))
        q_values = np.asarray(
            metrics.get("last_q_values", [0.0] * self.config.action_size),
            dtype=np.float32,
        )
        counts = list(metrics.get("action_counts", [0] * self.config.action_size))
        if (
            q_values.shape != (self.config.action_size,)
            or len(counts) != self.config.action_size
        ):
            raise ValueError("checkpoint action telemetry has an incompatible shape")
        self.last_q_values = q_values
        self.action_counts = [int(value) for value in counts]
        if "policy_rng_state" in state:
            self._rng.setstate(state["policy_rng_state"])

    def save(self, path: str | Path) -> Path:
        """Atomically replace a checkpoint after Torch has written it fully."""

        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(self.state_dict(), temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> None:
        state = self.read_checkpoint(path)
        self.load_state_dict(state, load_optimizer=load_optimizer)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, load_optimizer: bool = True
    ) -> "DrivingDQNAgent":
        state = cls.read_checkpoint(path)
        agent = cls(DQNConfig.from_dict(state["config"]))
        agent.load_state_dict(state, load_optimizer=load_optimizer)
        return agent

    @staticmethod
    def read_checkpoint(path: str | Path) -> dict[str, Any]:
        """Read a validated checkpoint payload for higher-level runtimes."""

        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Driving DQN checkpoint not found: {checkpoint}")
        try:
            return torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older Torch
            return torch.load(checkpoint, map_location="cpu")

    # Compatibility for callers written before the checkpoint reader became a
    # public part of the agent API.
    _read_checkpoint = read_checkpoint

    def _observation(self, observation: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        expected = (self.config.observation_size,)
        if array.shape != expected:
            raise ValueError(
                f"observation must have shape {expected}, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("observation values must be finite")
        return array.copy()
