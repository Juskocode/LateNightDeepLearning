"""Reusable DQN and Double-DQN learning agent for the Driving Lab."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from numbers import Real
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ..learning_health import build_learning_health
from .config import DQNConfig
from .network import DrivingQNetwork
from .replay import ReplayBuffer


class DrivingDQNAgent:
    """CPU DQN learner with replay, target network, and observable internals."""

    # Version 3 binds standalone policies to the progressive multi-lap reward
    # and termination contract. Earlier Q values treat the first completed lap
    # as terminal, so accepting them would silently change a learned target.
    CHECKPOINT_VERSION = 3
    LEGACY_OBSERVATION_SIZE = 12
    CURRENT_OBSERVATION_SIZE = 16
    OBSERVATION_BASE_FEATURES = 7
    # The denser nine-ray fan retains every legacy angle. The four odd slots
    # are new intermediate readings and receive neutral weights during
    # migration so the old policy's predictions remain exactly reproducible.
    LEGACY_RAY_INDICES = (0, 2, 4, 6, 8)

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
        self.gradient_clip_events = 0
        self.nonfinite_update_rejections = 0
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
        result = values.detach().cpu().numpy().astype(np.float32, copy=True)
        if not np.isfinite(result).all():
            raise FloatingPointError("driving policy produced non-finite Q-values")
        return result

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

        if not torch.isfinite(predicted).all() or not torch.isfinite(targets).all():
            self.nonfinite_update_rejections += 1
            raise FloatingPointError("DQN update produced non-finite values")

        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_function(predicted, targets)
        if not torch.isfinite(loss):
            self.nonfinite_update_rejections += 1
            raise FloatingPointError("DQN update produced a non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.config.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            self.nonfinite_update_rejections += 1
            raise FloatingPointError("DQN update produced non-finite gradients")
        if float(gradient_norm.detach()) > self.config.gradient_clip:
            self.gradient_clip_events += 1
        self.optimizer.step()
        if any(
            not torch.isfinite(parameter).all()
            for parameter in self.online_network.parameters()
        ):
            self.nonfinite_update_rejections += 1
            raise FloatingPointError(
                "DQN optimizer step produced non-finite network parameters"
            )

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
            clone.gradient_clip_events = self.gradient_clip_events
            clone.nonfinite_update_rejections = self.nonfinite_update_rejections
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

        telemetry_alerts: list[str] = []

        def finite_metric(name: str, value: object) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                telemetry_alerts.append(f"malformed:{name}")
                return 0.0
            if not math.isfinite(numeric):
                telemetry_alerts.append(f"non_finite:{name}")
                return 0.0
            return numeric

        try:
            q_values = (
                self.q_values(observation).tolist()
                if observation is not None
                else self.last_q_values.astype(float).tolist()
            )
        except (FloatingPointError, TypeError, ValueError):
            q_values = [0.0] * self.config.action_size
            telemetry_alerts.append("non_finite:q_values")
        if not all(math.isfinite(float(value)) for value in q_values):
            q_values = [0.0] * self.config.action_size
            telemetry_alerts.append("non_finite:q_values")
        parameter_norm = finite_metric(
            "parameter_norm",
            math.sqrt(
                sum(
                    float(torch.sum(parameter.detach() ** 2))
                    for parameter in self.online_network.parameters()
                )
            ),
        )
        target_gap = finite_metric(
            "target_parameter_gap",
            sum(
                float(torch.mean(torch.abs(online.detach() - target.detach())))
                for online, target in zip(
                    self.online_network.parameters(), self.target_network.parameters()
                )
            ),
        )
        replay = self.replay.stats()
        learning = {
            "algorithm": self.config.algorithm,
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "epsilon": finite_metric("epsilon", self.epsilon),
            "last_loss": finite_metric("last_loss", self.last_loss),
            "gradient_norm": finite_metric("gradient_norm", self.last_gradient_norm),
            "gradient_clip": self.config.gradient_clip,
            "gradient_clip_events": self.gradient_clip_events,
            "nonfinite_update_rejections": self.nonfinite_update_rejections,
            "mean_predicted_q": finite_metric(
                "mean_predicted_q", self.last_predicted_mean
            ),
            "mean_target_q": finite_metric("mean_target_q", self.last_target_mean),
            "mean_absolute_td_error": finite_metric(
                "mean_absolute_td_error", self.last_td_error
            ),
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
            "replay": replay,
        }
        health = build_learning_health(
            learning=learning,
            replay=replay,
            environment_decisions=self.environment_steps,
            batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            gradient_clip=self.config.gradient_clip,
        )
        if telemetry_alerts:
            health["finite"] = False
            health["status"] = "critical"
            health["alerts"] = list(
                dict.fromkeys([*health["alerts"], *telemetry_alerts])
            )
        learning["health"] = health
        return learning

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
                "gradient_clip_events": self.gradient_clip_events,
                "nonfinite_update_rejections": self.nonfinite_update_rejections,
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
        state = self._validated_checkpoint_payload(
            state,
            load_optimizer=load_optimizer,
        )
        if int(state.get("checkpoint_version", -1)) != self.CHECKPOINT_VERSION:
            raise ValueError("unsupported Driving DQN checkpoint version")
        saved_config = DQNConfig.from_dict(state["config"])
        if not self.checkpoint_config_compatible(self.config, saved_config):
            raise ValueError("checkpoint architecture or algorithm is incompatible")
        migrate_legacy_input = (
            saved_config.observation_size == self.LEGACY_OBSERVATION_SIZE
            and self.config.observation_size == self.CURRENT_OBSERVATION_SIZE
        )
        online_state = state["online_network"]
        target_state = state["target_network"]
        if migrate_legacy_input:
            online_state = self._migrate_legacy_network_state(online_state)
            target_state = self._migrate_legacy_network_state(target_state)
        optimizer_state = state.get("optimizer")
        if load_optimizer and migrate_legacy_input:
            optimizer_state = self._migrate_legacy_optimizer_state(optimizer_state)

        # Preflight every mutating PyTorch loader against independent objects.
        # A malformed target or optimizer payload therefore cannot leave the
        # live online network half-restored.
        candidate_online = deepcopy(self.online_network)
        candidate_target = deepcopy(self.target_network)
        candidate_online.load_state_dict(online_state)
        candidate_target.load_state_dict(target_state)
        if load_optimizer:
            candidate_optimizer = torch.optim.Adam(
                candidate_online.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            candidate_optimizer.load_state_dict(optimizer_state)
            self._validate_adam_optimizer(candidate_optimizer)

        self.online_network.load_state_dict(online_state)
        self.target_network.load_state_dict(target_state)
        self.target_network.eval()
        if load_optimizer:
            self.optimizer.load_state_dict(optimizer_state)
        self.environment_steps = int(state.get("environment_steps", 0))
        self.gradient_steps = int(state.get("gradient_steps", 0))
        self.target_syncs = int(state.get("target_syncs", 0))
        metrics = state.get("metrics", {})
        self.last_loss = float(metrics.get("last_loss", 0.0))
        self.last_gradient_norm = float(metrics.get("last_gradient_norm", 0.0))
        self.last_predicted_mean = float(metrics.get("last_predicted_mean", 0.0))
        self.last_target_mean = float(metrics.get("last_target_mean", 0.0))
        self.last_td_error = float(metrics.get("last_td_error", 0.0))
        self.gradient_clip_events = int(metrics.get("gradient_clip_events", 0))
        self.nonfinite_update_rejections = int(
            metrics.get("nonfinite_update_rejections", 0)
        )
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

    @classmethod
    def _validated_checkpoint_payload(
        cls,
        state: object,
        *,
        load_optimizer: bool,
    ) -> Mapping[str, Any]:
        """Validate the portable envelope before mutating a live learner."""

        if not isinstance(state, Mapping):
            raise ValueError("Driving DQN checkpoint must be a mapping")
        version = state.get("checkpoint_version", -1)
        if isinstance(version, bool) or not isinstance(version, (int, np.integer)):
            raise ValueError("Driving DQN checkpoint version must be an integer")
        if int(version) != cls.CHECKPOINT_VERSION:
            if int(version) in (1, 2):
                raise ValueError(
                    f"Driving DQN checkpoint v{int(version)} uses a legacy "
                    "action, reward, or lap-target contract; start a fresh "
                    "learner with --fresh"
                )
            raise ValueError("unsupported Driving DQN checkpoint version")
        required = {"config", "online_network", "target_network"}
        if load_optimizer:
            required.add("optimizer")
        missing = sorted(key for key in required if key not in state)
        if missing:
            raise ValueError("Driving DQN checkpoint is missing: " + ", ".join(missing))
        DQNConfig.from_dict(state["config"])
        for name in ("online_network", "target_network"):
            payload = state[name]
            if not isinstance(payload, Mapping):
                raise ValueError(f"checkpoint {name} must be a mapping")
            cls._require_finite_tensors(payload, name=name)
        if load_optimizer:
            optimizer = state["optimizer"]
            if not isinstance(optimizer, Mapping):
                raise ValueError("checkpoint optimizer must be a mapping")
            cls._require_finite_tensors(optimizer, name="optimizer")
        for name in ("environment_steps", "gradient_steps", "target_syncs"):
            cls._checkpoint_counter(name, state.get(name, 0))
        metrics = state.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("checkpoint metrics must be a mapping")
        for name in (
            "last_loss",
            "last_gradient_norm",
            "last_predicted_mean",
            "last_target_mean",
            "last_td_error",
        ):
            cls._checkpoint_finite(name, metrics.get(name, 0.0))
        for name in ("gradient_clip_events", "nonfinite_update_rejections"):
            cls._checkpoint_counter(name, metrics.get(name, 0))
        if cls._checkpoint_counter(
            "gradient_clip_events", metrics.get("gradient_clip_events", 0)
        ) > cls._checkpoint_counter("gradient_steps", state.get("gradient_steps", 0)):
            raise ValueError("checkpoint gradient clips cannot exceed gradient steps")
        last_action = metrics.get("last_action")
        saved_config = DQNConfig.from_dict(state["config"])
        if last_action is not None and (
            isinstance(last_action, bool)
            or not isinstance(last_action, (int, np.integer))
            or not 0 <= int(last_action) < saved_config.action_size
        ):
            raise ValueError("checkpoint last_action is invalid")
        q_values = np.asarray(
            metrics.get("last_q_values", [0.0] * saved_config.action_size),
            dtype=np.float32,
        )
        if (
            q_values.shape != (saved_config.action_size,)
            or not np.isfinite(q_values).all()
        ):
            raise ValueError("checkpoint action telemetry is malformed or non-finite")
        counts = metrics.get("action_counts", [0] * saved_config.action_size)
        if (
            not isinstance(counts, (list, tuple))
            or len(counts) != saved_config.action_size
        ):
            raise ValueError("checkpoint action telemetry has an incompatible shape")
        for value in counts:
            cls._checkpoint_counter("action_counts", value)
        if "policy_rng_state" in state:
            probe = random.Random()
            try:
                probe.setstate(state["policy_rng_state"])
            except (TypeError, ValueError) as error:
                raise ValueError("checkpoint policy RNG state is malformed") from error
        return state

    @classmethod
    def _require_finite_tensors(cls, value: object, *, name: str) -> None:
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError(f"checkpoint {name} contains non-finite tensors")
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                cls._require_finite_tensors(nested, name=f"{name}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                cls._require_finite_tensors(nested, name=f"{name}[{index}]")
            return
        if isinstance(value, Real) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise ValueError(f"checkpoint {name} contains non-finite numbers")

    @staticmethod
    def _checkpoint_counter(name: str, value: object) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"checkpoint {name} must be a non-negative integer")
        return int(value)

    @staticmethod
    def _checkpoint_finite(name: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"checkpoint {name} must be finite")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"checkpoint {name} must be finite")
        return numeric

    @classmethod
    def _validate_adam_optimizer(cls, optimizer: torch.optim.Adam) -> None:
        """Reject Adam payloads PyTorch accepts but cannot safely step."""

        for group_index, group in enumerate(optimizer.param_groups):
            lr = cls._checkpoint_finite(f"optimizer group {group_index} lr", group.get("lr"))
            eps = cls._checkpoint_finite(
                f"optimizer group {group_index} eps", group.get("eps")
            )
            weight_decay = cls._checkpoint_finite(
                f"optimizer group {group_index} weight_decay",
                group.get("weight_decay"),
            )
            if lr <= 0.0 or eps <= 0.0 or weight_decay < 0.0:
                raise ValueError("checkpoint optimizer hyperparameters are out of range")
            betas = group.get("betas")
            if not isinstance(betas, (list, tuple)) or len(betas) != 2:
                raise ValueError("checkpoint optimizer betas must contain two values")
            beta_values = [
                cls._checkpoint_finite(
                    f"optimizer group {group_index} beta {index}", value
                )
                for index, value in enumerate(betas)
            ]
            if any(not 0.0 <= value < 1.0 for value in beta_values):
                raise ValueError("checkpoint optimizer betas must be in [0, 1)")

        for parameter, metrics in optimizer.state.items():
            if not metrics:
                continue
            required = {"step", "exp_avg", "exp_avg_sq"}
            missing = sorted(required - set(metrics))
            if missing:
                raise ValueError(
                    "checkpoint optimizer state is incomplete: " + ", ".join(missing)
                )
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name not in metrics:
                    continue
                value = metrics[name]
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                    raise ValueError(
                        "checkpoint optimizer tensor shape is incompatible: "
                        f"{name} must match {tuple(parameter.shape)}"
                    )
                if value.dtype != parameter.dtype:
                    raise ValueError(
                        f"checkpoint optimizer {name} dtype must match its parameter"
                    )
                if not torch.isfinite(value).all():
                    raise ValueError(f"checkpoint optimizer {name} is non-finite")
            step = metrics["step"]
            if isinstance(step, torch.Tensor):
                if step.numel() != 1 or not torch.isfinite(step).all():
                    raise ValueError("checkpoint optimizer step must be one finite scalar")
                step_value = float(step.detach().cpu())
            else:
                step_value = cls._checkpoint_finite("optimizer step", step)
            if step_value < 0.0:
                raise ValueError("checkpoint optimizer step must be non-negative")

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
        config = DQNConfig.from_dict(state["config"])
        if config.observation_size == cls.LEGACY_OBSERVATION_SIZE:
            config = replace(config, observation_size=cls.CURRENT_OBSERVATION_SIZE)
        agent = cls(config)
        agent.load_state_dict(state, load_optimizer=load_optimizer)
        return agent

    @classmethod
    def checkpoint_config_compatible(cls, current: DQNConfig, saved: DQNConfig) -> bool:
        """Whether ``saved`` can load directly or through the 5→9 ray bridge."""

        observations_match = current.observation_size == saved.observation_size
        legacy_rays_can_expand = (
            current.observation_size == cls.CURRENT_OBSERVATION_SIZE
            and saved.observation_size == cls.LEGACY_OBSERVATION_SIZE
        )
        return (
            (observations_match or legacy_rays_can_expand)
            and current.action_size == saved.action_size
            and current.hidden_sizes == saved.hidden_sizes
            and current.algorithm == saved.algorithm
        )

    @classmethod
    def _migrate_legacy_network_state(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        migrated = deepcopy(dict(state))
        key = "layers.0.weight"
        weight = migrated.get(key)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError("legacy checkpoint is missing its first-layer weights")
        if weight.shape[1] != cls.LEGACY_OBSERVATION_SIZE:
            raise ValueError("legacy checkpoint has malformed first-layer weights")
        migrated[key] = cls._expand_legacy_input_tensor(weight)
        return migrated

    @classmethod
    def _migrate_legacy_optimizer_state(
        cls, state: Mapping[str, Any]
    ) -> dict[str, Any]:
        migrated = deepcopy(dict(state))
        groups = list(migrated.get("param_groups", ()))
        optimizer_states = migrated.get("state", {})
        if not groups or not groups[0].get("params") or not optimizer_states:
            return migrated
        first_parameter = groups[0]["params"][0]
        parameter_state = optimizer_states.get(first_parameter)
        if parameter_state is None:
            parameter_state = optimizer_states.get(str(first_parameter))
        if not isinstance(parameter_state, Mapping):
            return migrated
        for name, value in list(parameter_state.items()):
            if (
                isinstance(value, torch.Tensor)
                and value.ndim == 2
                and value.shape[1] == cls.LEGACY_OBSERVATION_SIZE
            ):
                parameter_state[name] = cls._expand_legacy_input_tensor(value)
        return migrated

    @classmethod
    def _expand_legacy_input_tensor(cls, tensor: torch.Tensor) -> torch.Tensor:
        expanded = tensor.new_zeros((tensor.shape[0], cls.CURRENT_OBSERVATION_SIZE))
        base = cls.OBSERVATION_BASE_FEATURES
        expanded[:, :base] = tensor[:, :base]
        for legacy_index, current_index in enumerate(cls.LEGACY_RAY_INDICES):
            expanded[:, base + current_index] = tensor[:, base + legacy_index]
        return expanded

    @staticmethod
    def read_checkpoint(path: str | Path) -> dict[str, Any]:
        """Read a validated checkpoint payload for higher-level runtimes."""

        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Driving DQN checkpoint not found: {checkpoint}")
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older Torch
            payload = torch.load(checkpoint, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError("Driving DQN checkpoint must contain a mapping")
        return dict(payload)

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
