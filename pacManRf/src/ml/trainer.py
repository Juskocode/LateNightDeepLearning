"""Correct DQN and Double-DQN optimization with measurable training state."""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Mapping, Sequence as SequenceABC
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import Algorithm, DQNConfig, normalize_algorithm
from .models import PacmanQNetwork
from .validation import binary_flag, boolean_mask, require_mapping, strict_int


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    step: int = 0
    batch_size: int = 0
    loss: float = 0.0
    gradient_norm: float = 0.0
    predicted_q_mean: float = 0.0
    target_q_mean: float = 0.0
    bootstrap_mean: float = 0.0
    td_error_mean: float = 0.0
    td_error_abs_mean: float = 0.0
    reward_mean: float = 0.0
    target_synced: bool = False
    gradient_to_clip_ratio: float = 0.0
    gradient_clipped: bool = False
    predicted_q_abs_max: float = 0.0
    target_q_abs_max: float = 0.0
    td_error_abs_max: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DQNTrainer:
    def __init__(self, model: PacmanQNetwork, config: DQNConfig):
        if model.input_size != config.observation_size or model.output_size != config.action_size:
            raise ValueError("model dimensions do not match DQNConfig")
        self.model = model
        self.config = config
        self.algorithm: Algorithm = normalize_algorithm(config.algorithm)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.target_model = copy.deepcopy(model).to(self.device)
        self.target_model.eval()
        self.target_model.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.SmoothL1Loss(reduction="none")
        self.train_steps = 0
        self.last_metrics = TrainingMetrics()
        self.gradient_clip_events = 0
        self.gradient_clip_history_complete = True
        self._recent_gradient_clips: deque[bool] = deque(maxlen=100)

    def synchronize_target(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def _states_tensor(self, values: Any, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.config.observation_size:
            raise ValueError(f"{name} must have shape (batch, {self.config.observation_size})")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite values")
        return tensor

    def _action_indices(self, values: Any, batch_size: int) -> torch.Tensor:
        raw = np.asarray(values)
        if raw.ndim == 0:
            indices = np.asarray(
                [strict_int(raw.item(), "action", minimum=0)],
                dtype=np.int64,
            )
        elif raw.ndim == 2:
            if raw.shape != (batch_size, self.config.action_size):
                raise ValueError("one-hot actions have an unexpected shape")
            masks = [
                boolean_mask(row, self.config.action_size, name="one-hot action")
                for row in raw
            ]
            if any(int(mask.sum()) != 1 for mask in masks):
                raise ValueError("one-hot actions must contain exactly one selected action")
            indices = np.asarray([int(mask.argmax()) for mask in masks], dtype=np.int64)
        elif raw.ndim == 1:
            if raw.size == self.config.action_size and batch_size == 1:
                try:
                    mask = boolean_mask(raw, self.config.action_size, name="one-hot action")
                except ValueError:
                    mask = None
                if mask is not None and int(mask.sum()) == 1:
                    indices = np.asarray([int(mask.argmax())], dtype=np.int64)
                elif raw.size == batch_size:
                    indices = np.asarray(
                        [strict_int(raw[0].item(), "action", minimum=0)],
                        dtype=np.int64,
                    )
                else:
                    raise ValueError("actions must be indices or one-hot vectors")
            elif raw.size == batch_size:
                indices = np.asarray(
                    [strict_int(item.item(), "action", minimum=0) for item in raw],
                    dtype=np.int64,
                )
            else:
                raise ValueError("actions must be indices or one-hot vectors")
        else:
            raise ValueError("actions must be indices or one-hot vectors")
        if indices.shape != (batch_size,):
            raise ValueError("actions must match the state batch size")
        if not np.all((0 <= indices) & (indices < self.config.action_size)):
            raise ValueError("action index is outside the configured action space")
        return torch.as_tensor(indices, dtype=torch.long, device=self.device)

    def _next_action_masks(self, values: Any, batch_size: int) -> torch.Tensor:
        if values is None:
            return torch.ones(
                (batch_size, self.config.action_size),
                dtype=torch.bool,
                device=self.device,
            )
        raw = np.asarray(values)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape != (batch_size, self.config.action_size):
            raise ValueError(
                "next_legal_action_masks must have shape "
                f"(batch, {self.config.action_size})"
            )
        masks = np.stack(
            [
                boolean_mask(
                    row,
                    self.config.action_size,
                    name="next_legal_action_masks row",
                )
                for row in raw
            ]
        )
        return torch.as_tensor(masks, dtype=torch.bool, device=self.device)

    @staticmethod
    def _done_flags(values: Any, batch_size: int) -> torch.Tensor:
        raw = np.asarray(values)
        if raw.ndim == 0:
            raw = raw.reshape(1)
        if raw.ndim != 1 or raw.size != batch_size:
            raise ValueError("done batch size must match states")
        flags = [binary_flag(item.item(), "done") for item in raw]
        return torch.as_tensor(flags, dtype=torch.bool)

    def _bootstrap_values(
        self,
        next_states: torch.Tensor,
        next_legal_action_masks: Any = None,
    ) -> torch.Tensor:
        """Compute the algorithm-specific bootstrap estimate.

        DQN selects and evaluates with the target network. Double-DQN selects
        with the online network and independently evaluates with the target.
        """

        masks = self._next_action_masks(next_legal_action_masks, next_states.shape[0])
        with torch.no_grad():
            target_values = self.target_model(next_states)
            if self.algorithm == "dqn":
                return target_values.masked_fill(~masks, -torch.inf).max(dim=1).values
            online_values = self.model(next_states).masked_fill(~masks, -torch.inf)
            selected = online_values.argmax(dim=1, keepdim=True)
            return target_values.gather(1, selected).squeeze(1)

    def compute_targets(
        self,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_legal_action_masks: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bootstrap = self._bootstrap_values(next_states, next_legal_action_masks)
        targets = rewards + (~dones).float() * self.config.gamma * bootstrap
        return targets, bootstrap

    def update(
        self,
        states: Any,
        actions: Any,
        rewards: Any,
        next_states: Any,
        dones: Any,
        next_legal_action_masks: Any = None,
    ) -> TrainingMetrics:
        state_tensor = self._states_tensor(states, "states")
        next_state_tensor = self._states_tensor(next_states, "next_states")
        if state_tensor.shape[0] != next_state_tensor.shape[0]:
            raise ValueError("states and next_states batch sizes differ")
        batch_size = state_tensor.shape[0]
        action_tensor = self._action_indices(actions, batch_size)
        raw_rewards = np.asarray(rewards)
        if raw_rewards.dtype.kind == "b":
            raise ValueError("rewards must contain finite numbers, not booleans")
        try:
            reward_tensor = torch.as_tensor(
                raw_rewards,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("rewards must contain finite numbers") from error
        done_tensor = self._done_flags(dones, batch_size).to(self.device)
        if reward_tensor.numel() != batch_size or done_tensor.numel() != batch_size:
            raise ValueError("reward and done batch sizes must match states")
        if not torch.isfinite(reward_tensor).all():
            raise ValueError("rewards must contain finite values")
        self.model.train()
        predicted = self.model(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        targets, bootstrap = self.compute_targets(
            next_state_tensor,
            reward_tensor,
            done_tensor,
            next_legal_action_masks,
        )
        per_sample_loss = self.criterion(predicted, targets)
        loss = per_sample_loss.mean()

        for name, tensor in (
            ("predicted Q-values", predicted),
            ("target Q-values", targets),
            ("bootstrap values", bootstrap),
            ("loss", loss),
        ):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"{name} became non-finite; optimizer step was skipped")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("gradient norm became non-finite; optimizer step was skipped")
        gradient_norm_value = float(gradient_norm.detach().cpu())
        gradient_to_clip_ratio = gradient_norm_value / self.config.gradient_clip
        gradient_clipped = gradient_norm_value > self.config.gradient_clip
        self.optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise FloatingPointError("online network became non-finite after optimizer step")
        self.train_steps += 1
        self.gradient_clip_events += int(gradient_clipped)
        self._recent_gradient_clips.append(gradient_clipped)
        target_synced = self.train_steps % self.config.target_update_interval == 0
        if target_synced:
            self.synchronize_target()

        td_error = targets.detach() - predicted.detach()
        self.last_metrics = TrainingMetrics(
            step=self.train_steps,
            batch_size=int(batch_size),
            loss=float(loss.detach().cpu()),
            gradient_norm=gradient_norm_value,
            predicted_q_mean=float(predicted.detach().mean().cpu()),
            target_q_mean=float(targets.detach().mean().cpu()),
            bootstrap_mean=float(bootstrap.detach().mean().cpu()),
            td_error_mean=float(td_error.mean().cpu()),
            td_error_abs_mean=float(td_error.abs().mean().cpu()),
            reward_mean=float(reward_tensor.mean().cpu()),
            target_synced=target_synced,
            gradient_to_clip_ratio=gradient_to_clip_ratio,
            gradient_clipped=gradient_clipped,
            predicted_q_abs_max=float(predicted.detach().abs().max().cpu()),
            target_q_abs_max=float(targets.detach().abs().max().cpu()),
            td_error_abs_max=float(td_error.abs().max().cpu()),
        )
        return self.last_metrics

    def train_step(
        self,
        state: Any,
        action: Any,
        reward: Any,
        next_state: Any,
        done: Any,
        next_legal_action_masks: Any = None,
    ) -> float:
        return self.update(
            state,
            action,
            reward,
            next_state,
            done,
            next_legal_action_masks,
        ).loss

    def predict(self, states: Any, *, target: bool = False) -> np.ndarray:
        tensor = self._states_tensor(states, "states")
        network = self.target_model if target else self.model
        was_training = network.training
        network.eval()
        with torch.no_grad():
            output = network(tensor)
            if not torch.isfinite(output).all():
                raise FloatingPointError("network prediction contains non-finite values")
            values = output.detach().cpu().numpy()
        if was_training and network is self.model:
            network.train()
        return values[0] if np.asarray(states).ndim == 1 else values

    @property
    def target_sync_progress(self) -> float:
        return (self.train_steps % self.config.target_update_interval) / self.config.target_update_interval

    def state_dict(self) -> dict[str, Any]:
        return {
            "online_model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "last_metrics": self.last_metrics.to_dict(),
            "algorithm": self.algorithm,
            "gradient_clip_events": self.gradient_clip_events,
            "gradient_clip_history_complete": self.gradient_clip_history_complete,
            "recent_gradient_clips": list(self._recent_gradient_clips),
        }

    def load_state_dict(self, state: dict[str, Any], *, load_optimizer: bool = True) -> None:
        state = require_mapping(state, "trainer checkpoint")
        saved_algorithm = normalize_algorithm(str(state.get("algorithm", self.algorithm)))
        if saved_algorithm != self.algorithm:
            raise ValueError(f"checkpoint uses {saved_algorithm}, agent uses {self.algorithm}")
        if "online_model" not in state:
            raise ValueError("trainer checkpoint is missing online_model")
        online_model = require_mapping(state["online_model"], "online_model")
        target_model = require_mapping(
            state.get("target_model", online_model),
            "target_model",
        )
        self._assert_finite_payload(online_model, "online_model")
        self._assert_finite_payload(target_model, "target_model")
        optimizer_state = state.get("optimizer")
        if load_optimizer and "optimizer" in state:
            optimizer_state = require_mapping(optimizer_state, "optimizer")
            self._assert_finite_payload(optimizer_state, "optimizer")

        train_steps = strict_int(state.get("train_steps", 0), "train_steps", minimum=0)
        metrics_raw = state.get("last_metrics", {})
        metrics_mapping = require_mapping(metrics_raw, "last_metrics") if metrics_raw else {}
        metrics = self._validated_metrics(metrics_mapping, train_steps)
        clip_history_complete = binary_flag(
            state.get(
                "gradient_clip_history_complete",
                "gradient_clip_events" in state and "recent_gradient_clips" in state,
            ),
            "gradient_clip_history_complete",
        )
        if clip_history_complete and (
            "gradient_clip_events" not in state
            or "recent_gradient_clips" not in state
        ):
            raise ValueError("complete gradient clip history is missing its counters")
        clip_events = strict_int(
            state.get("gradient_clip_events", 0),
            "gradient_clip_events",
            minimum=0,
        )
        if clip_events > train_steps:
            raise ValueError("gradient_clip_events cannot exceed train_steps")
        recent_clips_raw = state.get("recent_gradient_clips", ())
        if isinstance(recent_clips_raw, (str, bytes, bytearray)) or not isinstance(
            recent_clips_raw,
            SequenceABC,
        ):
            raise ValueError("recent_gradient_clips must be a sequence")
        recent_clips = [binary_flag(value, "recent gradient clip") for value in recent_clips_raw]

        candidate_model = copy.deepcopy(self.model)
        candidate_target = copy.deepcopy(self.target_model)
        candidate_model.load_state_dict(online_model)
        candidate_target.load_state_dict(target_model)
        if load_optimizer and "optimizer" in state:
            candidate_optimizer = torch.optim.AdamW(
                candidate_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            candidate_optimizer.load_state_dict(optimizer_state)
            self._validate_adamw_optimizer(candidate_optimizer)

        # Candidate loads above exercise every shape and optimizer invariant.
        # The live objects are mutated only after the complete payload passes.
        self.model.load_state_dict(online_model)
        self.target_model.load_state_dict(target_model)
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(optimizer_state)
            for optimizer_state in self.optimizer.state.values():
                for key, value in optimizer_state.items():
                    if isinstance(value, torch.Tensor):
                        optimizer_state[key] = value.to(self.device)
        self.train_steps = train_steps
        self.last_metrics = metrics
        self.gradient_clip_events = clip_events
        self.gradient_clip_history_complete = clip_history_complete
        self._recent_gradient_clips.clear()
        self._recent_gradient_clips.extend(recent_clips[-self._recent_gradient_clips.maxlen :])
        self.target_model.eval()

    @property
    def recent_gradient_clip_fraction(self) -> float:
        if not self._recent_gradient_clips:
            return 0.0
        return sum(self._recent_gradient_clips) / len(self._recent_gradient_clips)

    @property
    def recent_gradient_clip_window(self) -> int:
        return len(self._recent_gradient_clips)

    def finite_diagnostics(self) -> tuple[bool, list[str]]:
        """Return finite-state status without mutating the learner."""

        invalid: list[str] = []
        for prefix, network in (("online", self.model), ("target", self.target_model)):
            for name, parameter in network.named_parameters():
                if not torch.isfinite(parameter).all():
                    invalid.append(f"{prefix}.{name}")
        for key, value in self.last_metrics.to_dict().items():
            if isinstance(value, Real) and not isinstance(value, (bool, Integral)):
                if not np.isfinite(float(value)):
                    invalid.append(f"metrics.{key}")
        try:
            self._assert_finite_payload(self.optimizer.state_dict(), "optimizer")
            self._validate_adamw_optimizer(self.optimizer)
        except ValueError:
            invalid.append("optimizer")
        return not invalid, invalid

    @staticmethod
    def _assert_finite_payload(value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                raise ValueError(f"{path} contains non-finite tensors")
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                DQNTrainer._assert_finite_payload(nested, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                DQNTrainer._assert_finite_payload(nested, f"{path}[{index}]")
            return
        if isinstance(value, Real) and not isinstance(value, (bool, Integral)):
            if not np.isfinite(float(value)):
                raise ValueError(f"{path} contains non-finite numbers")

    @staticmethod
    def _optimizer_float(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite number")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        return numeric

    @classmethod
    def _validate_adamw_optimizer(cls, optimizer: torch.optim.AdamW) -> None:
        """Reject AdamW state PyTorch accepts but cannot safely step."""

        for group_index, group in enumerate(optimizer.param_groups):
            lr = cls._optimizer_float(group.get("lr"), f"optimizer group {group_index} lr")
            eps = cls._optimizer_float(group.get("eps"), f"optimizer group {group_index} eps")
            decay = cls._optimizer_float(
                group.get("weight_decay"),
                f"optimizer group {group_index} weight_decay",
            )
            if lr <= 0.0 or eps <= 0.0 or decay < 0.0:
                raise ValueError("optimizer hyperparameters are out of range")
            betas = group.get("betas")
            if not isinstance(betas, (list, tuple)) or len(betas) != 2:
                raise ValueError("optimizer betas must contain two values")
            beta_values = [
                cls._optimizer_float(value, f"optimizer group {group_index} beta")
                for value in betas
            ]
            if any(not 0.0 <= value < 1.0 for value in beta_values):
                raise ValueError("optimizer betas must be in [0, 1)")

        for parameter, metrics in optimizer.state.items():
            if not metrics:
                continue
            missing = sorted({"step", "exp_avg", "exp_avg_sq"} - set(metrics))
            if missing:
                raise ValueError("optimizer state is incomplete: " + ", ".join(missing))
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name not in metrics:
                    continue
                value = metrics[name]
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                    raise ValueError(
                        f"optimizer {name} shape must match {tuple(parameter.shape)}"
                    )
                if value.dtype != parameter.dtype or not torch.isfinite(value).all():
                    raise ValueError(f"optimizer {name} must be finite with parameter dtype")
            step = metrics["step"]
            if isinstance(step, torch.Tensor):
                if step.numel() != 1 or not torch.isfinite(step).all():
                    raise ValueError("optimizer step must be one finite scalar")
                step_value = float(step.detach().cpu())
            else:
                step_value = cls._optimizer_float(step, "optimizer step")
            if step_value < 0.0:
                raise ValueError("optimizer step must be non-negative")

    @staticmethod
    def _validated_metrics(values: Mapping[str, Any], train_steps: int) -> TrainingMetrics:
        if not values:
            return TrainingMetrics(step=train_steps)
        fields = TrainingMetrics.__dataclass_fields__
        unknown = set(values) - set(fields)
        if unknown:
            raise ValueError(f"last_metrics contains unknown fields: {', '.join(sorted(unknown))}")
        data = dict(values)
        for name in ("step", "batch_size"):
            if name in data:
                data[name] = strict_int(data[name], f"last_metrics.{name}", minimum=0)
        for name in ("target_synced", "gradient_clipped"):
            if name in data:
                data[name] = binary_flag(data[name], f"last_metrics.{name}")
        for name, value in tuple(data.items()):
            if name in {"step", "batch_size", "target_synced", "gradient_clipped"}:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"last_metrics.{name} must be a finite number")
            data[name] = float(value)
            if not np.isfinite(data[name]):
                raise ValueError(f"last_metrics.{name} must be finite")
        return TrainingMetrics(**data)


class QTrainer(DQNTrainer):
    """Compatibility wrapper accepting the original trainer-style arguments."""

    def __init__(
        self,
        model: PacmanQNetwork,
        learning_rate: float,
        gamma: float,
        algorithm: str = "double_dqn",
        *,
        target_update_interval: int = 1_000,
        gradient_clip: float = 10.0,
    ) -> None:
        config = DQNConfig(
            observation_size=model.input_size,
            action_size=model.output_size,
            hidden_sizes=model.hidden_sizes,
            action_labels=tuple(str(index) for index in range(model.output_size)),
            algorithm=normalize_algorithm(algorithm),
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            gradient_clip=gradient_clip,
        )
        super().__init__(model, config)
