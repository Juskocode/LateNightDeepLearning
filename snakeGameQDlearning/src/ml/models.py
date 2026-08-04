import os
from pathlib import Path
import math
from numbers import Integral, Real
import tempfile
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from snakeGameQDlearning.src.config.settings import MODEL_DIR


class CheckpointQNetwork(nn.Module):
    """Common checkpoint contract shared by the neural Snake policies."""

    architecture = "q_network"

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        for name, value in (
            ("input_size", input_size),
            ("hidden_size", hidden_size),
            ("output_size", output_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_size < 2:
            raise ValueError("hidden_size must be at least 2")
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)

    def save(
        self, filename: str = "model.pth", model_dir: Optional[str] = None
    ) -> None:
        if model_dir is None:
            model_dir = MODEL_DIR

        filepath = Path(model_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
        }
        self._validate_checkpoint_state(state)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{filepath.name}.", suffix=".tmp", dir=str(filepath.parent)
        )
        os.close(descriptor)
        try:
            torch.save(state, temporary_name)
            os.replace(temporary_name, filepath)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        print(f"Model saved to {filepath}")

    def load(
        self, filename: str = "model.pth", model_dir: Optional[str] = None
    ) -> None:
        if model_dir is None:
            model_dir = MODEL_DIR

        filepath = Path(model_dir) / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        state = torch.load(filepath, map_location="cpu", weights_only=True)
        self._validate_checkpoint_state(state)
        self.load_state_dict(state)
        self.eval()
        print(f"Model loaded from {filepath}")

    def _validate_checkpoint_state(self, state: dict) -> None:
        """Validate the full payload before ``load_state_dict`` can mutate us."""

        if not isinstance(state, dict) or not state:
            raise ValueError("checkpoint must contain a non-empty state dictionary")
        expected = self.state_dict()
        if set(state) != set(expected):
            missing = sorted(set(expected) - set(state))
            unexpected = sorted(set(state) - set(expected))
            raise ValueError(
                f"checkpoint keys do not match model (missing={missing}, unexpected={unexpected})"
            )
        for name, expected_tensor in expected.items():
            value = state[name]
            if not torch.is_tensor(value) or not torch.is_floating_point(value):
                raise ValueError(f"checkpoint tensor {name!r} must be floating point")
            if value.shape != expected_tensor.shape:
                raise ValueError(
                    f"checkpoint tensor {name!r} has shape {tuple(value.shape)}; "
                    f"expected {tuple(expected_tensor.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"checkpoint tensor {name!r} contains non-finite values")


class LinearQNet(CheckpointQNetwork):
    """Three-layer Q network with parameter-free dropout regularization.

    Dropout adds useful training-time regularization without changing the state
    dictionary, so all current three-layer checkpoints remain compatible.
    """

    architecture = "mlp"

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.05
    ):
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(float(dropout))
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        super().__init__(input_size, hidden_size, output_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = self.linear3(x)
        return x

    def load(
        self, filename: str = "model.pth", model_dir: Optional[str] = None
    ) -> None:
        if model_dir is None:
            model_dir = MODEL_DIR

        filepath = Path(model_dir) / filename
        if filepath.exists():
            state = torch.load(filepath, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not state:
                raise ValueError("checkpoint must contain a non-empty state dictionary")
            legacy_layer = state.get("linear2.weight")
            legacy_output = (
                legacy_layer.shape[0]
                if torch.is_tensor(legacy_layer) and legacy_layer.ndim >= 1
                else -1
            )
            migrated = (
                "linear3.weight" not in state
                and legacy_output == self.linear3.out_features
            )
            if migrated:
                state = self._migrate_legacy_state(state)
            self._validate_checkpoint_state(state)
            self.load_state_dict(state)
            self.eval()
            suffix = " (migrated legacy 2-layer checkpoint)" if migrated else ""
            print(f"Model loaded from {filepath}{suffix}")
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")

    def _migrate_legacy_state(self, legacy: dict) -> dict:
        """Embed an old 11→256→3 network into 11→512→256→3.

        The new middle layer is initialized as an identity projection, so the
        migrated network preserves the legacy checkpoint's predictions.
        """
        required = {"linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"}
        if set(legacy) != required:
            raise ValueError("legacy checkpoint keys are malformed")
        if not all(
            torch.is_tensor(legacy[name])
            and torch.is_floating_point(legacy[name])
            and torch.isfinite(legacy[name]).all()
            for name in required
        ):
            raise ValueError("legacy checkpoint contains invalid tensors")
        upgraded = {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }
        old_hidden = legacy["linear1.weight"].shape[0]
        if (
            legacy["linear1.weight"].shape != (old_hidden, self.linear1.in_features)
            or legacy["linear1.bias"].shape != (old_hidden,)
            or legacy["linear2.weight"].shape != (self.linear3.out_features, old_hidden)
            or legacy["linear2.bias"].shape != (self.linear3.out_features,)
        ):
            raise ValueError("legacy checkpoint tensor shapes are malformed")
        if (
            old_hidden > self.linear1.out_features
            or old_hidden > self.linear2.out_features
        ):
            raise ValueError("legacy hidden layer is too large for this model")
        upgraded["linear1.weight"].zero_()
        upgraded["linear1.bias"].zero_()
        upgraded["linear1.weight"][:old_hidden] = legacy["linear1.weight"]
        upgraded["linear1.bias"][:old_hidden] = legacy["linear1.bias"]
        upgraded["linear2.weight"].zero_()
        upgraded["linear2.bias"].zero_()
        upgraded["linear2.weight"][:old_hidden, :old_hidden] = torch.eye(old_hidden)
        upgraded["linear3.weight"].copy_(legacy["linear2.weight"])
        upgraded["linear3.bias"].copy_(legacy["linear2.bias"])
        return upgraded


class DuelingQNet(CheckpointQNetwork):
    """Dueling network that learns state value and action advantage separately."""

    architecture = "dueling_mlp"

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.05
    ):
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(float(dropout))
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        super().__init__(input_size, hidden_size, output_size)
        feature_size = hidden_size // 2
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, feature_size)
        self.value = nn.Linear(feature_size, 1)
        self.advantage = nn.Linear(feature_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.dropout(F.relu(self.linear1(x)))
        features = self.dropout(F.relu(self.linear2(features)))
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
