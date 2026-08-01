import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from snakeGameQDlearning.src.config.settings import MODEL_DIR


class CheckpointQNetwork(nn.Module):
    """Common checkpoint contract shared by the neural Snake policies."""

    architecture = "q_network"

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def save(
        self, filename: str = "model.pth", model_dir: Optional[str] = None
    ) -> None:
        if model_dir is None:
            model_dir = MODEL_DIR

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(str(model_dir), filename)
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(
        self, filename: str = "model.pth", model_dir: Optional[str] = None
    ) -> None:
        if model_dir is None:
            model_dir = MODEL_DIR

        filepath = os.path.join(str(model_dir), filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        state = torch.load(filepath, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        self.eval()
        print(f"Model loaded from {filepath}")


class LinearQNet(CheckpointQNetwork):
    """Three-layer Q network with parameter-free dropout regularization.

    Dropout adds useful training-time regularization without changing the state
    dictionary, so all current three-layer checkpoints remain compatible.
    """

    architecture = "mlp"

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.05
    ):
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

        filepath = os.path.join(str(model_dir), filename)
        if os.path.exists(filepath):
            state = torch.load(filepath, map_location="cpu", weights_only=True)
            legacy_output = state.get("linear2.weight", torch.empty(0)).shape[0]
            migrated = (
                "linear3.weight" not in state
                and legacy_output == self.linear3.out_features
            )
            if migrated:
                state = self._migrate_legacy_state(state)
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
        upgraded = self.state_dict()
        old_hidden = legacy["linear1.weight"].shape[0]
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
