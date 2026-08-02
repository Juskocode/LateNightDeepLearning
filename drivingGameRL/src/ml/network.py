"""Small multilayer Q network with inspectable real activations and weights."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import DQNConfig


class DrivingQNetwork(nn.Module):
    """Fully connected action-value network for the driving observation."""

    def __init__(
        self,
        observation_size: int | DQNConfig = 16,
        action_size: int = 5,
        hidden_sizes: Sequence[int] = (128, 128),
    ):
        super().__init__()
        if isinstance(observation_size, DQNConfig):
            config = observation_size
            observation_size = config.observation_size
            action_size = config.action_size
            hidden_sizes = config.hidden_sizes
        sizes = (
            int(observation_size),
            *(int(size) for size in hidden_sizes),
            int(action_size),
        )
        if len(sizes) < 3 or any(size <= 0 for size in sizes):
            raise ValueError("network layer sizes must be positive")
        self.observation_size = sizes[0]
        self.action_size = sizes[-1]
        self.hidden_sizes = sizes[1:-1]
        self.layers = nn.ModuleList(
            nn.Linear(input_size, output_size)
            for input_size, output_size in zip(sizes, sizes[1:])
        )

    @property
    def architecture(self) -> tuple[int, ...]:
        return (self.observation_size, *self.hidden_sizes, self.action_size)

    @property
    def parameter_count(self) -> int:
        """Number of trainable scalar parameters in the actual network."""

        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for index, layer in enumerate(self.layers):
            values = layer(values)
            if index < len(self.layers) - 1:
                values = torch.relu(values)
        return values

    def snapshot(self, observation: Sequence[float] | np.ndarray) -> dict[str, Any]:
        """Return full, JSON-friendly weights and activations for one state.

        Nothing is synthesized for presentation: every connection weight,
        bias, pre-activation, and displayed activation comes directly from the
        live model. A renderer can down-sample these arrays if space is tight.
        """

        array = np.asarray(observation, dtype=np.float32)
        if array.shape != (self.observation_size,):
            raise ValueError(
                f"observation must have shape ({self.observation_size},), "
                f"got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("observation values must be finite")

        tensor = torch.from_numpy(array)
        layers: list[dict[str, Any]] = [
            {
                "name": "observation",
                "kind": "input",
                "activations": array.astype(float).tolist(),
            }
        ]
        values = tensor
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for index, layer in enumerate(self.layers):
                    raw = layer(values)
                    is_output = index == len(self.layers) - 1
                    values = raw if is_output else torch.relu(raw)
                    layers.append(
                        {
                            "name": "q_values" if is_output else f"hidden_{index + 1}",
                            "kind": "output" if is_output else "hidden",
                            "pre_activations": raw.detach().cpu().tolist(),
                            "activations": values.detach().cpu().tolist(),
                            "weights": layer.weight.detach().cpu().tolist(),
                            "biases": layer.bias.detach().cpu().tolist(),
                        }
                    )
        finally:
            self.train(was_training)

        return {
            "architecture": list(self.architecture),
            "parameter_count": self.parameter_count,
            "layers": layers,
            "q_values": layers[-1]["activations"],
        }

    visualization_snapshot = snapshot
