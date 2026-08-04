"""Neural-network models with first-class introspection for live rendering."""

from __future__ import annotations

import copy
from collections.abc import Mapping
import math
from numbers import Integral
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PacmanQNetwork(nn.Module):
    """A configurable MLP that maps Pacman observations to action values."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Iterable[int] = (256, 128),
    ) -> None:
        super().__init__()
        raw_hidden_sizes = tuple(hidden_sizes)
        dimensions = (input_size, output_size, *raw_hidden_sizes)
        if any(
            isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0
            for size in dimensions
        ):
            raise ValueError("network dimensions must be positive integers")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_sizes = tuple(int(size) for size in raw_hidden_sizes)
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must contain positive integer sizes")

        dimensions = (self.input_size, *self.hidden_sizes, self.output_size)
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(dimensions, dimensions[1:])
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers[:-1]:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), nonlinearity="relu")
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in else 0
            nn.init.uniform_(layer.bias, -bound, bound)
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for layer in self.layers[:-1]:
            values = F.relu(layer(values))
        return self.layers[-1](values)

    def forward_with_activations(self, inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return Q-values and the real post-activation values at every layer."""

        activations = [inputs]
        values = inputs
        for layer in self.layers[:-1]:
            values = F.relu(layer(values))
            activations.append(values)
        values = self.layers[-1](values)
        activations.append(values)
        return values, tuple(activations)

    @staticmethod
    def _activation_stats(values: torch.Tensor) -> dict[str, float]:
        flat = values.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "l2": 0.0, "active_fraction": 0.0}
        return {
            "min": float(flat.min().cpu()),
            "max": float(flat.max().cpu()),
            "mean": float(flat.mean().cpu()),
            "std": float(flat.std(unbiased=False).cpu()),
            "l2": float(torch.linalg.vector_norm(flat).cpu()),
            "active_fraction": float((flat.abs() > 1e-8).float().mean().cpu()),
        }

    @staticmethod
    def _selected_indices(values: torch.Tensor, limit: int | None) -> list[int]:
        count = int(values.numel())
        if limit is None or count <= limit:
            return list(range(count))
        strongest = torch.topk(values.detach().abs(), k=limit, sorted=False).indices
        return sorted(int(index) for index in strongest.cpu().tolist())

    def network_snapshot(
        self,
        state: torch.Tensor | Any,
        max_neurons_per_layer: int | None = 16,
    ) -> dict[str, Any]:
        """Build a JSON-friendly graph from actual activations and parameters.

        Large hidden layers are represented by their strongest activated nodes.
        ``selected_indices`` and ``full_shape`` make the sampling explicit, and
        every returned connection is the exact learned weight between the two
        selected nodes.  Pass ``None`` to include every neuron and weight.
        """

        if max_neurons_per_layer is not None and max_neurons_per_layer <= 0:
            raise ValueError("max_neurons_per_layer must be positive or None")
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape != (1, self.input_size):
            raise ValueError(f"network_snapshot expects one state with shape ({self.input_size},)")
        if not torch.isfinite(tensor).all():
            raise ValueError("network_snapshot state must contain finite values")

        was_training = self.training
        self.eval()
        with torch.no_grad():
            _, batch_activations = self.forward_with_activations(tensor)
        if was_training:
            self.train()
        activations = [values[0].detach() for values in batch_activations]
        names = ["observation", *(f"hidden_{index + 1}" for index in range(len(self.hidden_sizes))), "q_values"]
        selected = [self._selected_indices(values, max_neurons_per_layer) for values in activations]

        layer_data: list[dict[str, Any]] = []
        for name, values, indices in zip(names, activations, selected):
            selected_tensor = values[torch.as_tensor(indices, device=values.device)]
            layer_data.append(
                {
                    "name": name,
                    # ``size`` describes the matrix included in this payload so
                    # generic renderers can index it directly. ``full_size``
                    # preserves the real architecture width.
                    "size": len(indices),
                    "full_size": int(values.numel()),
                    "selected_indices": indices,
                    "activations": selected_tensor.float().cpu().tolist(),
                    "sampled": len(indices) != values.numel(),
                    "stats": self._activation_stats(values),
                }
            )

        connections: list[dict[str, Any]] = []
        for index, linear in enumerate(self.layers):
            source_indices = torch.as_tensor(selected[index], dtype=torch.long, device=linear.weight.device)
            target_indices = torch.as_tensor(selected[index + 1], dtype=torch.long, device=linear.weight.device)
            weights = linear.weight.detach().index_select(0, target_indices).index_select(1, source_indices)
            biases = linear.bias.detach().index_select(0, target_indices)
            connections.append(
                {
                    "from": names[index],
                    "to": names[index + 1],
                    "full_shape": list(linear.weight.shape),
                    "source_indices": selected[index],
                    "target_indices": selected[index + 1],
                    "weights": weights.float().cpu().tolist(),
                    "biases": biases.float().cpu().tolist(),
                    "sampled": weights.numel() != linear.weight.numel(),
                    "stats": self._activation_stats(linear.weight),
                }
            )

        return {
            "architecture": [self.input_size, *self.hidden_sizes, self.output_size],
            "parameter_count": self.parameter_count,
            "weight_layout": "out_in",
            "layers": layer_data,
            "connections": connections,
            "activations": [layer["activations"] for layer in layer_data],
            "weights": [connection["weights"] for connection in connections],
            "biases": [connection["biases"] for connection in connections],
        }

    def save_weights(self, path: str | Path) -> Path:
        """Atomically save model weights."""

        if any(not torch.isfinite(parameter).all() for parameter in self.parameters()):
            raise ValueError("cannot save non-finite model weights")
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
                temporary = handle.name
            torch.save(self.state_dict(), temporary)
            os.replace(temporary, destination)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    def load_weights(self, path: str | Path, *, strict: bool = True) -> None:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"model file not found: {source}")
        try:
            state = torch.load(source, map_location=self.device, weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older PyTorch
            state = torch.load(source, map_location=self.device)
        if not isinstance(state, Mapping):
            raise ValueError("model checkpoint must be a state-dict mapping")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"model checkpoint entry {name!r} must be a tensor")
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                raise ValueError(f"model checkpoint entry {name!r} contains non-finite values")
        staged = copy.deepcopy(self)
        staged.load_state_dict(state, strict=strict)
        self.load_state_dict(staged.state_dict())


class LinearQNet(PacmanQNetwork):
    """Compatibility constructor matching the original project convention."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=(hidden_size, max(1, hidden_size // 2)),
        )
