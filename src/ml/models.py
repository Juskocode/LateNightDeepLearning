import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from typing import Optional

from src.config.settings import MODEL_DIR


class LinearQNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor with Q-values for each action
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, filename: str = 'model.pth', model_dir: Optional[str] = None) -> None:
        if model_dir is None:
            model_dir = MODEL_DIR

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(model_dir, filename)
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(self, filename: str = 'model.pth', model_dir: Optional[str] = None) -> None:
        if model_dir is None:
            model_dir = MODEL_DIR

        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            self.load_state_dict(torch.load(filepath))
            self.eval()
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")