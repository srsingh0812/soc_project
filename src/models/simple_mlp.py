"""Simple MLP model for battery SOC estimation.

This is the first model used to verify the data pipeline and training loop
before moving to more complex architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, window_size: int = 100, hidden_dim: int = 256):
        super().__init__()  # Initialize the parent nn.Module state first

        flat_size = window_size * 3
        self.net = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        output = self.net(x)
        return output.squeeze(dim=-1)


if __name__ == "__main__":
    model = SimpleMLP()
    fake_input = torch.rand(16, 100, 3)
    output = model(fake_input)
    print(f"input shape: {fake_input.shape}")
    print(f"output shape: {output.shape}")
    print(f"output min: {output.min().item():.4f}")
    print(f"output max: {output.max().item():.4f}")
    print("SimpleMLP sanity check passed ✓")


"""
nn.Sequential builds a chain of layers and functions that are executed in order.
It is useful because it keeps the model definition concise and readable for simple
feedforward architectures.

Sigmoid squashes the final scalar output to the range 0-1, which is useful for
predicting SOC values expressed as a fraction of battery charge.

flatten(start_dim=1) converts the input from shape (batch, window_size, 3)
into (batch, window_size * 3), which is required because a fully connected MLP
expects a 2D tensor of shape (batch, features).

Run the sanity check from the terminal with:
    python src/models/simple_mlp.py
"""