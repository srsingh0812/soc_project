"""CNN + UKF (linear Kalman) model for battery SOC estimation.

This module implements the CNN-UKF approach described in Paper 1 (Ma et al. 2024).
The CNN produces a SOC estimate from a short history window, and the UKF-like
corrector smooths that estimate using simple Coulomb-counting physics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, input_channels: int = 3, window_size: int = 100) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        flat_size = 64 * (window_size // 4)
        self.fc_block = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, window_size, 3)
        x = x.permute(0, 2, 1)  # to (batch, channels, sequence_length)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.flatten(start_dim=1)
        x = self.fc_block(x)
        return x.squeeze(dim=-1)


class UKFCorrector:
    def __init__(self, Q: float = 1e-4, R: float = 1.0) -> None:
        self.Q = Q
        self.R = R
        self.soc_estimate: float | None = None
        self.uncertainty: float | None = None

    def reset(self, initial_soc: float) -> None:
        self.soc_estimate = float(initial_soc)
        self.uncertainty = 0.1

    def update(
        self,
        cnn_soc: float,
        current_A: float,
        dt: float = 0.1,
        capacity_Ah: float = 2.9,
    ) -> float:
        if self.soc_estimate is None:
            self.reset(cnn_soc)
            return float(cnn_soc)

        soc_physics = self.soc_estimate - (current_A * dt) / (capacity_Ah * 3600.0)
        P_pred = self.uncertainty + self.Q

        K = P_pred / (P_pred + self.R)
        self.soc_estimate = soc_physics + K * (cnn_soc - soc_physics)
        self.uncertainty = (1.0 - K) * P_pred

        self.soc_estimate = float(torch.clamp(torch.tensor(self.soc_estimate), 0.0, 1.0).item())
        return self.soc_estimate


if __name__ == "__main__":
    model = CNNModel()
    corrector = UKFCorrector()
    fake_input = torch.rand(4, 100, 3)
    cnn_out = model(fake_input)
    corrected = [corrector.update(float(val), current_A=0.5) for val in cnn_out]
    print(f"cnn_out shape: {cnn_out.shape}")
    print(f"corrected sample: {corrected[:3]}")
