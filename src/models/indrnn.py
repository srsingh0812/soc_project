"""IndRNN model for battery SOC estimation.

IndRNN (Independently Recurrent Neural Network) uses element-wise multiplication
between hidden states across time steps, which prevents gradient explosion/vanishing
over long sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IndRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Linear(input_size, hidden_size, bias=True)
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size))

        # Initialize weight_hh uniformly between -1/hidden_size and 1/hidden_size
        nn.init.uniform_(self.weight_hh, -1.0 / hidden_size, 1.0 / hidden_size)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_size)
        h_prev: (batch, hidden_size)

        * is element-wise multiply, NOT matrix multiply — this is what makes it IndRNN
        """
        return torch.relu(self.weight_ih(x) + h_prev * self.weight_hh)


class IndRNNSOC(nn.Module):
    def __init__(
        self, input_size: int = 3, hidden_size: int = 256, num_layers: int = 2
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Build RNN cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.cells.append(IndRNNCell(input_size, hidden_size))
            else:
                self.cells.append(IndRNNCell(hidden_size, hidden_size))

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, window_size, input_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden states for all layers
        h = [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

        # Loop over time steps
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)

            # Process through each layer
            for layer_idx, cell in enumerate(self.cells):
                if layer_idx == 0:
                    h[layer_idx] = cell(x_t, h[layer_idx])
                else:
                    h[layer_idx] = cell(h[layer_idx - 1], h[layer_idx])

        # Pass final hidden state through output head
        output = self.fc(h[-1])
        return output.squeeze(dim=-1)


if __name__ == "__main__":
    model = IndRNNSOC(input_size=3, hidden_size=256, num_layers=2)
    fake_input = torch.rand(16, 100, 3)
    output = model(fake_input)
    print(f"input shape: {fake_input.shape}")
    print(f"output shape: {output.shape}")
    print(f"output min: {output.min().item():.4f}")
    print(f"output max: {output.max().item():.4f}")
    print("IndRNNSOC sanity check passed ✓")


"""
Element-wise multiplication (instead of matrix multiplication) helps prevent the
vanishing gradient problem because:

In standard RNNs, hidden states are updated via matrix multiplication: h_t = f(W @ h_{t-1})
When backpropagating through many time steps, gradients are multiplied by the same
weight matrix repeatedly, causing gradients to shrink to zero (vanishing) or explode.

In IndRNN, element-wise multiplication is used: h_t = f(x + h_{t-1} * w)
where w is initialized small (between -1/hidden_size and 1/hidden_size).
Element-wise multiplication means each neuron only scales its own previous state,
not all neurons together. This keeps gradient magnitudes more stable across long
sequences, preventing vanishing/exploding gradients.
"""
