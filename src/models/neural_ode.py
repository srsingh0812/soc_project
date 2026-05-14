"""Neural ODE function for battery SOC modeling.

In a Neural ODE, instead of a fixed physics equation dh/dt = f_physics(h, t),
we have dh/dt = f_network(h, t) where f_network is a neural network we train.
The ODE solver calls this network many times to integrate from t=0 to t=T.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from src.utils.config import ModelConfig


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int = 64, input_dim: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim),
        )

        # Tanh not ReLU — ODE solvers need smooth derivatives everywhere.
        # ReLU has a kink at 0 which can cause step-size issues in the solver.
        self._external_inputs: torch.Tensor | None = None
        self._time_grid: torch.Tensor | None = None

    def set_external_inputs(self, inputs: torch.Tensor, t_grid: torch.Tensor) -> None:
        """Store battery measurements and the integration time grid.

        inputs shape: (batch, seq_len, 3)
        t_grid shape: (seq_len,)
        """
        self._external_inputs = inputs
        self._time_grid = t_grid

    def _get_inputs_at_time(self, t: torch.Tensor) -> torch.Tensor:
        """Interpolate battery measurements [V, I, T] at arbitrary time t.

        The ODE solver evaluates at arbitrary time points (not just our 0.1s grid),
        so we interpolate between measured values.
        """
        if self._external_inputs is None or self._time_grid is None:
            raise RuntimeError("External inputs and time grid must be set before calling the ODE function.")

        # t is a scalar tensor representing the current solver time.
        # Expand to a 1D tensor for searchsorted compatibility.
        t_scalar = t.reshape(-1)
        idx = torch.searchsorted(self._time_grid, t_scalar)

        # Clamp to valid range for interpolation.
        idx = torch.clamp(idx, 1, self._time_grid.shape[0] - 1)
        idx_lo = idx - 1
        idx_hi = idx

        t_lo = self._time_grid[idx_lo]
        t_hi = self._time_grid[idx_hi]

        inputs_lo = self._external_inputs[:, idx_lo, :]
        inputs_hi = self._external_inputs[:, idx_hi, :]

        # Linear interpolation weight.
        alpha = (t_scalar - t_lo) / (t_hi - t_lo)
        alpha = alpha.unsqueeze(-1)

        # Interpolate between adjacent measurements.
        interpolated = inputs_lo + alpha * (inputs_hi - inputs_lo)
        return interpolated.squeeze(1)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute dh/dt given current state h and time t.

        Args:
            t: scalar tensor (current time)
            h: shape (batch, hidden_dim)

        Returns:
            dh/dt of shape (batch, hidden_dim)
        """
        if self._external_inputs is None or self._time_grid is None:
            raise RuntimeError("External inputs and time grid must be set before calling forward().")

        inputs = self._get_inputs_at_time(t)
        # Concatenate h and the interpolated battery measurements,
        # producing shape (batch, hidden_dim + input_dim).
        x = torch.cat([h, inputs], dim=-1)
        dh_dt = self.net(x)
        return dh_dt


class NeuralODESOC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_dim = ModelConfig.INPUT_DIM
        hidden_dim = ModelConfig.HIDDEN_DIM
        output_dim = ModelConfig.OUTPUT_DIM

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
        )
        # Takes ONLY the time step as initial condition.

        self.ode_func = ODEFunc(hidden_dim=hidden_dim, input_dim=input_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )
        # Sigmoid guarantees output is always between 0 and 1.

        self.integrator = odeint_adjoint if ModelConfig.USE_ADJOINT else odeint
        # odeint_adjoint is memory-efficient - it solves the ODE backwards to compute gradients instead of storing all intermediate states.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the NeuralODESOC model.

        Args:
            x: shape (batch, window_size, 3)

        Returns:
            SOC prediction shape (batch,)
        """
        batch, window_size, _ = x.shape
        t_grid = torch.arange(window_size, device=x.device) * ModelConfig.DT

        h0 = self.encoder(x[:, 0, :])
        # We only look at t=0 as starting point. The ODE will figure out the rest.

        self.ode_func.set_external_inputs(x, t_grid)

        h_trajectory = self.integrator(
            self.ode_func,
            h0,
            t_grid,
            method=ModelConfig.ODE_METHOD,
            options={"step_size": ModelConfig.DT},
        )
        # h_trajectory shape: (seq_len, batch, hidden_dim) = h at EVERY time step

        h_final = h_trajectory[-1]
        soc = self.decoder(h_final)
        return soc.squeeze(-1)


if __name__ == "__main__":
    ode_func = ODEFunc()
    batch = 2
    seq_len = 10
    fake_inputs = torch.rand(batch, seq_len, 3)
    time_grid = torch.linspace(0.0, 0.9, seq_len)
    ode_func.set_external_inputs(fake_inputs, time_grid)
    h = torch.rand(batch, ode_func.hidden_dim)
    t = torch.tensor(0.45)
    dh_dt = ode_func(t, h)
    print(f"dh_dt shape: {dh_dt.shape}")

    model = NeuralODESOC()
    x = torch.rand(2, 100, 3)
    soc = model(x)
    print(f"input shape: {x.shape}")
    print(f"output shape: {soc.shape}")
    print(f"output values: {soc.detach().cpu().numpy()}")
    print("NeuralODESOC sanity check passed ✓")
