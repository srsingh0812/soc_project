import torch
import torch.nn as nn

from src.models.cnn_ukf import CNNModel
from src.models.indrnn import IndRNNSOC
from src.models.neural_ode import NeuralODESOC
from src.models.simple_mlp import SimpleMLP


def make_fake_batch(batch_size: int = 4, window_size: int = 100) -> torch.Tensor:
    return torch.rand(batch_size, window_size, 3)


def test_simple_mlp_output_shape() -> None:
    model = SimpleMLP(window_size=100)
    x = make_fake_batch()
    y = model(x)
    assert y.shape == (4,)
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_cnn_output_shape() -> None:
    model = CNNModel(input_channels=3, window_size=100)
    x = make_fake_batch()
    y = model(x)
    assert y.shape == (4,)
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_indrnn_output_shape() -> None:
    model = IndRNNSOC(input_size=3, hidden_size=256, num_layers=2)
    x = make_fake_batch()
    y = model(x)
    assert y.shape == (4,)
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_neural_ode_output_shape() -> None:
    model = NeuralODESOC()
    x = make_fake_batch()
    y = model(x)
    assert y.shape == (4,)
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_neural_ode_trains_without_nan() -> None:
    model = NeuralODESOC()
    x = make_fake_batch()
    target = torch.rand(4)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_lstm_inline() -> None:
    lstm = nn.LSTM(input_size=3, hidden_size=32, num_layers=1, batch_first=True)
    x = make_fake_batch()
    output, _ = lstm(x)
    assert output.shape == (4, 100, 32)


def test_gru_inline() -> None:
    gru = nn.GRU(input_size=3, hidden_size=32, num_layers=1, batch_first=True)
    x = make_fake_batch()
    output, _ = gru(x)
    assert output.shape == (4, 100, 32)
