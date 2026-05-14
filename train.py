"""Main training script for the battery SOC project.

Run from the repository root:
    python train.py
    python train.py --model simple_mlp
    python train.py --model cnn
    python train.py --model indrnn
    python train.py --model neural_ode
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import load_all_files
from src.data.preprocessor import BatteryPreprocessor
from src.models.cnn_ukf import CNNModel
from src.models.indrnn import IndRNNSOC
from src.models.neural_ode import NeuralODESOC
from src.models.simple_mlp import SimpleMLP
from src.training.trainer import Trainer
from src.utils.config import DataConfig, ModelConfig, Paths, TrainingConfig


def prepare_data(debug: bool = False) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Load raw files, split by filename, fit scaler, create windows, and return loaders."""
    all_data = load_all_files()
    file_names = sorted(all_data.keys())

    n_files = len(file_names)
    if n_files == 0:
        raise RuntimeError("Not enough files. Download all driving cycles at all temperatures.")

    train_end = int(n_files * 0.70)
    val_end = train_end + int(n_files * 0.15)

    train_names = file_names[:train_end]
    val_names = file_names[train_end:val_end]
    test_names = file_names[val_end:]

    if not val_names or not test_names:
        raise RuntimeError("Not enough files. Download all driving cycles at all temperatures.")

    train_dfs = [all_data[name] for name in train_names]
    val_dfs = [all_data[name] for name in val_names]
    test_dfs = [all_data[name] for name in test_names]

    preprocessor = BatteryPreprocessor(window_size=DataConfig.WINDOW_SIZE, step_size=DataConfig.STEP_SIZE)
    preprocessor.fit_scaler(train_dfs)
    preprocessor.save_scaler()

    def build_split(dataframes: list[torch.Tensor | np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        X_parts = []
        y_parts = []
        for df in dataframes:
            features, soc = preprocessor.transform(df)
            if len(features) < DataConfig.WINDOW_SIZE + 1:
                continue
            X_split, y_split = preprocessor.create_windows(features, soc)
            X_parts.append(X_split)
            y_parts.append(y_split)
        X = np.concatenate(X_parts, axis=0) if X_parts else np.empty((0, DataConfig.WINDOW_SIZE, DataConfig.INPUT_DIM), dtype=np.float32)
        y = np.concatenate(y_parts, axis=0) if y_parts else np.empty((0,), dtype=np.float32)
        return X, y

    X_train, y_train = build_split(train_dfs)
    X_val, y_val = build_split(val_dfs)
    X_test, y_test = build_split(test_dfs)

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise RuntimeError("Not enough files. Download all driving cycles at all temperatures.")

    if debug:
        max_debug_samples = TrainingConfig.BATCH_SIZE
        X_train = X_train[:max_debug_samples]
        y_train = y_train[:max_debug_samples]
        X_val = X_val[:max_debug_samples]
        y_val = y_val[:max_debug_samples]
        X_test = X_test[:max_debug_samples]
        y_test = y_test[:max_debug_samples]

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise RuntimeError("Not enough files. Download all driving cycles at all temperatures.")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader, X_test, y_test


def main(model_type: str, debug: bool = False) -> None:
    train_loader, val_loader, test_loader, X_test, y_test = prepare_data(debug=debug)

    if model_type == "simple_mlp":
        model = SimpleMLP(window_size=DataConfig.WINDOW_SIZE)
    elif model_type == "cnn":
        model = CNNModel(input_channels=ModelConfig.INPUT_DIM, window_size=DataConfig.WINDOW_SIZE)
    elif model_type == "indrnn":
        model = IndRNNSOC(input_size=ModelConfig.INPUT_DIM, hidden_size=256, num_layers=2)
    elif model_type == "neural_ode":
        model = NeuralODESOC()
    else:
        raise ValueError(
            "Invalid model type. Choose from ['simple_mlp', 'cnn', 'indrnn', 'neural_ode']."
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_type}")
    print(f"Trainable parameters: {trainable_params}")

    os.makedirs(Paths.MODELS_SAVED, exist_ok=True)
    trainer = Trainer(model)
    trainer.train(train_loader, val_loader, run_name=model_type, debug=debug)

    # Load the best saved model weights after training finishes.
    checkpoint_path = Paths.best_model_pt(model_type)
    state_dict = torch.load(checkpoint_path, map_location=trainer.device)
    model.load_state_dict(state_dict)
    model.to(trainer.device)
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).to(trainer.device)
        y_pred = model(X_test_tensor)
        y_true = torch.from_numpy(y_test).to(trainer.device)
        rmse = torch.sqrt(torch.nn.functional.mse_loss(y_pred, y_true)).item()

    print(f"Final test RMSE: {rmse:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a battery SOC model.")
    parser.add_argument(
        "--model",
        choices=["simple_mlp", "cnn", "indrnn", "neural_ode"],
        default="simple_mlp",
        help="Model type to train (default: simple_mlp).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a fast sanity check using one epoch and one batch.",
    )
    args = parser.parse_args()
    main(args.model, debug=args.debug)
