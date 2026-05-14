"""Benchmark saved SOC models and compare test-set performance.

This script loads saved checkpoint files for each supported model type and
computes standard battery SOC metrics on the held-out test split.

Usage:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --models neural_ode cnn indrnn
    python scripts/benchmark_models.py --save-csv benchmark_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from src.data.loader import load_all_files
from src.data.preprocessor import BatteryPreprocessor
from src.evaluation.metrics import compare_all_models, compute_metrics, print_metrics
from src.models.cnn_ukf import CNNModel
from src.models.indrnn import IndRNNSOC
from src.models.neural_ode import NeuralODESOC
from src.models.simple_mlp import SimpleMLP
from src.utils.config import DataConfig, ModelConfig, Paths

MODEL_TYPES = ["simple_mlp", "cnn", "indrnn", "neural_ode"]


def split_file_names(file_names: list[str]) -> tuple[list[str], list[str], list[str]]:
    train_end = int(len(file_names) * 0.70)
    val_end = train_end + int(len(file_names) * 0.15)
    return file_names[:train_end], file_names[train_end:val_end], file_names[val_end:]


def build_windows_from_dfs(preprocessor: BatteryPreprocessor, dfs: list[object]) -> tuple[np.ndarray, np.ndarray]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for df in dfs:
        features, soc = preprocessor.transform(df)
        if len(features) < DataConfig.WINDOW_SIZE + 1:
            continue
        X_split, y_split = preprocessor.create_windows(features, soc)
        X_parts.append(X_split)
        y_parts.append(y_split)
    X = np.concatenate(X_parts, axis=0) if X_parts else np.empty((0, DataConfig.WINDOW_SIZE, DataConfig.INPUT_DIM), dtype=np.float32)
    y = np.concatenate(y_parts, axis=0) if y_parts else np.empty((0,), dtype=np.float32)
    return X, y


def instantiate_model(model_type: str) -> torch.nn.Module:
    if model_type == "simple_mlp":
        return SimpleMLP(window_size=DataConfig.WINDOW_SIZE)
    if model_type == "cnn":
        return CNNModel(input_channels=ModelConfig.INPUT_DIM, window_size=DataConfig.WINDOW_SIZE)
    if model_type == "indrnn":
        return IndRNNSOC(input_size=ModelConfig.INPUT_DIM, hidden_size=256, num_layers=2)
    if model_type == "neural_ode":
        return NeuralODESOC()
    raise ValueError(f"Unsupported model type: {model_type}")


def load_checkpoint(model_type: str, model: torch.nn.Module) -> torch.nn.Module:
    checkpoint_path = Paths.best_model_pt(model_type)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found for {model_type}. Expected: {checkpoint_path}. "
            f"Train the model first with python train.py --model {model_type}."
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model_type: str, model: torch.nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    device = torch.device("cpu")
    model.to(device)
    X_tensor = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics, model_name=model_type)
    return metrics


def main(models_to_evaluate: list[str], save_csv: str | None = None) -> None:
    all_data = load_all_files()
    file_names = sorted(all_data.keys())
    _, _, test_names = split_file_names(file_names)

    if not test_names:
        raise RuntimeError("Not enough files. Download all driving cycles at all temperatures.")

    preprocessor = BatteryPreprocessor(window_size=DataConfig.WINDOW_SIZE, step_size=DataConfig.STEP_SIZE)
    preprocessor.load_scaler()

    test_dfs = [all_data[name] for name in test_names]
    X_test, y_test = build_windows_from_dfs(preprocessor, test_dfs)

    if X_test.size == 0 or y_test.size == 0:
        raise RuntimeError("Not enough test windows available. Check the dataset and scaler.")

    results: dict[str, dict[str, float]] = {}
    for model_type in models_to_evaluate:
        print(f"\nEvaluating saved model: {model_type}")
        model = instantiate_model(model_type)
        model = load_checkpoint(model_type, model)
        results[model_type] = evaluate_model(model_type, model, X_test, y_test)

    compare_all_models(results, save_path=save_csv)

    if save_csv is not None:
        print(f"\nSaved benchmark table to: {save_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark saved battery SOC models on the common test split.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_TYPES,
        default=MODEL_TYPES,
        help="List of saved models to benchmark. Defaults to all supported models.",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional CSV path to save the model comparison table.",
    )
    args = parser.parse_args()
    main(args.models, save_csv=args.save_csv)
