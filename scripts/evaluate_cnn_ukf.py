"""Evaluate the CNN + UKF SOC pipeline on the test set.

This script reproduces the main result from Paper 1 (Ma et al. 2024):
CNN alone should be around ~4% RMSE at 0°C, while CNN+UKF should improve
that by roughly an order of magnitude.

Usage:
    python scripts/evaluate_cnn_ukf.py
    python scripts/evaluate_cnn_ukf.py --model-path models_saved/cnn_best.pt
    python scripts/evaluate_cnn_ukf.py --model-path /path/to/custom_cnn.pt
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
from src.evaluation.metrics import compute_metrics, compare_all_models, print_metrics
from src.models.cnn_ukf import CNNModel, UKFCorrector
from src.utils.config import DataConfig, ModelConfig, Paths


def split_file_names(file_names: list[str]) -> tuple[list[str], list[str], list[str]]:
    train_end = int(len(file_names) * 0.70)
    val_end = train_end + int(len(file_names) * 0.15)
    train_names = file_names[:train_end]
    val_names = file_names[train_end:val_end]
    test_names = file_names[val_end:]
    return train_names, val_names, test_names


def get_window_end_currents(df: object) -> np.ndarray:
    raw_current = df[DataConfig.CURRENT_COL].values.astype(np.float32)
    start_positions = np.arange(0, len(raw_current) - DataConfig.WINDOW_SIZE, DataConfig.STEP_SIZE)
    return raw_current[start_positions + DataConfig.WINDOW_SIZE]


def evaluate_file(filename: str, df: object, model: CNNModel, preprocessor: BatteryPreprocessor) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], np.ndarray]:
    features, soc = preprocessor.transform(df)
    X_windows, y_windows = preprocessor.create_windows(features, soc)
    currents = get_window_end_currents(df)

    if len(X_windows) != len(currents):
        raise RuntimeError(
            f"Prediction count ({len(X_windows)}) does not match current count ({len(currents)}) for {filename}."
        )

    model.eval()
    device = torch.device("cpu")
    model.to(device)

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_windows).to(device)
        preds = []
        batch_size = 256
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start : start + batch_size]
            batch_pred = model(batch)
            preds.append(batch_pred.cpu().numpy())
        cnn_preds = np.concatenate(preds, axis=0)

    ukf = UKFCorrector()
    ukf_preds = np.empty_like(cnn_preds, dtype=np.float32)
    for idx, (cnn_soc, current_A) in enumerate(zip(cnn_preds, currents)):
        ukf_preds[idx] = ukf.update(float(cnn_soc), float(current_A), dt=DataConfig.SAMPLING_RATE)

    metrics_cnn = compute_metrics(y_windows, cnn_preds)
    metrics_ukf = compute_metrics(y_windows, ukf_preds)

    predictions = {
        "CNN": cnn_preds,
        "CNN-UKF": ukf_preds,
    }
    return {
        "CNN": metrics_cnn,
        "CNN-UKF": metrics_ukf,
    }, predictions, y_windows


def main(model_path: str | None = None) -> None:
    all_data = load_all_files()
    file_names = sorted(all_data.keys())
    _, _, test_names = split_file_names(file_names)

    if not test_names:
        raise RuntimeError("Not enough files. Download all driving cycles at all temperatures.")

    preprocessor = BatteryPreprocessor(window_size=DataConfig.WINDOW_SIZE, step_size=DataConfig.STEP_SIZE)
    preprocessor.load_scaler()

    model = CNNModel(input_channels=ModelConfig.INPUT_DIM, window_size=DataConfig.WINDOW_SIZE)
    
    # Resolve model path: use provided path or default to cnn_best.pt
    if model_path is None:
        model_path = os.path.join(Paths.MODELS_SAVED, "cnn_best.pt")
    
    # Convert relative paths to absolute paths
    if not os.path.isabs(model_path):
        model_path = os.path.join(Paths.MODELS_SAVED, os.path.basename(model_path)) if model_path.count(os.sep) == 0 else os.path.join(str(ROOT_DIR), model_path)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN weights not found at {model_path}. Train CNN first with python train.py --model cnn.")

    print(f"Loading CNN model from: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    total_true = []
    total_cnn = []
    total_ukf = []

    for test_name in test_names:
        df = all_data[test_name]
        file_metrics, file_predictions, y_true_windows = evaluate_file(test_name, df, model, preprocessor)

        print(f"\nResults for {test_name}")
        print_metrics(file_metrics["CNN"], model_name="CNN")
        print_metrics(file_metrics["CNN-UKF"], model_name="CNN-UKF")

        total_true.append(y_true_windows)
        total_cnn.append(file_predictions["CNN"])
        total_ukf.append(file_predictions["CNN-UKF"])

        file_rmse_cnn = file_metrics["CNN"]["RMSE_percent"]
        file_rmse_ukf = file_metrics["CNN-UKF"]["RMSE_percent"]
        improvement = ((file_rmse_cnn - file_rmse_ukf) / file_rmse_cnn * 100.0) if file_rmse_cnn > 0 else 0.0
        print(f"File improvement: {improvement:.2f}% lower RMSE with UKF")

    y_true = np.concatenate(total_true, axis=0)
    y_cnn = np.concatenate(total_cnn, axis=0)
    y_ukf = np.concatenate(total_ukf, axis=0)

    overall_metrics = {
        "CNN": compute_metrics(y_true, y_cnn),
        "CNN-UKF": compute_metrics(y_true, y_ukf),
    }

    print("\nOverall metrics")
    print_metrics(overall_metrics["CNN"], model_name="CNN")
    print_metrics(overall_metrics["CNN-UKF"], model_name="CNN-UKF")

    compare_all_models(overall_metrics)

    percent_improvement = (
        (overall_metrics["CNN"]["RMSE"] - overall_metrics["CNN-UKF"]["RMSE"]) / overall_metrics["CNN"]["RMSE"] * 100.0
        if overall_metrics["CNN"]["RMSE"] > 0
        else 0.0
    )
    print(f"\nTotal RMSE improvement from CNN to CNN-UKF: {percent_improvement:.2f}%")

    print("\nRun after training CNN: python train.py --model cnn")
    print("Then run: python scripts/evaluate_cnn_ukf.py")

    print("\nWhy we update UKF sequentially:")
    print("- UKF has internal state and must process each prediction in time order.")
    print("- Each update depends on the prior SOC estimate, so we cannot collapse all steps into one batch.")
    print("- The current at the window end point is the physics input that drives the next SOC step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CNN + UKF SOC estimation pipeline on test set."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the trained CNN model weights (.pt file). "
             "If not provided, defaults to models_saved/cnn_best.pt. "
             "Can be an absolute path or relative path from project root.",
    )
    args = parser.parse_args()
    main(model_path=args.model_path)
