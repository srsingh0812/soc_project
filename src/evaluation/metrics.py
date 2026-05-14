"""Evaluation utilities for battery SOC model performance.

This module provides metric computation, printing, plotting, and model
comparison helpers for the SOC estimation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute SOC regression metrics for predictions.

    Both y_true and y_pred should be numpy arrays with values between 0 and 1.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    error = y_true - y_pred
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MaxError": max_error,
        "RMSE_percent": rmse * 100.0,
        "MAE_percent": mae * 100.0,
        "MaxError_percent": max_error * 100.0,
    }
    return metrics


def print_metrics(metrics: dict[str, float], model_name: str = "Model") -> None:
    """Print evaluation metrics in a formatted table with a verdict."""
    print(f"\nMetrics for {model_name}")
    print("-------------------------")
    print(f"RMSE:           {metrics['RMSE']:.6f}")
    print(f"MAE:            {metrics['MAE']:.6f}")
    print(f"MaxError:       {metrics['MaxError']:.6f}")
    print(f"RMSE %:         {metrics['RMSE_percent']:.3f}%")
    print(f"MAE %:          {metrics['MAE_percent']:.3f}%")
    print(f"MaxError %:     {metrics['MaxError_percent']:.3f}%")

    rmse_pct = metrics["RMSE_percent"]
    if rmse_pct < 1.0:
        verdict = "✅ RMSE < 1% — Production quality"
    elif rmse_pct < 2.0:
        verdict = "🟡 RMSE 1-2% — Acceptable prototype"
    else:
        verdict = "❌ RMSE >= 2% — Needs improvement"

    print(f"Verdict:        {verdict}\n")


def plot_soc_comparison(
    time_s: np.ndarray,
    y_true: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
    title: str,
    save_path: str | None = None,
) -> None:
    """Plot SOC curves and absolute prediction error for each model."""
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Top panel: SOC curves in percent
    axs[0].plot(time_s, y_true * 100.0, color="black", linewidth=2.5, label="Ground truth")
    for model_name, y_pred in predictions_dict.items():
        axs[0].plot(time_s, y_pred * 100.0, linestyle="--", label=model_name)

    axs[0].set_ylabel("SOC (%)")
    axs[0].set_title(title)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Bottom panel: absolute error in percent
    for model_name, y_pred in predictions_dict.items():
        axs[1].plot(time_s, np.abs(y_true - y_pred) * 100.0, label=model_name)

    axs[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="1% target")
    axs[1].set_ylabel("Absolute error (%)")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved SOC comparison plot to {save_path}")
    plt.show()


def compare_all_models(results_dict: dict[str, dict[str, float]], save_path: str | None = None) -> pd.DataFrame:
    """Compare metrics for all models and optionally save the table to CSV."""
    df = pd.DataFrame(results_dict).T
    df = df[["RMSE", "MAE", "MaxError", "RMSE_percent", "MAE_percent", "MaxError_percent"]]
    df = df.sort_values(by="RMSE")

    best_model = df.index[0]
    df["Rank"] = range(1, len(df) + 1)
    df.loc[best_model, "Best"] = "⭐ BEST"

    print("\nModel comparison")
    print(df.to_string(float_format="{:.6f}".format))

    if save_path is not None:
        df.to_csv(save_path, index=True)
        print(f"Saved comparison table to {save_path}")

    return df


def plot_training_history(history: dict[str, list[float]], save_path: str | None = None) -> None:
    """Plot training and validation loss over epochs.

    Use log scale on the y-axis because loss values often decrease by orders of
    magnitude during training. A log scale makes it easier to see progress when
    the loss is high and when it becomes very small.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training history plot to {save_path}")
    plt.show()


# Usage example:
#
# from src.evaluation.metrics import compute_metrics, compare_all_models
#
# results = {
#     "Neural ODE": compute_metrics(y_true, y_pred_ode),
#     "CNN-UKF": compute_metrics(y_true, y_pred_cnn_ukf),
#     "IndRNN": compute_metrics(y_true, y_pred_indrnn),
#     "Simple MLP": compute_metrics(y_true, y_pred_mlp),
# }
#
# compare_all_models(results, save_path="model_comparison.csv")
