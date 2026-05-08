"""Training script for Extreme Learning Machine baseline.

This script demonstrates how to:
1. Load and preprocess battery data
2. Create windows (which are 3D)
3. Flatten windows to 2D for ELM
4. Train ELM instantly
5. Evaluate on test set
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path so 'src' module can be found
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data.loader import load_all_files
from src.data.preprocessor import BatteryPreprocessor
from src.models.baseline_elm import ELM
from src.utils.config import DataConfig


def main():
    print("=" * 60)
    print("ELM Baseline Training Script")
    print("=" * 60)

    # Step 1: Load all data
    print("\n[1] Loading battery data...")
    all_data = load_all_files()
    if not all_data:
        print("ERROR: No data loaded. Check that data files exist in data/raw/panasonic_18650pf/")
        return

    # Convert dict to list of DataFrames
    dataframes = list(all_data.values())
    n_files = len(dataframes)
    print(f"Loaded {n_files} files")

    # Step 2: Train-test split
    print("\n[2] Splitting data (70% train, 15% val, 15% test)...")
    train_size = max(1, int(n_files * DataConfig.TRAIN_RATIO))
    val_size = max(1, int(n_files * DataConfig.VAL_RATIO))
    train_dfs = dataframes[:train_size]
    val_dfs = dataframes[train_size : train_size + val_size]
    test_dfs = dataframes[train_size + val_size :]
    print(f"Train: {len(train_dfs)}, Val: {len(val_dfs)}, Test: {len(test_dfs)}")

    # Step 3: Preprocess
    print("\n[3] Creating preprocessor and fitting scaler on training data...")
    preprocessor = BatteryPreprocessor(
        window_size=DataConfig.WINDOW_SIZE, step_size=DataConfig.STEP_SIZE
    )
    preprocessor.fit_scaler(train_dfs)

    # Step 4: Create windows for train, val, test
    print("\n[4] Creating windows...")
    X_train_list, y_train_list = [], []
    train_count = 0
    for df in train_dfs:
        if len(df) < DataConfig.WINDOW_SIZE + 1:
            print(f"  Skipping file with {len(df)} samples (need at least {DataConfig.WINDOW_SIZE + 1})")
            continue
        features, soc = preprocessor.transform(df)
        X, y = preprocessor.create_windows(features, soc)
        X_train_list.append(X)
        y_train_list.append(y)
        train_count += len(X)
        # Limit total training windows to avoid memory issues
        if train_count > 100000:
            print(f"  Reached 100k training windows ({train_count} total), stopping to avoid memory overload")
            break
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    X_val_list, y_val_list = [], []
    val_count = 0
    for df in val_dfs:
        if len(df) < DataConfig.WINDOW_SIZE + 1:
            print(f"  Skipping file with {len(df)} samples (need at least {DataConfig.WINDOW_SIZE + 1})")
            continue
        features, soc = preprocessor.transform(df)
        X, y = preprocessor.create_windows(features, soc)
        X_val_list.append(X)
        y_val_list.append(y)
        val_count += len(X)
        if val_count > 10000:
            print(f"  Reached 10k validation windows, stopping")
            break
    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)

    X_test_list, y_test_list = [], []
    test_count = 0
    for df in test_dfs:
        if len(df) < DataConfig.WINDOW_SIZE + 1:
            print(f"  Skipping file with {len(df)} samples (need at least {DataConfig.WINDOW_SIZE + 1})")
            continue
        features, soc = preprocessor.transform(df)
        X, y = preprocessor.create_windows(features, soc)
        X_test_list.append(X)
        y_test_list.append(y)
        test_count += len(X)
        if test_count > 10000:
            print(f"  Reached 10k test windows, stopping")
            break
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    print(
        f"Train windows: {X_train.shape} (n_samples, window_size, features) => {y_train.shape}",
        f"Val: {X_val.shape}, Test: {X_test.shape}",
        sep="\n",
    )

    # Step 5: FLATTEN for ELM
    print("\n[5] Flattening 3D windows to 2D...")
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    print(f"Train flattened: {X_train_flat.shape}")
    print(f"Val flattened: {X_val_flat.shape}")
    print(f"Test flattened: {X_test_flat.shape}")

    # Step 6: Train ELM (instant!)
    print("\n[6] Training ELM (this should be instant)...")
    n_input = X_train_flat.shape[1]
    elm = ELM(n_input=n_input, n_hidden=256, activation="sigmoid")
    elm.fit(X_train_flat, y_train)

    # Step 7: Predict and evaluate
    print("\n[7] Evaluating on train, val, and test sets...")
    y_train_pred = elm.predict(X_train_flat)
    y_val_pred = elm.predict(X_val_flat)
    y_test_pred = elm.predict(X_test_flat)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Train RMSE: {train_rmse:.6f} | MAE: {train_mae:.6f}")
    print(f"Val   RMSE: {val_rmse:.6f} | MAE: {val_mae:.6f}")
    print(f"Test  RMSE: {test_rmse:.6f} | MAE: {test_mae:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Why is ELM so much faster than a regular neural network?

Regular neural networks (like SimpleMLP and IndRNNSOC) use gradient descent:
- Loop through epochs (often 100-500+)
- Each epoch: forward pass, compute loss, backward pass, update weights
- Many hyperparameters to tune (learning rate, momentum, etc.)
- Training takes seconds to hours depending on data size

ELM is different:
- Input weights are set ONCE (random) and never change
- Only output weights need to be learned
- Output weights are computed using a single matrix equation (pseudoinverse)
- No loops, no gradient descent, no hyperparameter tuning
- Training completes in milliseconds

Tradeoff:
- ELM is fast, but often has worse accuracy than well-tuned neural networks
- ELM is a good BASELINE: if ELM performs poorly on your data, something is wrong
- Neural networks are slower but can learn more complex patterns
- Use ELM for quick sanity checks; use neural networks for production
"""
