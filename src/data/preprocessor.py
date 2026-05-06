"""
Battery Data Preprocessor for SOC Estimation

WHY we normalize:
- Voltage, current, and temperature have very different scales (e.g., voltage 3-4V, current -10 to +10A, temp -20 to +50C)
- Neural networks train better when features are on similar scales (typically 0-1)
- Normalization prevents features with larger values from dominating the learning

WHY we do NOT normalize SOC:
- SOC is already in a 0-1 range (0% to 100% battery charge)
- It's the target variable we want to predict exactly as-is
- Normalizing it would make predictions harder to interpret

WHY we only fit on training data:
- Fitting the scaler on training data prevents "data leakage"
- If we fit on test data too, the model sees future information it shouldn't have
- This would make test performance look artificially good, but real-world predictions would be worse
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import DataConfig, Paths


class BatteryPreprocessor:
    def __init__(self, window_size=None, step_size=None):
        self.window_size = window_size or DataConfig.WINDOW_SIZE
        self.step_size = step_size or DataConfig.STEP_SIZE
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.feature_cols = [
            DataConfig.VOLTAGE_COL,
            DataConfig.CURRENT_COL,
            DataConfig.TEMPERATURE_COL,
        ]

    def fit_scaler(self, train_dataframes: list[pd.DataFrame]):
        """
        Fit the MinMaxScaler on training data features only.

        NEVER call this with test data — only training data!
        Calling it with test data causes "data leakage" where the model
        sees information it shouldn't have, making test results invalid.
        """
        if not train_dataframes:
            raise ValueError("No training data provided to fit scaler.")

        # Combine all training DataFrames
        combined_df = pd.concat(train_dataframes, ignore_index=True)

        # Fit scaler on the three feature columns only
        self.scaler.fit(combined_df[self.feature_cols])
        self.is_fitted = True

        # Print what the scaler learned
        print("Scaler fitted on training data:")
        for i, col in enumerate(self.feature_cols):
            min_val = self.scaler.data_min_[i]
            max_val = self.scaler.data_max_[i]
            print(f"  {col}: min={min_val:.3f}, max={max_val:.3f}")

    def transform(self, df: pd.DataFrame):
        """
        Transform a DataFrame by normalizing features and extracting SOC.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Scaler not fitted. Call fit_scaler() on training data first."
            )

        # Apply scaler to feature columns
        normalized_features = self.scaler.transform(df[self.feature_cols])

        # SOC stays as-is (not normalized)
        soc = df[DataConfig.SOC_COL].values

        return (
            normalized_features.astype(np.float32),
            soc.astype(np.float32),
        )

    def create_windows(self, features: np.ndarray, soc: np.ndarray):
        """
        Create sliding windows from normalized features and SOC targets.
        """
        n_samples = len(features)
        if n_samples < self.window_size + 1:
            raise ValueError(
                f"Not enough samples ({n_samples}) to create windows of size {self.window_size}. "
                f"Need at least {self.window_size + 1} samples."
            )

        windows = []
        targets = []

        for start in range(0, n_samples - self.window_size, self.step_size):
            # Input window: features[start:start+window_size]
            window = features[start : start + self.window_size]
            # Target: SOC value right after the window
            target = soc[start + self.window_size]

            windows.append(window)
            targets.append(target)

        if not windows:
            raise ValueError(
                f"No windows created. Check window_size={self.window_size}, "
                f"step_size={self.step_size}, and data length={n_samples}."
            )

        X = np.array(windows, dtype=np.float32)  # Shape: [n_windows, window_size, 3]
        y = np.array(targets, dtype=np.float32)  # Shape: [n_windows]

        return X, y

    def save_scaler(self, filepath=None):
        """Save the fitted scaler to disk."""
        filepath = filepath or Paths.SCALER_PATH
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler.")
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")

    def load_scaler(self, filepath=None):
        """Load a fitted scaler from disk."""
        filepath = filepath or Paths.SCALER_PATH
        self.scaler = joblib.load(filepath)
        self.is_fitted = True
        print(f"Scaler loaded from {filepath}")