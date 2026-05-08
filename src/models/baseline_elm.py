"""Extreme Learning Machine (ELM) for battery SOC estimation.

ELM is a baseline model where:
- Input weights are fixed (random, never trained)
- Only output weights are learned (instantly, using math instead of gradient descent)
- Training is ~1000x faster than neural networks
"""

from __future__ import annotations

import numpy as np


class ELM:
    def __init__(
        self, n_input: int, n_hidden: int, activation: str = "sigmoid"
    ):
        """
        n_input: number of input features (after flattening)
        n_hidden: number of hidden neurons
        activation: 'sigmoid', 'tanh', or 'relu'
        """
        self.n_input = n_input
        self.n_hidden = n_hidden

        # Random input weights — fixed forever, never trained
        self.W_input = np.random.randn(n_input, n_hidden)
        self.b_hidden = np.random.randn(1, n_hidden)

        # Output weights — will be set by fit()
        self.W_output = None

        # Activation function
        if activation == "sigmoid":
            self.activation = lambda x: 1.0 / (1.0 + np.exp(-x))
        elif activation == "tanh":
            self.activation = np.tanh
        elif activation == "relu":
            self.activation = lambda x: np.maximum(0, x)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _compute_hidden(self, X: np.ndarray) -> np.ndarray:
        """
        Compute hidden layer activations.
        X shape: (n_samples, n_input)
        Returns: (n_samples, n_hidden)
        """
        H = self.activation(X @ self.W_input + self.b_hidden)
        return H

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit ELM by computing optimal output weights.

        X_train shape: (n_samples, n_input) — must be 2D and FLAT
        y_train shape: (n_samples,) or (n_samples, 1)
        """
        if X_train.ndim != 2:
            raise ValueError(
                f"X_train must be 2D (n_samples, n_input), got shape {X_train.shape}"
            )

        H = self._compute_hidden(X_train)

        # pinv = Moore-Penrose pseudoinverse. This is one matrix equation that gives
        # the best possible output weights instantly — no loops, no gradient descent.
        self.W_output = np.linalg.pinv(H) @ y_train.reshape(-1, 1)

        print(f"ELM trained. W_output shape: {self.W_output.shape}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        X shape: (n_samples, n_input) — must be 2D and FLAT
        Returns: (n_samples,) predictions clipped to [0.0, 1.0]
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_input), got shape {X.shape}")

        if self.W_output is None:
            raise RuntimeError("ELM not fitted. Call fit() first.")

        H = self._compute_hidden(X)
        output = (H @ self.W_output).flatten()
        return np.clip(output, 0.0, 1.0)
