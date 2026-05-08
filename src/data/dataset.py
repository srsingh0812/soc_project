"""Dataset and dataloader utilities for battery SOC estimation.

A PyTorch Dataset is a lightweight object that provides access to samples and labels by
index. It standardizes how data is presented to the model during training, validation,
and testing. Converting your arrays into a Dataset enables PyTorch to efficiently fetch
examples and batch them for the training loop.

A DataLoader wraps a Dataset and handles batching, shuffling, and parallel loading.
Batching means grouping a fixed number of samples into one tensor batch so the model
can process many examples at once. Shuffling means randomizing the order of training
samples each epoch to reduce bias and help the model generalize better.

We shuffle training data because the model should not learn from the original sample
order. Validation and test data are not shuffled so evaluation is deterministic and
results are easy to reproduce.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class BatteryDataset(Dataset):
    def __init__(self, X: "np.ndarray", y: "np.ndarray"):
        import numpy as np

        assert len(X) == len(y), (
            f"X and y must have the same number of samples, got {len(X)} and {len(y)}"
        )

        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(np.asarray(X))
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(np.asarray(y))

        self.X = X.to(torch.float32)
        self.y = y.to(torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def __repr__(self) -> str:
        n_samples = len(self)
        window_size = self.X.shape[1] if self.X.ndim > 1 else 1
        n_features = self.X.shape[2] if self.X.ndim > 2 else self.X.shape[1] if self.X.ndim > 1 else 1
        return (
            f"BatteryDataset(n_samples={n_samples}, window_size={window_size}, "
            f"n_features={n_features})"
        )


def create_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size: int = 32,
):
    train_dataset = BatteryDataset(X_train, y_train)
    val_dataset = BatteryDataset(X_val, y_val)
    test_dataset = BatteryDataset(X_test, y_test)

    print(
        "Dataset sizes: ",
        f"train={len(train_dataset)}, ",
        f"val={len(val_dataset)}, ",
        f"test={len(test_dataset)}",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import numpy as np
    X = np.random.rand(16, 100, 3)
    y = np.random.rand(16)
    ds = BatteryDataset(X, y)
    print(f"Dataset: {ds}")
    train_loader, val_loader, test_loader = create_dataloaders(
        X[:10], y[:10], X[10:13], y[10:13], X[13:], y[13:], batch_size=4
    )
    x_batch, y_batch = next(iter(train_loader))
    print(f"train batch shape: {x_batch.shape}, {y_batch.shape}")
    print("BatteryDataset sanity check passed ✓")
