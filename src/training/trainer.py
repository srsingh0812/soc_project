"""Training utilities for battery SOC models.

This module defines training helpers used by the project, including early
stopping and the main PyTorch training loop with MLflow logging.
"""

from __future__ import annotations

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.config import Paths, ModelConfig, TrainingConfig


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.epochs_no_improve = 0
        self.should_stop = False

    def check(self, val_loss: float) -> bool:
        if self.best_loss - val_loss > self.min_delta:
            self.epochs_no_improve = 0
            self.best_loss = val_loss
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device | None = None) -> None:
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=TrainingConfig.LEARNING_RATE,
            weight_decay=TrainingConfig.WEIGHT_DECAY,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=TrainingConfig.SCHEDULER_PATIENCE,
            factor=0.5,
            verbose=True,
        )
        self.loss_fn = nn.MSELoss()
        self.early_stopping = EarlyStopping()
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            y_pred = self.model(X_batch)
            loss = self.loss_fn(y_pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), TrainingConfig.GRAD_CLIP)
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(val_loader.dataset)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        run_name: str = "model1",
        max_epochs: int | None = None,
        debug: bool = False,
    ) -> dict[str, list[float]]:
        effective_epochs = 1 if debug else (max_epochs or TrainingConfig.MAX_EPOCHS)
        with mlflow.start_run(run_name=run_name):
            for epoch in range(1, effective_epochs + 1):
                train_loss = self.train_epoch(train_loader)
                val_loss = self.validate(val_loader)
                self.scheduler.step(val_loss)

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)

                print(
                    f"Epoch {epoch}/{effective_epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
                )

                if val_loss <= min(self.history["val_loss"]):
                    torch.save(self.model.state_dict(), Paths.BEST_MODEL_PT)
                    mlflow.log_artifact(Paths.BEST_MODEL_PT)

                if self.early_stopping.check(val_loss):
                    print("Early stopping triggered.")
                    break

        return self.history


# Training step explanation:
# 1. Move X_batch and y_batch to the selected device so the model and data are on the same hardware.
# 2. Compute y_pred = model(X_batch) to get the model's current prediction for this batch.
# 3. Compute loss_fn(y_pred, y_batch) to measure how far predictions are from targets.
# 4. optimizer.zero_grad() clears previous gradients; this must be done before backward().
# 5. loss.backward() computes gradients of the loss with respect to model parameters.
# 6. clip_grad_norm_ prevents gradients from exploding by limiting the norm of all gradients.
# 7. optimizer.step() updates model parameters using the computed gradients.

# If optimizer.zero_grad() is skipped, gradients accumulate across batches.
# That means each backward() call adds to the existing gradient values,
# causing parameter updates to use stale information from previous batches.
# The optimizer will then perform incorrect weight updates and training becomes unstable.
