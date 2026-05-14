import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from src.evaluation.metrics import compute_metrics, print_metrics


def main() -> None:
    y_true = np.array([0.0, 0.5, 1.0])
    y_pred = np.array([0.01, 0.47, 0.98])

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, model_name="My ELM")


if __name__ == "__main__":
    main()
