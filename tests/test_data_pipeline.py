import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import BatteryPreprocessor
from src.utils.config import DataConfig


def make_fake_dataframe(n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    df = pd.DataFrame(
        {
            DataConfig.VOLTAGE_COL: rng.uniform(3.0, 4.2, size=n_rows),
            DataConfig.CURRENT_COL: rng.uniform(-5.0, 5.0, size=n_rows),
            DataConfig.TEMPERATURE_COL: rng.uniform(0.0, 45.0, size=n_rows),
            DataConfig.SOC_COL: np.linspace(0.9, 0.1, num=n_rows, dtype=np.float32),
        }
    )
    return df


def test_scaler_fits_on_training_only() -> None:
    df = make_fake_dataframe(500)
    preprocessor = BatteryPreprocessor(window_size=50, step_size=10)
    preprocessor.fit_scaler([df])

    assert preprocessor.is_fitted is True

    features, soc = preprocessor.transform(df)
    assert features.min() >= -0.1
    assert features.max() <= 1.1
    assert soc.shape == (500,)


def test_soc_not_normalized() -> None:
    df = make_fake_dataframe(500)
    preprocessor = BatteryPreprocessor(window_size=50, step_size=10)
    preprocessor.fit_scaler([df])

    _, soc = preprocessor.transform(df)
    assert soc.min() >= 0.0
    assert soc.max() <= 1.0


def test_window_creation_shapes() -> None:
    df = make_fake_dataframe(500)
    preprocessor = BatteryPreprocessor(window_size=50, step_size=10)
    preprocessor.fit_scaler([df])
    features, soc = preprocessor.transform(df)

    X, y = preprocessor.create_windows(features, soc)
    assert X.ndim == 3
    assert X.shape[1] == 50
    assert X.shape[2] == 3
    assert X.shape[0] == len(y)


def test_no_data_leakage() -> None:
    df = make_fake_dataframe(500)
    preprocessor = BatteryPreprocessor(window_size=50, step_size=10)
    assert preprocessor.is_fitted is False
    with pytest.raises(RuntimeError):
        preprocessor.transform(df)
