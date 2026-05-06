from pathlib import Path

import pandas as pd
from scipy.io import loadmat

from src.utils.config import DataConfig, Paths


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in df.columns:
        normalized = col.upper().replace(' ', '').replace('-', '')
        if normalized in [candidate.upper().replace(' ', '').replace('-', '') for candidate in candidates]:
            return col
    return None


def _extract_array(data):
    if hasattr(data, 'flatten'):
        if data.ndim == 2 and data.shape[0] == 1 and data.shape[1] == 1:
            return data[0][0].flatten()
        return data.flatten()
    return data


def load_single_file(filepath: str) -> pd.DataFrame:
    """Load a single Panasonic 18650PF data file into a DataFrame."""
    filepath = str(filepath)
    lower_path = filepath.lower()

    if lower_path.endswith('.csv'):
        df = pd.read_csv(filepath)

        required_cols = [DataConfig.VOLTAGE_COL, DataConfig.CURRENT_COL, DataConfig.TEMPERATURE_COL]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"CSV file {filepath} does not contain required columns {required_cols}. "
                "It may be a metadata or unsupported CSV file."
            )

        soc_candidates = ['SOC', 'SoC', 'StateOfCharge', 'State_Of_Charge', 'State Of Charge']
        soc_col = _find_column(df, soc_candidates)
        if soc_col:
            if soc_col != DataConfig.SOC_COL:
                df = df.rename(columns={soc_col: DataConfig.SOC_COL})
        else:
            ah_candidates = ['Ah', 'AmpHour', 'Amp_Hour', 'Amp-Hour']
            ah_col = _find_column(df, ah_candidates)
            if ah_col is None:
                raise ValueError(f"CSV file {filepath} does not contain SOC or Ah.")
            capacity = DataConfig.BATTERY_CAPACITY_AH
            if capacity <= 0:
                raise ValueError("Battery capacity must be set to compute SOC from Ah values.")
            df[DataConfig.SOC_COL] = df[ah_col] / capacity

        if df[DataConfig.SOC_COL].max() > 1.5:
            df[DataConfig.SOC_COL] = df[DataConfig.SOC_COL] / 100.0

        return df

    if lower_path.endswith('.mat'):
        mat = loadmat(filepath)
        if 'meas' not in mat:
            raise ValueError(f"MAT file {filepath} does not contain 'meas' data.")

        meas = mat['meas']
        if not hasattr(meas, 'dtype') or meas.dtype.names is None:
            raise ValueError(f"MAT file {filepath} does not contain structured 'meas' fields.")

        field_names = set(meas.dtype.names)

        if 'Time' in field_names:
            time_values = _extract_array(meas['Time'])
        elif 'TimeStamp' in field_names:
            time_values = _extract_array(meas['TimeStamp'])
        else:
            raise ValueError(f"MAT file {filepath} is missing a Time/TimeStamp field.")

        voltage_values = _extract_array(meas['Voltage'])
        current_values = _extract_array(meas['Current'])
        temp_values = _extract_array(meas['Battery_Temp_degC'])

        if 'SOC' in field_names:
            soc_values = _extract_array(meas['SOC'])
        elif 'Ah' in field_names:
            ah_values = _extract_array(meas['Ah'])
            capacity = DataConfig.BATTERY_CAPACITY_AH
            if capacity <= 0:
                raise ValueError("Battery capacity must be set to compute SOC from Ah values.")
            soc_values = ah_values / capacity
        else:
            raise ValueError(f"MAT file {filepath} does not contain SOC or Ah.")

        df = pd.DataFrame(
            {
                DataConfig.TIME_COL: time_values,
                DataConfig.VOLTAGE_COL: voltage_values,
                DataConfig.CURRENT_COL: current_values,
                DataConfig.TEMPERATURE_COL: temp_values,
                DataConfig.SOC_COL: soc_values,
            }
        )

        df = df.dropna(how='all')
        if DataConfig.SOC_COL not in df.columns:
            raise ValueError(f"Failed to create SOC column for file {filepath}")

        if df[DataConfig.SOC_COL].max() > 1.5:
            df[DataConfig.SOC_COL] = df[DataConfig.SOC_COL] / 100.0

        return df

    raise ValueError(
        f"Unsupported file type for '{filepath}'. Only .csv and .mat files are supported."
    )


def load_all_files(data_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """Load all .csv and .mat files from the specified data directory."""
    data_dir = Path(data_dir or Paths.DATA_RAW)
    csv_files = sorted(data_dir.rglob('*.csv'))
    mat_files = sorted(data_dir.rglob('*.mat'))
    files = csv_files + mat_files

    if not files:
        raise FileNotFoundError(
            f"No .csv or .mat files found in {data_dir} or its subdirectories. "
            "Please download the Panasonic dataset and place it in the data/raw/panasonic_18650pf folder."
        )

    loaded_data: dict[str, pd.DataFrame] = {}
    for filepath in files:
        filename = filepath.stem
        try:
            df = load_single_file(str(filepath))
            if DataConfig.SOC_COL not in df.columns:
                print(f"Loading {filename}... ERROR: Missing SOC column")
                continue
            loaded_data[filename] = df
            print(f"Loading {filename}... OK {len(df)} samples")
        except Exception as exc:
            print(f"Loading {filename}... ERROR: {exc}")

    return loaded_data
