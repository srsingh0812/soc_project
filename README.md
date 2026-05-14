# Battery SOC Estimation Project

This repository estimates battery state-of-charge (SOC) from Panasonic 18650PF test data. It contains data ingestion, preprocessing, PyTorch model training, evaluation utilities, ONNX export, and a FastAPI inference scaffold.

## What this project includes

- Raw data loading and SOC/Ah handling from `data/raw/panasonic_18650pf`
- Feature scaling and sliding-window generation for time-series modeling
- Multiple model types:
  - `simple_mlp`
  - `cnn`
  - `indrnn`
  - `neural_ode`
  - CNN + UKF hybrid evaluation
- Evaluation utilities for RMSE/MAE/MaxError and model comparison
- API server scaffold with ONNX inference support

## Installation

1. Clone the repository:
```powershell
git clone <repository-url>
cd soc_project
```

2. Create a Python 3.11 virtual environment:
```powershell
python -m venv battery_soc_env
```

3. Activate the virtual environment:
```powershell
.\battery_soc_env\Scripts\Activate.ps1
```

4. Install dependencies:
```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Data setup

Place the Panasonic 18650PF raw dataset under:

```text
data/raw/panasonic_18650pf
```

The project expects raw `.mat` and `.csv` files in that directory.

## Training

Train a model with `train.py`:

```powershell
python train.py --model simple_mlp
python train.py --model cnn
python train.py --model indrnn
python train.py --model neural_ode
```

## Evaluation

### CNN + UKF evaluation

Evaluate a trained CNN model and apply UKF correction with:

```powershell
python scripts/evaluate_cnn_ukf.py
```

To use a custom checkpoint:

```powershell
python scripts/evaluate_cnn_ukf.py --model-path models_saved/custom_cnn.pt
```

### Metric sanity check

Run a simple metrics validation script:

```powershell
python scripts/test_metrics.py
```

## ONNX export and API

Export a trained model to ONNX for inference:

```powershell
python scripts/export_to_onnx.py
```

Start the FastAPI server:

1. Create a `.env` file containing:
```text
SOC_API_KEY=your_secret_api_key_here
```

2. Run:
```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

3. Open the Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Example API request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your_secret_api_key_here" \
  -d '{"readings": [{"voltage": 3.7, "current": 0.1, "temperature": 25.0}, ...]}'
```

## Project structure

- `src/data/loader.py` — raw file loading, SOC/Ah conversion, data validation
- `src/data/preprocessor.py` — normalization, window creation, scaler persistence
- `src/data/dataset.py` — PyTorch dataset utilities
- `src/models/` — model implementations
- `src/evaluation/metrics.py` — evaluation helpers and plotting
- `train.py` — model training entrypoint
- `scripts/` — evaluation, export, and utility scripts
- `api/main.py` — FastAPI REST API scaffold

## Notes

- The API requires a trained model and exported ONNX artifact before `/predict` will work.
- `scripts/evaluate_cnn_ukf.py` uses a default checkpoint at `models_saved/cnn_best.pt` unless `--model-path` is supplied.
- The training script uses split ratios roughly 70% train, 15% validation, and 15% test by filename.
