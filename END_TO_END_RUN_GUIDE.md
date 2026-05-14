# End-to-End Project Run Guide

This guide walks through running the battery SOC project from start to finish on Windows.

## 1. Navigate to the project folder

```powershell
cd C:\Users\rahul\Documents\GitHub\soc_project
```

What it does:
- Moves you into the repository root where `train.py`, `src/`, `scripts/`, and `README.md` are located.

What you should see:
- Your prompt changes to show the project folder.

Common issue:
- If the folder is not found, verify the path and use `dir` to locate the repository.

## 2. Activate the virtual environment

```powershell
.\battery_soc_env\Scripts\Activate.ps1
```

What it does:
- Activates the Python virtual environment for this repository.

What you should see:
- The prompt should include `(battery_soc_env)`.

Common issue:
- If `Activate.ps1` does not exist, recreate the venv with `python -m venv battery_soc_env`.

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

What it does:
- Updates pip and installs the libraries listed in `requirements.txt`.

What you should see:
- Installation finishes successfully without errors.

Common issue:
- If a package fails to install, check your internet connection and confirm Python 3.11 is active.

## 4. Verify Python and PyTorch basics

```powershell
python -c "import sys, torch; print(sys.version); print('torch', torch.__version__)"
```

What it does:
- Confirms Python and PyTorch are available.

What you should see:
- Python version information and the installed Torch version.

Common issue:
- If `torch` fails to import, reinstall dependencies and confirm the virtual environment is active.

## 5. Run the SimpleMLP sanity check

```powershell
python .\src\models\simple_mlp.py
```

What it does:
- Executes a quick sanity check for the baseline MLP model.

What you should see:
- Input/output shapes and `SimpleMLP sanity check passed ✓`.

Common issue:
- If `ModuleNotFoundError` occurs, make sure you are running from the repo root.

## 6. Start MLflow to monitor training

```powershell
mlflow ui --backend-store-uri mlflow.db --port 5000
```

What it does:
- Starts the MLflow UI so you can monitor loss and model runs.

What you should see:
- The dashboard is available at `http://127.0.0.1:5000`.

Common issue:
- If `mlflow` is not found, install it with `python -m pip install mlflow`.

## 7. Train the Neural ODE model

```powershell
python train.py --model neural_ode
```

What it does:
- Trains the main Neural ODE SOC model, which is the primary project focus.

What you should see:
- Training progress and saved best checkpoint to `models_saved/neural_ode_best.pt`.

Common issue:
- If `torchdiffeq` is missing, install it with `python -m pip install torchdiffeq`.

## 8. Train baseline models (optional)

```powershell
python train.py --model simple_mlp
python train.py --model cnn
python train.py --model indrnn
```

What it does:
- Trains the baseline comparison models that are used to benchmark against Neural ODE.

What you should see:
- Checkpoints saved to `models_saved/simple_mlp_best.pt`, `models_saved/cnn_best.pt`, and `models_saved/indrnn_best.pt`.

Common issue:
- These models are useful as baselines, but the Neural ODE path is the recommended deployment route.

## 9. Benchmark all saved models

```powershell
python scripts/benchmark_models.py
```

What it does:
- Loads saved checkpoints for available models and computes RMSE/MAE metrics on the same held-out test split.

What you should see:
- A comparison table showing which model performs best on the test set.

Common issue:
- Ensure the model checkpoint files exist before running this script.

## 10. Run CNN+UKF evaluation (optional)

```powershell
python .\scripts\evaluate_cnn_ukf.py
```

What it does:
- Evaluates the CNN predictions and applies the UKF corrector.

What you should see:
- CNN and CNN-UKF metric outputs per file and overall.

Common issue:
- If the checkpoint is not found, run CNN training first or provide `--model-path`.

## 11. Export the trained model to ONNX

```powershell
python .\scripts\export_to_onnx.py
```

What it does:
- Converts the trained model into ONNX format for inference.

What you should see:
- ONNX files and any scaler artifacts saved.

Common issue:
- If no model weights are found, train the chosen model first.

## 12. Start the API server

Create a `.env` file with:

```text
SOC_API_KEY=your_secret_api_key_here
```

Then run:

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

What it does:
- Starts the REST API for SOC prediction.

What you should see:
- The server starts successfully and `http://127.0.0.1:8000/docs` is available.

Common issue:
- If `SOC_API_KEY` is missing, the API will fail to start.

## 13. Test the API endpoint

```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -H "X-API-KEY: your_secret_api_key_here" `
  -d '{"readings":[{"voltage":3.7,"current":0.1,"temperature":25.0}, {"voltage":3.7,"current":0.1,"temperature":25.0}, {"voltage":3.7,"current":0.1,"temperature":25.0}, {"voltage":3.7,"current":0.1,"temperature":25.0}, {"voltage":3.7,"current":0.1,"temperature":25.0}, {"voltage":3.7,"current":0.1,"temperature":25.0}, {"voltage":3.7,"current":0.1,"temperature":25.0}, {"voltage":3.7,"current":0.1,"temperature":25.0}, {"voltage":3.7,"current":0.1,"temperature":25.0}, {"voltage":3.7,"current":0.1,"temperature":25.0}]}'
```

What it does:
- Sends a prediction request to the API.

What you should see:
- JSON response with `soc`, `soc_percent`, and `model_ver`.

Common issue:
- If the API returns 503, the model or scaler artifacts are not ready.

## 14. Run all tests

```powershell
python -m pytest tests/ -q
```

What it does:
- Runs the repository test suite.

What you should see:
- `11 passed`.

Common issue:
- If `pytest` is missing, install it with `python -m pip install pytest`.

## 15. Push everything to GitHub

```powershell
git add .
git commit -m "Run project end-to-end and verify pipeline"
git push origin main
```

What it does:
- Saves your changes and publishes them to the remote repository.

What you should see:
- A successful push to GitHub.

Common issue:
- If push fails, check your remote settings and authentication.
