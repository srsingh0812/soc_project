# Project Work Summary

## Overview
This repository implements a battery state-of-charge (SOC) estimation pipeline for Panasonic 18650PF data. It includes raw data ingestion, preprocessing, PyTorch model training, evaluation utilities, and an API scaffold for inference.

## Completed Work

### Environment
- Python virtual environment created at `battery_soc_env`
- Dependencies installed from `requirements.txt`
- Verified runtime in the venv

### Core Pipeline
- `src/data/loader.py` — raw data ingestion, SOC/Ah handling, data validation
- `src/data/preprocessor.py` — feature normalization and sliding-window generation
- `src/data/dataset.py` — PyTorch dataset utilities and DataLoader support
- `src/utils/config.py` — shared configuration values and project paths

### Models and Training
- `src/models/simple_mlp.py` — simple MLP baseline
- `src/models/cnn_ukf.py` — CNN model plus UKF corrector for hybrid estimation
- `src/models/indrnn.py` — IndRNN recurrent SOC model
- `src/models/neural_ode.py` — Neural ODE SOC model
- `src/models/baseline_elm.py` — NumPy ELM baseline
- `train.py` — main training entrypoint supporting `simple_mlp`, `cnn`, `indrnn`, and `neural_ode`
- `src/training/trainer.py` — training loop and checkpointing logic

### Evaluation and Inference
- `src/evaluation/metrics.py` — RMSE/MAE/MaxError computation, formatted output, plotting, comparison helpers
- `scripts/evaluate_cnn_ukf.py` — test-set evaluation with optional CNN checkpoint path
- `scripts/test_metrics.py` — sanity check for the evaluation utilities
- `api/main.py` — FastAPI scaffold for serving predictions

## Current Status
- Raw data ingestion and SOC/Ah handling implemented
- Preprocessing and window generation validated
- PyTorch training pipeline available and runnable
- CNN+UKF hybrid evaluation path implemented
- Evaluation utilities in place for model comparison
- API server scaffold ready for future inference deployment

## Key Fixes
- Added robust loader behavior for missing or malformed SOC/Ah columns
- Added `Ah` to SOC conversion when SOC is missing
- Fixed `scripts/` import paths using `sys.path` insertion
- Added flexible `--model-path` handling in `scripts/evaluate_cnn_ukf.py`

## How to Use
1. Activate the virtual environment:
```powershell
cd c:\Users\rahul\Documents\GitHub\soc_project
.\battery_soc_env\Scripts\Activate.ps1
```
2. Install dependencies:
```powershell
python -m pip install -r requirements.txt
```
3. Train a model:
```powershell
python train.py --model cnn
```
4. Evaluate CNN+UKF:
```powershell
python scripts/evaluate_cnn_ukf.py
```
5. Run metric sanity check:
```powershell
python scripts/test_metrics.py
```
6. Start the API:
```powershell
uvicorn api.main:app --reload --port 8000
```
