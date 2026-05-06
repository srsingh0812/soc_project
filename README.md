# Battery SOC Estimation Project

This project estimates the state of charge (SOC) for electric vehicle (EV) batteries using a Neural Ordinary Differential Equation (Neural ODE) model. It includes data preparation, model training, and an API for inference.

## What this project does

- Loads battery data from Panasonic 18650PF tests
- Processes the data into time-series windows
- Trains machine learning models to predict SOC
- Supports multiple model types, including:
  - `simple_mlp`
  - `cnn`
  - `neural_ode`
- Serves a prediction API using FastAPI and Uvicorn

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd soc_project
```

2. Create a virtual environment:

```bash
py -3.11 -m venv battery_soc_env
```

3. Activate the virtual environment:

```powershell
.\battery_soc_env\Scripts\Activate.ps1
```

4. Install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Dataset location

Place the Panasonic 18650PF dataset inside:

```text
data/raw/panasonic_18650pf
```

The code expects raw data files to be available under that folder.

## Training

Start training with one of the supported models:

```bash
python train.py --model simple_mlp
python train.py --model cnn
python train.py --model neural_ode
```

Use `simple_mlp` to begin, then experiment with `cnn` and `neural_ode`.

## Start the API

Run the API server locally with:

```bash
uvicorn api.main:app --reload --port 8000
```

Then open `http://localhost:8000` or use the API endpoints from a client.

## Results

| Model        | RMSE @25C | RMSE @0C |
|--------------|-----------|----------|
| simple_mlp   | -         | -        |
| cnn          | -         | -        |
| neural_ode   | -         | -        |
