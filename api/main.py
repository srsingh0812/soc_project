"""REST API for battery State of Charge estimation.

A REST API is a web service interface where clients send requests to defined
URLs and receive structured responses. A POST to /predict means the client is
submitting data in the request body and asking the server to return a prediction
based on that input.

The @ symbol before @app.get and @app.post is a Python decorator. It tells FastAPI
"register this function as a handler for this route." In other words, it wraps the
function and connects it to the web server path.
"""

from __future__ import annotations

import os
from typing import List

import joblib
import numpy as np
import onnxruntime
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.utils.config import DataConfig, Paths

load_dotenv()


API_KEY = os.getenv("SOC_API_KEY")
if API_KEY is None:
    raise RuntimeError("SOC_API_KEY must be set in the environment before starting the API.")

app = FastAPI(title="Neural ODE SOC API")

onnx_session: onnxruntime.InferenceSession | None = None
scaler = None


class BatteryReading(BaseModel):
    voltage: float = Field(..., ge=2.5, le=4.5, description="Battery voltage in volts")
    current: float = Field(..., ge=-20.0, le=20.0, description="Battery current in amps")
    temperature: float = Field(..., ge=-30.0, le=60.0, description="Battery temperature in Celsius")


class SOCRequest(BaseModel):
    readings: List[BatteryReading] = Field(
        ...,
        min_items=DataConfig.WINDOW_SIZE,
        description=f"A list of {DataConfig.WINDOW_SIZE} battery readings, in time order.",
    )


class SOCResponse(BaseModel):
    soc: float = Field(..., ge=0.0, le=1.0, description="Estimated state of charge (0-1).")
    soc_percent: float = Field(..., ge=0.0, le=100.0, description="Estimated state of charge as percentage.")
    model_ver: str = Field(..., description="Model version string.")


def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-KEY")) -> str:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


@app.on_event("startup")
def load_model_and_scaler() -> None:
    global onnx_session, scaler
    try:
        scaler = joblib.load(Paths.SCALER_PATH)
        onnx_session = onnxruntime.InferenceSession(Paths.ONNX_PATH)
        print("✓ Scaler and ONNX model loaded successfully")
    except FileNotFoundError as e:
        print(f"⚠ Warning: Could not load model artifacts: {e}")
        print("   The /health and / endpoints will work, but /predict will fail until training completes.")
        print("   Run: python train.py --model neural_ode")
        print("   Then: python scripts/export_to_onnx.py")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "model": "NeuralODE-SOC-v1"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Welcome to the Neural ODE SOC API. POST to /predict with battery readings.",
        "usage": "Send X-API-KEY header and a JSON body with 'readings'.",
    }


@app.post("/predict", response_model=SOCResponse)
def predict(request: SOCRequest, api_key: str = Depends(verify_api_key)) -> SOCResponse:
    if onnx_session is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Train a model first: python train.py --model neural_ode && python scripts/export_to_onnx.py",
        )

    readings = request.readings[-DataConfig.WINDOW_SIZE :]
    arr = np.array(
        [[r.voltage, r.current, r.temperature] for r in readings], dtype=np.float32
    )
    normalized = scaler.transform(arr)
    batch_input = normalized[np.newaxis, :, :]

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: batch_input})[0]
    soc_value = float(np.clip(result.flatten()[0], 0.0, 1.0))

    return SOCResponse(
        soc=soc_value,
        soc_percent=soc_value * 100.0,
        model_ver="NeuralODE-v1",
    )


# To run the API server:
# 1. Create a .env file with:
#    SOC_API_KEY=your_secret_api_key_here
# 2. Start the server from the repository root:
#    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# 3. Open the browser at http://127.0.0.1:8000/docs to use the Swagger UI.
# 4. Call the endpoint from curl:
#    curl -X POST "http://127.0.0.1:8000/predict" \
#      -H "Content-Type: application/json" \
#      -H "X-API-KEY: your_secret_api_key_here" \
#      -d '{"readings": [{"voltage":3.7,"current":0.1,"temperature":25.0}, ...]}'
