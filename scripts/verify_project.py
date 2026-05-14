"""Comprehensive project verification script."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
import pandas as pd

print("="*70)
print("COMPREHENSIVE PROJECT VERIFICATION")
print("="*70)

# Test 1: Core imports
print("\n[1/6] Testing core imports...")
try:
    from src.data.loader import load_all_files
    from src.data.preprocessor import BatteryPreprocessor
    from src.models.cnn_ukf import CNNModel, UKFCorrector
    from src.models.simple_mlp import SimpleMLP
    from src.models.indrnn import IndRNNSOC
    from src.models.neural_ode import NeuralODESOC
    from src.evaluation.metrics import compute_metrics, print_metrics
    from src.utils.config import DataConfig, ModelConfig, Paths, TrainingConfig
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: CUDA/GPU
print("\n[2/6] Checking GPU setup...")
try:
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        print("✅ GPU enabled")
    else:
        print("⚠️  Running on CPU (GPU not available)")
except Exception as e:
    print(f"❌ GPU check failed: {e}")

# Test 3: Config values
print("\n[3/6] Verifying configuration...")
try:
    assert DataConfig.WINDOW_SIZE == 100, "WINDOW_SIZE incorrect"
    assert DataConfig.STEP_SIZE == 1, "STEP_SIZE incorrect"
    assert ModelConfig.INPUT_DIM == 3, "INPUT_DIM incorrect"
    assert ModelConfig.HIDDEN_DIM == 64, "HIDDEN_DIM incorrect"
    assert TrainingConfig.BATCH_SIZE == 64, "BATCH_SIZE incorrect"
    print(f"  WINDOW_SIZE: {DataConfig.WINDOW_SIZE}")
    print(f"  INPUT_DIM: {ModelConfig.INPUT_DIM}")
    print(f"  BATCH_SIZE: {TrainingConfig.BATCH_SIZE}")
    print("✅ Configuration valid")
except Exception as e:
    print(f"❌ Config check failed: {e}")
    sys.exit(1)

# Test 4: Model instantiation
print("\n[4/6] Testing model creation...")
try:
    mlp = SimpleMLP(window_size=DataConfig.WINDOW_SIZE)
    cnn = CNNModel(input_channels=ModelConfig.INPUT_DIM, window_size=DataConfig.WINDOW_SIZE)
    indrnn = IndRNNSOC(input_size=ModelConfig.INPUT_DIM, hidden_size=256, num_layers=2)
    ode = NeuralODESOC()
    print(f"  SimpleMLP params: {sum(p.numel() for p in mlp.parameters()):,}")
    print(f"  CNN params: {sum(p.numel() for p in cnn.parameters()):,}")
    print(f"  IndRNN params: {sum(p.numel() for p in indrnn.parameters()):,}")
    print(f"  NeuralODE params: {sum(p.numel() for p in ode.parameters()):,}")
    print("✅ All models created successfully")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    sys.exit(1)

# Test 5: Metrics computation
print("\n[5/6] Testing evaluation metrics...")
try:
    y_true = np.array([0.0, 0.5, 1.0, 0.2, 0.8])
    y_pred = np.array([0.01, 0.48, 0.99, 0.19, 0.82])
    metrics = compute_metrics(y_true, y_pred)
    assert "RMSE" in metrics, "RMSE not in metrics"
    assert "MAE" in metrics, "MAE not in metrics"
    assert "RMSE_percent" in metrics, "RMSE_percent not in metrics"
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  RMSE %: {metrics['RMSE_percent']:.3f}%")
    print("✅ Metrics computation working")
except Exception as e:
    print(f"❌ Metrics test failed: {e}")
    sys.exit(1)

# Test 6: UKF
print("\n[6/6] Testing UKF corrector...")
try:
    ukf = UKFCorrector(Q=1e-4, R=1.0)
    ukf.reset(0.5)
    pred1 = ukf.update(cnn_soc=0.51, current_A=1.0, dt=0.1, capacity_Ah=2.9)
    pred2 = ukf.update(cnn_soc=0.52, current_A=1.0, dt=0.1, capacity_Ah=2.9)
    assert 0.0 <= pred1 <= 1.0, "UKF output out of bounds"
    assert 0.0 <= pred2 <= 1.0, "UKF output out of bounds"
    print(f"  Initial SOC: 0.5")
    print(f"  After step 1: {pred1:.4f}")
    print(f"  After step 2: {pred2:.4f}")
    print("✅ UKF working correctly")
except Exception as e:
    print(f"❌ UKF test failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - PROJECT IS READY")
print("="*70)
