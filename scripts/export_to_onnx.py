"""Export the trained Neural ODE model to ONNX for production deployment.

ONNX is a universal model format that lets you run the model without PyTorch
installed — it is smaller and faster for production use.

The script:
1. Imports NeuralODESOC and loads the best saved model
2. Creates a dummy input matching real input shape
3. Exports to ONNX using torch.onnx.export:
   - input_names=['battery_sequence']
   - output_names=['soc_prediction']
   - dynamic_axes to allow different batch sizes
   - opset_version=17
   - Save to Paths.ONNX_PATH
4. Verifies the exported model works:
   - Load it with onnxruntime.InferenceSession
   - Run dummy input through it
   - Print the output SOC value

Note: If torch.onnx.export fails with odeint_adjoint, this script temporarily
sets ModelConfig.USE_ADJOINT = False and re-exports. The model behavior is
identical for inference; adjoint only affects training.

Why use ONNX instead of PyTorch in production?
- Smaller file size (less storage and bandwidth)
- Faster inference (optimized runtime)
- No PyTorch dependency required on target machine
- Cross-platform and hardware-agnostic
- Industry standard for production ML deployment
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import onnxruntime

from src.models.neural_ode import NeuralODESOC
from src.utils.config import DataConfig, ModelConfig, Paths


def export_model() -> None:
    model = NeuralODESOC()
    state_dict = torch.load(Paths.BEST_MODEL_PT, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.zeros(1, DataConfig.WINDOW_SIZE, ModelConfig.INPUT_DIM)

    try:
        torch.onnx.export(
            model,
            dummy,
            Paths.ONNX_PATH,
            input_names=["battery_sequence"],
            output_names=["soc_prediction"],
            dynamic_axes={
                "battery_sequence": {0: "batch_size"},
                "soc_prediction": {0: "batch_size"},
            },
            opset_version=17,
        )
    except Exception as exc:
        print("ONNX export failed with the current ODE adjoint setting:", exc)
        print("Retrying with ModelConfig.USE_ADJOINT = False...")
        ModelConfig.USE_ADJOINT = False
        model = NeuralODESOC()
        model.load_state_dict(state_dict)
        model.eval()
        torch.onnx.export(
            model,
            dummy,
            Paths.ONNX_PATH,
            input_names=["battery_sequence"],
            output_names=["soc_prediction"],
            dynamic_axes={
                "battery_sequence": {0: "batch_size"},
                "soc_prediction": {0: "batch_size"},
            },
            opset_version=17,
        )

    print(f"Saved ONNX model to {Paths.ONNX_PATH}")

    session = onnxruntime.InferenceSession(Paths.ONNX_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: dummy.numpy()})
    print(f"\n✅ ONNX export verified — model is ready for deployment")
    print(f"Output SOC value: {result[0]}")
    print(f"Model saved at: {Paths.ONNX_PATH}")


if __name__ == "__main__":
    export_model()

    print("\n" + "="*70)
    print("WHY USE ONNX FOR PRODUCTION?")
    print("="*70)
    print("\nONNX advantages over running PyTorch models directly:")
    print("  • Smaller model file (60-70% size reduction typical)")
    print("  • Faster inference (optimized runtime execution)")
    print("  • No PyTorch dependency on target machine")
    print("  • Cross-platform compatibility (Windows, Linux, macOS, mobile)")
    print("  • Hardware-agnostic (CPU, GPU, NPU, TPU support)")
    print("  • Industry standard for production ML deployment")
    print("  • Compatible with many inference engines (TensorRT, CoreML, etc.)")
    print("\nTo use the exported model in inference:")
    print("  1. Install onnxruntime (smaller than PyTorch)")
    print("  2. Load: session = onnxruntime.InferenceSession('model.onnx')")
    print("  3. Run predictions without PyTorch installed")
