# End-to-End Project Checklist

Use this checklist to confirm each step is complete.

- [ ] Navigate to the repository root
  - `cd C:\Users\rahul\Documents\GitHub\soc_project`
- [ ] Activate the virtual environment
  - `.\battery_soc_env\Scripts\Activate.ps1`
- [ ] Install dependencies
  - `python -m pip install --upgrade pip setuptools wheel`
  - `python -m pip install -r requirements.txt`
- [ ] Confirm Python and PyTorch
  - `python -c "import sys, torch; print(sys.version); print('torch', torch.__version__)"`
- [ ] Run SimpleMLP sanity check
  - `python .\src\models\simple_mlp.py`
- [ ] Start MLflow UI if you want monitoring
  - `mlflow ui --backend-store-uri mlflow.db --port 5000`
- [ ] Train the Neural ODE model
  - `python train.py --model neural_ode`
- [ ] Train baseline models (optional)
  - `python train.py --model simple_mlp`
  - `python train.py --model cnn`
  - `python train.py --model indrnn`
- [ ] Benchmark saved models
  - `python scripts/benchmark_models.py`
- [ ] Run CNN + UKF evaluation (optional)
  - `python .\scripts\evaluate_cnn_ukf.py`
- [ ] Export model to ONNX
  - `python .\scripts\export_to_onnx.py`
- [ ] Start the API server
  - `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
- [ ] Test the API endpoint
  - Use curl or Postman to call `/predict`
- [ ] Run project tests
  - `python -m pytest tests/ -q`
- [ ] Push changes to GitHub
  - `git add .`
  - `git commit -m "Run project end-to-end and verify pipeline"`
  - `git push origin main`
