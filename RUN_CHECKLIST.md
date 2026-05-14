# End-to-End Project Checklist

Use this checklist to confirm each step is complete.

- [ ] Navigate to the repository root
  - `cd C:\Users\rahul\Documents\GitHub\soc_project`
- [ ] Activate the virtual environment
  - `.attery_soc_env
dScripts
ame
ame
ame` (use `.attery_soc_env\Scripts\Activate.ps1`)
- [ ] Install dependencies
  - `python -m pip install --upgrade pip setuptools wheel`
  - `python -m pip install -r requirements.txt`
- [ ] Confirm Python and PyTorch
  - `python -c "import sys, torch; print(sys.version); print('torch', torch.__version__)"`
- [ ] Run SimpleMLP sanity check
  - `python .\src\models\simple_mlp.py`
- [ ] Start MLflow UI if you want monitoring
  - `mlflow ui --backend-store-uri mlflow.db --port 5000`
- [ ] Train CNN baseline
  - `python train.py --model cnn`
- [ ] Run CNN + UKF evaluation
  - `python .\scripts\evaluate_cnn_ukf.py`
- [ ] Train IndRNN model
  - `python train.py --model indrnn`
- [ ] Train Neural ODE model
  - `python train.py --model neural_ode`
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
