import os


class Paths:
    # Root folder of the project, three levels up from this file
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Folder containing the raw Panasonic 18650PF battery data
    DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw", "panasonic_18650pf")

    # Folder to store processed datasets generated from raw data
    DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

    # Folder to save trained models and other artifacts
    MODELS_SAVED = os.path.join(PROJECT_ROOT, "models_saved")

    # File path for the saved scaler object used to normalize data
    SCALER_PATH = os.path.join(MODELS_SAVED, "scaler.pkl")

    # File path for the best saved PyTorch model weights
    BEST_MODEL_PT = os.path.join(MODELS_SAVED, "neural_ode_best.pt")

    # File path for the exported ONNX model
    ONNX_PATH = os.path.join(MODELS_SAVED, "neural_ode.onnx")


class DataConfig:
    # Number of timesteps used in each input window
    WINDOW_SIZE = 100

    # Number of timesteps to move the window on each step
    STEP_SIZE = 1

    # Time interval in seconds between consecutive samples
    SAMPLING_RATE = 0.1

    # Fraction of the dataset used for training
    TRAIN_RATIO = 0.70

    # Fraction of the dataset used for validation
    VAL_RATIO = 0.15

    # Fraction of the dataset used for testing
    TEST_RATIO = 0.15

    # Column name for voltage values in the dataset
    VOLTAGE_COL = "Voltage(V)"

    # Column name for current values in the dataset
    CURRENT_COL = "Current(A)"

    # Column name for temperature values in the dataset
    TEMPERATURE_COL = "Temperature(C)"

    # Column name for amp-hour values in the dataset
    AH_COL = "Ah"

    # Column name for state-of-charge values in the dataset
    SOC_COL = "SOC"

    # Column name for time values in the dataset
    TIME_COL = "Time(s)"

    # Nominal battery capacity in amp-hours for Panasonic 18650PF cells
    BATTERY_CAPACITY_AH = 2.85


class ModelConfig:
    # Number of input features for the model (voltage, current, temperature)
    INPUT_DIM = 3

    # Number of hidden units in the model layers
    HIDDEN_DIM = 64

    # Number of output features from the model (SOC prediction)
    OUTPUT_DIM = 1

    # Time step used by the ODE solver
    DT = 0.1

    # Whether to use the adjoint method for ODE gradients
    USE_ADJOINT = True

    # ODE integration method to use for the neural ODE
    ODE_METHOD = "rk4"


class TrainingConfig:
    # Number of samples per batch during training
    BATCH_SIZE = 64

    # Learning rate for the optimizer
    LEARNING_RATE = 1e-3

    # L2 regularization strength for the optimizer
    WEIGHT_DECAY = 1e-5

    # Maximum number of training epochs
    MAX_EPOCHS = 200

    # Number of epochs with no improvement before stopping early
    PATIENCE = 20

    # Maximum norm for gradient clipping
    GRAD_CLIP = 1.0

    # Number of epochs with no improvement before reducing the learning rate
    SCHEDULER_PATIENCE = 10

    # Multiplicative factor to reduce the learning rate when plateauing
    SCHEDULER_FACTOR = 0.5

    # Number of worker processes for data loading on Windows
    NUM_WORKERS = 0
