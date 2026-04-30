"""Central configuration for the CIFAR-10 image classifier project."""

from pathlib import Path


# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

# Model/input configuration
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"
MODEL_PATH = SAVED_MODEL_DIR / "cifar10_model.h5"
TRAINING_CURVES_PATH = OUTPUT_DIR / "training_curves.png"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
