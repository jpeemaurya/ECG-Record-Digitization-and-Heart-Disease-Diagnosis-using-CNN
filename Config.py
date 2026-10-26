import os

# Project Configuration & Hyperparameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Diagnostic Classes (6 Standard Classes)
CLASSES = [
    "Normal Sinus Rhythm (NSR)",
    "Atrial Fibrillation (AFib)",
    "Premature Ventricular Contractions (PVC)",
    "Bradycardia (Br)",
    "Ventricular Tachycardia (VT)",
    "Others"
]
NUM_CLASSES = len(CLASSES)

# Signal and Digitization Parameters
SAMPLING_RATE = 360  # Hz
TARGET_SIGNAL_LENGTH = 5000
WINDOW_DURATION = 1.5  # seconds
OVERLAP = 0.5

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15