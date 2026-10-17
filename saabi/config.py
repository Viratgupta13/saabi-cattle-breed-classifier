from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

MODEL_PATH = MODEL_DIR / "saabi_mobilenetv2.keras"
CLASSES_PATH = MODEL_DIR / "class_names.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

IMG_SIZE = 224
SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CALIBRATION_PATH = MODEL_DIR / "calibration.json"
TARGET_SELECTIVE_ACCURACY = 0.92  # accuracy we want on the predictions we choose to answer
OOD_PATH = MODEL_DIR / "ood_centroids.npy"
OOD_ACCEPT_FRACTION = 0.95  # keep 95% of in-distribution validation images
TFLITE_PATH = MODEL_DIR / "saabi.tflite"

# Model version for API and reproducibility
MODEL_VERSION = "2.0.0"
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
