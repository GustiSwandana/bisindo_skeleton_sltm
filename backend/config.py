from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
ARTIFACTS_DIR = BACKEND_DIR / 'artifacts'


def resolve_dataset_root() -> Path:
    candidates = [
        PROJECT_ROOT / 'dataset',
        PROJECT_ROOT / 'dataset_bisindo',
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / 'train').exists() and (candidate / 'val').exists():
            return candidate
    return PROJECT_ROOT / 'dataset_bisindo'


DATASET_ROOT = resolve_dataset_root()
TRAIN_DIR = DATASET_ROOT / 'train'
VAL_DIR = DATASET_ROOT / 'val'
MODEL_PATH = ARTIFACTS_DIR / 'bisindo_lstm.keras'
METADATA_PATH = ARTIFACTS_DIR / 'label_map.json'
HISTORY_PATH = ARTIFACTS_DIR / 'training_history.json'
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / 'confusion_matrix.json'
CONFUSION_MATRIX_IMAGE_PATH = ARTIFACTS_DIR / 'confusion_matrix.png'
CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / 'classification_report.json'
