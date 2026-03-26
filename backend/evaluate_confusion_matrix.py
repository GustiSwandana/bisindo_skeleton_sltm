from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_IMAGE_PATH,
    CONFUSION_MATRIX_PATH,
    METADATA_PATH,
    MODEL_PATH,
    VAL_DIR,
)
from model.evaluation import (
    compute_classification_report,
    compute_confusion_matrix,
    save_confusion_matrix_figure,
)
from utils.dataset import load_split
from utils.hand_skeleton import HandSkeletonExtractor
from utils.serialization import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate confusion matrix from saved BISINDO model.')
    parser.add_argument('--val-dir', type=Path, default=VAL_DIR)
    args = parser.parse_args()

    metadata = load_json(METADATA_PATH)
    class_names = list(metadata['class_names'])
    extractor = HandSkeletonExtractor(static_image_mode=True, max_num_hands=2)
    val_split = load_split(args.val_dir, class_names, extractor)

    if val_split.features.shape[0] == 0:
        raise ValueError('Data validasi kosong setelah preprocessing.')

    model = keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(val_split.features, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    confusion_matrix = compute_confusion_matrix(
        val_split.labels,
        predictions,
        num_classes=len(class_names),
    )
    classification_report = compute_classification_report(confusion_matrix, class_names)

    save_confusion_matrix_figure(confusion_matrix, class_names, CONFUSION_MATRIX_IMAGE_PATH)
    save_json(
        CONFUSION_MATRIX_PATH,
        {
            'class_names': class_names,
            'matrix': confusion_matrix.tolist(),
            'total_samples': int(np.sum(confusion_matrix)),
            'image_path': str(CONFUSION_MATRIX_IMAGE_PATH),
        },
    )
    save_json(CLASSIFICATION_REPORT_PATH, classification_report)

    print(f'Confusion matrix saved: {CONFUSION_MATRIX_PATH}')
    print(f'Confusion matrix image: {CONFUSION_MATRIX_IMAGE_PATH}')
    print(f'Classification report: {CLASSIFICATION_REPORT_PATH}')


if __name__ == '__main__':
    main()
