from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from config import (
    ARTIFACTS_DIR,
    CONFUSION_MATRIX_IMAGE_PATH,
    CONFUSION_MATRIX_PATH,
    HISTORY_PATH,
    METADATA_PATH,
    MODEL_PATH,
    TRAIN_DIR,
    VAL_DIR,
)
from model.evaluation import compute_confusion_matrix, save_confusion_matrix_figure
from model.network import build_lstm_classifier
from utils.dataset import discover_class_names, load_split
from utils.hand_skeleton import HandSkeletonExtractor
from utils.serialization import save_json


def train_model(
    train_dir: Path,
    val_dir: Path,
    epochs: int,
    batch_size: int,
) -> dict:
    extractor = HandSkeletonExtractor(static_image_mode=True, max_num_hands=2)
    class_names = discover_class_names(train_dir, val_dir)
    if not class_names:
        raise ValueError('Tidak ada kelas yang ditemukan pada dataset.')

    train_split = load_split(train_dir, class_names, extractor)
    val_split = load_split(val_dir, class_names, extractor)

    if train_split.features.shape[0] == 0:
        raise ValueError('Seluruh data train gagal diproses. Periksa deteksi tangan pada dataset.')
    if val_split.features.shape[0] == 0:
        raise ValueError('Seluruh data validasi gagal diproses. Periksa deteksi tangan pada dataset.')

    model = build_lstm_classifier(
        sequence_length=train_split.features.shape[1],
        feature_dim=train_split.features.shape[2],
        num_classes=len(class_names),
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-5,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_split.features,
        train_split.labels,
        validation_data=(val_split.features, val_split.labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    metrics = model.evaluate(val_split.features, val_split.labels, verbose=0)
    val_probabilities = model.predict(val_split.features, verbose=0)
    val_predictions = np.argmax(val_probabilities, axis=1)
    confusion_matrix = compute_confusion_matrix(
        val_split.labels,
        val_predictions,
        num_classes=len(class_names),
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    save_confusion_matrix_figure(
        confusion_matrix,
        class_names=class_names,
        output_path=CONFUSION_MATRIX_IMAGE_PATH,
    )

    metadata = {
        'class_names': class_names,
        'sequence_length': int(train_split.features.shape[1]),
        'feature_dim': int(train_split.features.shape[2]),
        'train_samples': int(train_split.processed_count),
        'val_samples': int(val_split.processed_count),
        'train_skipped': int(train_split.skipped_count),
        'val_skipped': int(val_split.skipped_count),
        'train_dir': str(train_dir),
        'val_dir': str(val_dir),
        'final_val_loss': float(metrics[0]),
        'final_val_accuracy': float(metrics[1]),
    }

    save_json(METADATA_PATH, metadata)
    save_json(
        CONFUSION_MATRIX_PATH,
        {
            'class_names': class_names,
            'matrix': confusion_matrix.tolist(),
            'total_samples': int(np.sum(confusion_matrix)),
            'image_path': str(CONFUSION_MATRIX_IMAGE_PATH),
        },
    )
    save_json(
        HISTORY_PATH,
        {
            'history': {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            'metadata': metadata,
            'skipped_files': {
                'train': train_split.skipped_files,
                'val': val_split.skipped_files,
            },
        },
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train BISINDO skeleton + LSTM model.')
    parser.add_argument('--train-dir', type=Path, default=TRAIN_DIR)
    parser.add_argument('--val-dir', type=Path, default=VAL_DIR)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    metadata = train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print('Training selesai.')
    print(f"Akurasi validasi: {metadata['final_val_accuracy']:.4f}")
    print(f'Model tersimpan di: {MODEL_PATH}')
    print(f'Confusion matrix tersimpan di: {CONFUSION_MATRIX_IMAGE_PATH}')


if __name__ == '__main__':
    main()
