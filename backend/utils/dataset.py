from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from utils.hand_skeleton import HandSkeletonExtractor
from utils.preprocessing import FEATURE_DIM, LANDMARK_COUNT


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


@dataclass
class DatasetSplit:
    features: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    processed_count: int
    skipped_count: int
    skipped_files: list[str]


def read_image(image_path: Path) -> np.ndarray | None:
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def resolve_image_root(split_dir: Path) -> Path:
    image_root = split_dir / 'images'
    return image_root if image_root.exists() else split_dir


def discover_class_names(*split_dirs: Path) -> list[str]:
    class_names: set[str] = set()
    for split_dir in split_dirs:
        image_root = resolve_image_root(Path(split_dir))
        if not image_root.exists():
            continue
        for item in image_root.iterdir():
            if item.is_dir():
                class_names.add(item.name)
    return sorted(class_names)


def load_split(
    split_dir: Path,
    class_names: list[str],
    extractor: HandSkeletonExtractor,
) -> DatasetSplit:
    image_root = resolve_image_root(split_dir)
    features: list[np.ndarray] = []
    labels: list[int] = []
    skipped_files: list[str] = []
    processed_count = 0
    skipped_count = 0
    class_to_index = {name: index for index, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = image_root / class_name
        if not class_dir.exists() or not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            image = read_image(image_path)
            if image is None:
                skipped_count += 1
                skipped_files.append(str(image_path))
                continue

            detection = extractor.detect(image, draw=False)
            if not detection.success:
                skipped_count += 1
                skipped_files.append(str(image_path))
                continue

            features.append(detection.sequence)
            labels.append(class_to_index[class_name])
            processed_count += 1

    sequence_length = extractor.max_num_hands * LANDMARK_COUNT
    feature_array = (
        np.stack(features).astype(np.float32)
        if features
        else np.empty((0, sequence_length, FEATURE_DIM), dtype=np.float32)
    )

    return DatasetSplit(
        features=feature_array,
        labels=np.array(labels, dtype=np.int32),
        class_names=class_names,
        processed_count=processed_count,
        skipped_count=skipped_count,
        skipped_files=skipped_files,
    )
