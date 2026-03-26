from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tensorflow import keras

from config import METADATA_PATH, MODEL_PATH
from utils.hand_skeleton import HandSkeletonExtractor
from utils.preprocessing import LANDMARK_COUNT
from utils.serialization import load_json


@dataclass
class PredictionResult:
    success: bool
    predicted_label: str | None
    confidence: float
    top_predictions: list[dict[str, float]]
    landmarks: list[list[float]]
    skeleton_image_base64: str | None
    message: str = ''


class SignPredictor:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.extractor = HandSkeletonExtractor(static_image_mode=True, max_num_hands=2)
        self._model: keras.Model | None = None
        self._metadata: dict[str, Any] | None = None

    @property
    def model_loaded(self) -> bool:
        return self.model_path.exists() and self.metadata_path.exists()

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            self._metadata = load_json(self.metadata_path)
        return self._metadata

    @property
    def class_names(self) -> list[str]:
        return list(self.metadata['class_names'])

    @property
    def expected_feature_dim(self) -> int:
        return int(self.metadata.get('feature_dim', 5))

    @property
    def expected_num_hands(self) -> int:
        sequence_length = int(self.metadata.get('sequence_length', LANDMARK_COUNT))
        return max(1, sequence_length // LANDMARK_COUNT)

    @property
    def model(self) -> keras.Model:
        if self._model is None:
            self._model = keras.models.load_model(self.model_path)
        return self._model

    def decode_image(self, image_bytes: bytes) -> np.ndarray | None:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def predict_bytes(self, image_bytes: bytes) -> PredictionResult:
        image_bgr = self.decode_image(image_bytes)
        if image_bgr is None:
            return PredictionResult(
                success=False,
                predicted_label=None,
                confidence=0.0,
                top_predictions=[],
                landmarks=[],
                skeleton_image_base64=None,
                message='Gambar tidak valid atau gagal dibaca.',
            )
        return self.predict_image(image_bgr)

    def predict_image(self, image_bgr: np.ndarray) -> PredictionResult:
        detection = self.extractor.detect(
            image_bgr,
            draw=True,
            feature_dim=self.expected_feature_dim,
            max_hands=self.expected_num_hands,
        )
        if not detection.success:
            return PredictionResult(
                success=False,
                predicted_label=None,
                confidence=0.0,
                top_predictions=[],
                landmarks=[],
                skeleton_image_base64=HandSkeletonExtractor.encode_image_base64(
                    detection.annotated_image
                )
                if detection.annotated_image is not None
                else None,
                message=detection.message,
            )

        probabilities = self.model.predict(detection.sequence[np.newaxis, ...], verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                'label': self.class_names[index],
                'confidence': float(probabilities[index]),
            }
            for index in top_indices
        ]

        return PredictionResult(
            success=True,
            predicted_label=self.class_names[best_index],
            confidence=float(probabilities[best_index]),
            top_predictions=top_predictions,
            landmarks=detection.landmarks,
            skeleton_image_base64=HandSkeletonExtractor.encode_image_base64(
                detection.annotated_image
            )
            if detection.annotated_image is not None
            else None,
            message=detection.message if detection.message else 'Prediksi berhasil.',
        )
