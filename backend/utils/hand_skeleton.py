from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np

from utils.preprocessing import FEATURE_DIM, MAX_HANDS, empty_sequence, landmarks_to_sequence


@dataclass
class HandDetectionResult:
    success: bool
    sequence: np.ndarray
    landmarks: list[list[float]]
    annotated_image: Optional[np.ndarray]
    message: str = ''


@dataclass
class DetectionVariant:
    name: str
    image: np.ndarray
    restore_image: Callable[[np.ndarray], np.ndarray]


class HandSkeletonExtractor:
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = MAX_HANDS,
        min_detection_confidence: float = 0.35,
    ) -> None:
        self.max_num_hands = max_num_hands
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

    @staticmethod
    def _identity(image: np.ndarray) -> np.ndarray:
        return image.copy()

    @staticmethod
    def _flip_horizontal(image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 1)

    @staticmethod
    def _rotate_cw(image: np.ndarray) -> np.ndarray:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    @staticmethod
    def _rotate_ccw(image: np.ndarray) -> np.ndarray:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @staticmethod
    def _rotate_180(image: np.ndarray) -> np.ndarray:
        return cv2.rotate(image, cv2.ROTATE_180)

    @classmethod
    def _generate_detection_variants(cls, image_bgr: np.ndarray) -> list[DetectionVariant]:
        # Augmentasi ringan di tahap inferensi membantu file rotate/flip tetap bisa terbaca.
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        clahe_bgr = cv2.cvtColor(
            cv2.merge([enhanced_l, a_channel, b_channel]),
            cv2.COLOR_LAB2BGR,
        )

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(clahe_bgr, -1, sharpen_kernel)

        base_images = [
            ('original', image_bgr),
            ('clahe', clahe_bgr),
            ('sharpened', sharpened),
        ]
        transforms = [
            ('identity', cls._identity, cls._identity),
            ('flip', cls._flip_horizontal, cls._flip_horizontal),
            ('rot_cw', cls._rotate_cw, cls._rotate_ccw),
            ('rot_ccw', cls._rotate_ccw, cls._rotate_cw),
            ('rot_180', cls._rotate_180, cls._rotate_180),
        ]

        variants: list[DetectionVariant] = []
        for base_name, base_image in base_images:
            for transform_name, apply_fn, restore_fn in transforms:
                variants.append(
                    DetectionVariant(
                        name=f'{base_name}_{transform_name}',
                        image=apply_fn(base_image),
                        restore_image=restore_fn,
                    )
                )
        return variants

    @staticmethod
    def _extract_handedness(results: object, index: int) -> str:
        handedness = getattr(results, 'multi_handedness', None)
        if not handedness or index >= len(handedness):
            return 'Unknown'
        if not handedness[index].classification:
            return 'Unknown'
        return handedness[index].classification[0].label

    @staticmethod
    def _combine_sequences(
        hand_sequences: list[np.ndarray],
        feature_dim: int,
        max_hands: int,
    ) -> np.ndarray:
        padded_sequences = hand_sequences[:max_hands]
        while len(padded_sequences) < max_hands:
            padded_sequences.append(empty_sequence(feature_dim=feature_dim, max_hands=1))
        return np.concatenate(padded_sequences, axis=0).astype(np.float32)

    def detect(
        self,
        image_bgr: np.ndarray,
        draw: bool = True,
        feature_dim: int = FEATURE_DIM,
        max_hands: int | None = None,
    ) -> HandDetectionResult:
        expected_hands = max_hands or self.max_num_hands

        # Coba beberapa varian gambar dan ambil varian pertama yang berhasil mendeteksi tangan.
        for variant in self._generate_detection_variants(image_bgr):
            rgb_image = cv2.cvtColor(variant.image, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb_image)

            if not results.multi_hand_landmarks:
                continue

            detected_hands: list[dict] = []
            world_landmarks = getattr(results, 'multi_hand_world_landmarks', None) or []

            for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                feature_landmarks = (
                    world_landmarks[index].landmark
                    if index < len(world_landmarks)
                    else hand_landmarks.landmark
                )
                image_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                mean_x = float(np.mean([point[0] for point in image_landmarks]))
                handedness = self._extract_handedness(results, index)
                detected_hands.append(
                    {
                        'handedness': handedness,
                        'mean_x': mean_x,
                        'sequence': landmarks_to_sequence(feature_landmarks, feature_dim=feature_dim),
                        'landmarks': image_landmarks,
                    }
                )

            if not detected_hands:
                continue

            # Urutan tangan dibuat stabil agar slot sequence tangan-1 dan tangan-2 konsisten.
            detected_hands.sort(
                key=lambda item: (
                    {'Left': 0, 'Right': 1}.get(item['handedness'], 2),
                    item['mean_x'],
                )
            )

            sequence = self._combine_sequences(
                [item['sequence'] for item in detected_hands],
                feature_dim=feature_dim,
                max_hands=expected_hands,
            )
            landmarks = [point for item in detected_hands[:expected_hands] for point in item['landmarks']]

            annotated = None
            if draw:
                annotated_variant = variant.image.copy()
                for hand_landmarks in results.multi_hand_landmarks[:expected_hands]:
                    self._mp_drawing.draw_landmarks(
                        annotated_variant,
                        hand_landmarks,
                        self._mp_hands.HAND_CONNECTIONS,
                        self._mp_styles.get_default_hand_landmarks_style(),
                        self._mp_styles.get_default_hand_connections_style(),
                    )
                annotated = variant.restore_image(annotated_variant)

            detected_count = min(len(detected_hands), expected_hands)
            return HandDetectionResult(
                success=True,
                sequence=sequence,
                landmarks=landmarks,
                annotated_image=annotated,
                message=f'{detected_count} tangan berhasil diekstraksi.',
            )

        return HandDetectionResult(
            success=False,
            sequence=empty_sequence(feature_dim=feature_dim, max_hands=expected_hands),
            landmarks=[],
            annotated_image=image_bgr.copy() if draw else None,
            message='Tangan tidak terdeteksi pada gambar.',
        )

    @staticmethod
    def encode_image_base64(image_bgr: np.ndarray) -> Optional[str]:
        success, buffer = cv2.imencode('.png', image_bgr)
        if not success:
            return None
        return base64.b64encode(buffer.tobytes()).decode('utf-8')

