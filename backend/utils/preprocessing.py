from __future__ import annotations

from typing import Iterable

import numpy as np


LANDMARK_COUNT = 21
MAX_HANDS = 2
FEATURE_DIM = 8
LEGACY_FEATURE_DIM = 5
PARENT_INDICES = np.array(
    [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
    dtype=np.int32,
)


def landmarks_to_xyz(landmarks: Iterable) -> np.ndarray:
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    if points.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected shape {(LANDMARK_COUNT, 3)}, got {points.shape}")
    return points


def normalize_landmarks(points: np.ndarray) -> np.ndarray:
    if points.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected shape {(LANDMARK_COUNT, 3)}, got {points.shape}")

    wrist = points[0]
    translated = points - wrist

    # Align the wrist-to-middle-finger axis so samples are less sensitive to hand tilt.
    middle_mcp_xy = translated[9, :2]
    if np.linalg.norm(middle_mcp_xy) > 1e-6:
        angle = np.arctan2(middle_mcp_xy[1], middle_mcp_xy[0])
        target_angle = -np.pi / 2
        rotation = target_angle - angle
        cos_theta = np.cos(rotation)
        sin_theta = np.sin(rotation)
        rotation_matrix = np.array(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
            dtype=np.float32,
        )
        translated[:, :2] = translated[:, :2] @ rotation_matrix.T

    palm_height = np.linalg.norm(translated[9])
    palm_width = np.linalg.norm(translated[5] - translated[17])
    max_radius = np.max(np.linalg.norm(translated[1:], axis=1))
    scale = max(palm_height, palm_width, max_radius, 1e-6)

    return translated / scale


def build_feature_stack(normalized: np.ndarray, feature_dim: int = FEATURE_DIM) -> np.ndarray:
    centroid = normalized.mean(axis=0, keepdims=True)
    wrist_distance = np.linalg.norm(normalized, axis=1, keepdims=True)
    centroid_distance = np.linalg.norm(normalized - centroid, axis=1, keepdims=True)

    if feature_dim == LEGACY_FEATURE_DIM:
        return np.concatenate([normalized, wrist_distance, centroid_distance], axis=1).astype(
            np.float32
        )

    parent_vectors = normalized - normalized[PARENT_INDICES]
    parent_vectors[0] = 0.0

    return np.concatenate(
        [normalized, parent_vectors, wrist_distance, centroid_distance],
        axis=1,
    ).astype(np.float32)


def landmarks_to_sequence(landmarks: Iterable, feature_dim: int = FEATURE_DIM) -> np.ndarray:
    points = landmarks_to_xyz(landmarks)
    normalized = normalize_landmarks(points)
    return build_feature_stack(normalized, feature_dim=feature_dim)


def empty_sequence(feature_dim: int = FEATURE_DIM, max_hands: int = MAX_HANDS) -> np.ndarray:
    return np.zeros((LANDMARK_COUNT * max_hands, feature_dim), dtype=np.float32)
