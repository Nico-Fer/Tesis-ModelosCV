"""Esquema canónico del esqueleto COCO de 17 keypoints.

Define el orden de los keypoints, las conexiones del esqueleto (para dibujo) y
el umbral de visibilidad por defecto. Es la referencia compartida por el
normalizador (:mod:`src.inference.normalizer`) y el pipeline de video
(:mod:`src.inference.video_pipeline`).

Convención de visibility en la salida COCO JSON:
    * 0 = no anotado / no detectado.
    * 1 = anotado pero no visible (ocluido).
    * 2 = visible y detectado (confidence por encima del umbral).
"""

from __future__ import annotations

# Nombres de los 17 keypoints en el orden canónico de COCO.
COCO_KEYPOINT_NAMES: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Conexiones del esqueleto COCO como pares de índices (para visualización).
COCO_SKELETON: list[tuple[int, int]] = [
    (15, 13),  # left ankle  - left knee
    (13, 11),  # left knee   - left hip
    (16, 14),  # right ankle - right knee
    (14, 12),  # right knee  - right hip
    (11, 12),  # left hip    - right hip
    (5, 11),  # left shoulder  - left hip
    (6, 12),  # right shoulder - right hip
    (5, 6),  # left shoulder  - right shoulder
    (5, 7),  # left shoulder  - left elbow
    (6, 8),  # right shoulder - right elbow
    (7, 9),  # left elbow  - left wrist
    (8, 10),  # right elbow - right wrist
    (1, 2),  # left eye - right eye
    (0, 1),  # nose - left eye
    (0, 2),  # nose - right eye
    (1, 3),  # left eye  - left ear
    (2, 4),  # right eye - right ear
    (3, 5),  # left ear  - left shoulder
    (4, 6),  # right ear - right shoulder
]

# Confianza mínima para marcar un keypoint como visible (visibility = 2).
DEFAULT_VISIBILITY_THRESHOLD: float = 0.3

# Número de keypoints del esqueleto COCO.
NUM_COCO_KEYPOINTS: int = len(COCO_KEYPOINT_NAMES)
