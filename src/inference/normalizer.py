"""Normalización de salidas de pose a COCO Keypoints JSON.

Unifica las salidas heterogéneas de las tres familias (YOLOv8-pose, MoveNet,
BlazePose) en un único formato COCO::

    {
        "image_id": int,
        "category_id": 1,
        "keypoints": [x1, y1, v1, ..., x17, y17, v17],  # 51 valores
        "score": float,
        "bbox": [x, y, w, h],   # píxeles absolutos
    }

El módulo es **agnóstico a los frameworks**: no importa TensorFlow, MediaPipe ni
Ultralytics. Sólo lee ndarrays / atributos de objetos ya producidos por cada
modelo, por lo que es unitariamente testeable sin esas dependencias.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from src.inference.keypoint_schema import (
    DEFAULT_VISIBILITY_THRESHOLD,
    NUM_COCO_KEYPOINTS,
)
from src.models.blazepose import BLAZEPOSE_TO_COCO

Source = Literal["yolo", "movenet", "blazepose"]

# Padding del bounding box como fracción del lado, por cada borde.
_BBOX_PADDING = 0.10


def _to_numpy(array_like: Any) -> np.ndarray:
    """Convierte tensores (torch/tf) o secuencias a ``np.ndarray`` float.

    Args:
        array_like: tensor con método ``.cpu().numpy()`` / ``.numpy()`` o algo
            convertible por :func:`numpy.asarray`.

    Returns:
        Array numpy de tipo float64.
    """
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    if hasattr(array_like, "numpy"):
        array_like = array_like.numpy()
    return np.asarray(array_like, dtype=np.float64)


class KeypointNormalizer:
    """Convierte salidas crudas de cada familia a detecciones COCO JSON."""

    @staticmethod
    def normalize(
        raw_output: Any,
        source: Source,
        frame_w: int,
        frame_h: int,
        image_id: int = 0,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> list[dict]:
        """Normaliza la salida de un modelo a una lista de detecciones COCO.

        Args:
            raw_output: salida cruda del modelo. ``Results`` de Ultralytics para
                ``"yolo"``; ndarray ``(1, 1, 17, 3)`` ``(y, x, conf)`` para
                ``"movenet"``; ``NormalizedLandmarkList`` (33 landmarks) para
                ``"blazepose"``.
            source: familia que produjo la salida.
            frame_w: ancho del frame de entrada en píxeles.
            frame_h: alto del frame de entrada en píxeles.
            image_id: identificador del frame/imagen para el JSON.
            visibility_threshold: confianza mínima para marcar un keypoint como
                visible (``v = 2``); por debajo se marca ``v = 0``.

        Returns:
            Lista de detecciones en formato COCO JSON (una por persona). Vacía si
            no hubo detecciones.

        Raises:
            ValueError: si ``source`` no es una familia soportada.
        """
        if source == "yolo":
            return KeypointNormalizer._from_yolo(
                raw_output, frame_w, frame_h, image_id, visibility_threshold
            )
        if source == "movenet":
            return KeypointNormalizer._from_movenet(
                raw_output, frame_w, frame_h, image_id, visibility_threshold
            )
        if source == "blazepose":
            return KeypointNormalizer._from_blazepose(
                raw_output, frame_w, frame_h, image_id, visibility_threshold
            )
        raise ValueError(f"Fuente no soportada: {source!r}. Use 'yolo', 'movenet' o 'blazepose'.")

    # ------------------------------------------------------------------
    # Builders por familia.
    # ------------------------------------------------------------------

    @staticmethod
    def _from_yolo(
        raw_output: Any,
        frame_w: int,
        frame_h: int,
        image_id: int,
        visibility_threshold: float,
    ) -> list[dict]:
        """Construye detecciones desde ``Results`` de Ultralytics (en píxeles)."""
        # Ultralytics predict() devuelve una lista de Results; tomamos el primero.
        result = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output

        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or getattr(keypoints, "xy", None) is None:
            return []

        xy = _to_numpy(keypoints.xy)  # (num_persons, 17, 2) en píxeles
        if xy.size == 0:
            return []

        conf_attr = getattr(keypoints, "conf", None)
        if conf_attr is not None:
            conf = _to_numpy(conf_attr)  # (num_persons, 17)
        else:
            conf = np.ones(xy.shape[:2], dtype=np.float64)

        detections: list[dict] = []
        for person_xy, person_conf in zip(xy, conf):
            detections.append(
                _build_detection(
                    person_xy, person_conf, frame_w, frame_h, image_id, visibility_threshold
                )
            )
        return detections

    @staticmethod
    def _from_movenet(
        raw_output: Any,
        frame_w: int,
        frame_h: int,
        image_id: int,
        visibility_threshold: float,
    ) -> list[dict]:
        """Construye una detección desde la salida ``(1,1,17,3)`` ``(y,x,conf)``."""
        out = np.asarray(raw_output, dtype=np.float64).reshape(-1, NUM_COCO_KEYPOINTS, 3)
        if out.shape[0] == 0:
            return []

        person = out[0]  # (17, 3) con columnas (y, x, conf) normalizadas 0..1
        # Invertir (y, x) -> (x, y) y escalar a píxeles.
        xy = np.empty((NUM_COCO_KEYPOINTS, 2), dtype=np.float64)
        xy[:, 0] = person[:, 1] * frame_w  # x
        xy[:, 1] = person[:, 0] * frame_h  # y
        conf = person[:, 2]

        return [_build_detection(xy, conf, frame_w, frame_h, image_id, visibility_threshold)]

    @staticmethod
    def _from_blazepose(
        raw_output: Any,
        frame_w: int,
        frame_h: int,
        image_id: int,
        visibility_threshold: float,
    ) -> list[dict]:
        """Construye una detección desde un ``NormalizedLandmarkList`` (33 kp)."""
        if raw_output is None:
            return []
        landmarks = getattr(raw_output, "landmark", None)
        if landmarks is None or len(landmarks) == 0:
            return []

        xy = np.empty((NUM_COCO_KEYPOINTS, 2), dtype=np.float64)
        conf = np.empty(NUM_COCO_KEYPOINTS, dtype=np.float64)
        for coco_idx, bp_idx in enumerate(BLAZEPOSE_TO_COCO):
            lm = landmarks[bp_idx]
            xy[coco_idx, 0] = lm.x * frame_w
            xy[coco_idx, 1] = lm.y * frame_h
            conf[coco_idx] = getattr(lm, "visibility", 0.0)

        return [_build_detection(xy, conf, frame_w, frame_h, image_id, visibility_threshold)]


def _build_detection(
    xy: np.ndarray,
    conf: np.ndarray,
    frame_w: int,
    frame_h: int,
    image_id: int,
    visibility_threshold: float,
) -> dict:
    """Arma un dict de detección COCO a partir de keypoints y confidencias.

    Args:
        xy: ``(17, 2)`` coordenadas x, y en píxeles.
        conf: ``(17,)`` confidencias por keypoint.
        frame_w: ancho del frame (para clampear el bbox).
        frame_h: alto del frame (para clampear el bbox).
        image_id: identificador del frame/imagen.
        visibility_threshold: umbral para marcar ``v = 2`` vs ``v = 0``.

    Returns:
        Detección en formato COCO JSON.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(NUM_COCO_KEYPOINTS, 2)
    conf = np.asarray(conf, dtype=np.float64).reshape(NUM_COCO_KEYPOINTS)

    visible = conf > visibility_threshold  # (17,) bool

    # keypoints planos [x, y, v] * 17. v = 2 si visible, 0 si no.
    keypoints: list[float] = []
    for i in range(NUM_COCO_KEYPOINTS):
        v = 2 if visible[i] else 0
        keypoints.extend([float(xy[i, 0]), float(xy[i, 1]), float(v)])

    bbox = _bbox_from_keypoints(xy, visible, frame_w, frame_h)
    score = float(conf[visible].mean()) if visible.any() else 0.0

    return {
        "image_id": int(image_id),
        "category_id": 1,
        "keypoints": keypoints,
        "score": score,
        "bbox": bbox,
    }


def _bbox_from_keypoints(
    xy: np.ndarray,
    visible: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> list[float]:
    """Bounding box mínimo sobre keypoints visibles, con 10% de padding por lado.

    Args:
        xy: ``(17, 2)`` coordenadas en píxeles.
        visible: ``(17,)`` máscara booleana de keypoints visibles.
        frame_w: ancho del frame (para clampear).
        frame_h: alto del frame (para clampear).

    Returns:
        ``[x, y, w, h]`` en píxeles, clampeado al frame. ``[0, 0, 0, 0]`` si no
        hay keypoints visibles.
    """
    if not visible.any():
        return [0.0, 0.0, 0.0, 0.0]

    pts = xy[visible]
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    pad_x = (x_max - x_min) * _BBOX_PADDING
    pad_y = (y_max - y_min) * _BBOX_PADDING

    x0 = float(np.clip(x_min - pad_x, 0.0, frame_w))
    y0 = float(np.clip(y_min - pad_y, 0.0, frame_h))
    x1 = float(np.clip(x_max + pad_x, 0.0, frame_w))
    y1 = float(np.clip(y_max + pad_y, 0.0, frame_h))

    return [x0, y0, x1 - x0, y1 - y0]
