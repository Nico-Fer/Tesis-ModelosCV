"""Backends de pose detrás de una interfaz única para el generador de referencias.

``PoseBackend.keypoints(frame) -> (17, 3)`` en **píxeles** de la imagen
original, orden COCO-17, ``(x, y, conf)``. Persona principal por cuadro =
**bbox de mayor área**. El frame NUNCA se espeja ni voltea (el flip invierte el
signo del ángulo).

Default: Ultralytics YOLOv8-pose sobre un ``.pt`` (``models/yolov8n-pose.pt``
stock por ahora — el fine-tuneado se enchufa cambiando sólo ``model`` en la
config). Los imports de framework son diferidos (patrón del registry) para que
el módulo cargue en cualquier venv; un backend mmpose/ViTPose-H futuro es un
branch más en :func:`build_backend`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "models/yolov8n-pose.pt"


def _to_numpy(x: Any) -> np.ndarray:  # noqa: ANN401 — Tensor (quizás en GPU) o ndarray
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x)


class PoseBackend(Protocol):
    """Contrato mínimo: keypoints COCO-17 en px o ``None`` si no hay persona."""

    def keypoints(self, frame: np.ndarray) -> np.ndarray | None:
        """``frame`` BGR (como lo entrega cv2) → ``(17, 3)`` px o ``None``."""
        ...


class YoloPoseBackend:
    """YOLOv8-pose vía Ultralytics sobre un checkpoint ``.pt``."""

    def __init__(self, model_path: str | Path, *, imgsz: int = 640, conf: float = 0.25) -> None:
        from ultralytics import YOLO  # noqa: PLC0415 — import diferido

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint de pose no encontrado: {path}")
        # task="pose" explícito: costumbre del repo (un .tflite no declara task).
        self._yolo = YOLO(str(path), task="pose")
        self._imgsz = imgsz
        self._conf = conf

    def keypoints(self, frame: np.ndarray) -> np.ndarray | None:
        result = self._yolo.predict(
            source=frame, imgsz=self._imgsz, conf=self._conf, verbose=False
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or result.keypoints is None:
            return None
        xywh = _to_numpy(boxes.xywh)
        best = int(np.argmax(xywh[:, 2] * xywh[:, 3]))  # persona principal = mayor área
        kpts = _to_numpy(result.keypoints.data[best])  # (17, 3) px, (x, y, conf)
        if kpts.shape != (17, 3):
            return None
        return kpts.astype(np.float64)


def build_backend(config: dict[str, Any]) -> PoseBackend:
    """Fábrica por ``config['backend']`` (default ``yolo``), imports diferidos."""
    kind = str(config.get("backend", "yolo")).lower()
    if kind == "yolo":
        return YoloPoseBackend(
            config.get("model", DEFAULT_MODEL),
            imgsz=int(config.get("imgsz", 640)),
            conf=float(config.get("det_conf", 0.25)),
        )
    # Futuro: elif kind == "mmpose": ...  (ViTPose-H top-down, venv-mmpose)
    raise ValueError(f"Backend de pose desconocido: {kind!r} (soportados: yolo)")
