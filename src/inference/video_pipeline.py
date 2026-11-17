"""Pipeline de inferencia sobre video / cámara con volcado a COCO JSON.

Recorre un video (archivo o cámara) con un :class:`~src.inference.pose_estimator.PoseEstimator`,
acumula las detecciones COCO de cada frame y las serializa a un JSON. Opcional:
muestra una ventana de OpenCV con los keypoints y el esqueleto superpuestos.

Depende sólo de ``cv2`` y ``numpy`` (requirements-base).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.inference.keypoint_schema import COCO_SKELETON, NUM_COCO_KEYPOINTS
from src.inference.pose_estimator import PoseEstimator

# Colores BGR para el dibujo de keypoints y conexiones.
_KEYPOINT_COLOR = (0, 255, 0)
_SKELETON_COLOR = (255, 128, 0)


def run_inference_on_video(
    estimator: PoseEstimator,
    source: str | int,
    output_json: Path,
    max_frames: int | None = None,
    show_preview: bool = False,
) -> Path:
    """Corre inferencia frame a frame sobre un video y vuelca las detecciones.

    Args:
        estimator: estimador ya construido (no es necesario haber entrado al
            context manager; esta función carga el modelo vía el propio
            estimador).
        source: ruta a un archivo de video o índice de cámara (p. ej. ``0``).
        output_json: archivo destino donde se escribe la lista de detecciones.
        max_frames: si se indica, corta el procesamiento tras esa cantidad de
            frames (útil para smoke tests).
        show_preview: si ``True`` abre una ventana de OpenCV con keypoints y
            esqueleto dibujados; se cierra con la tecla ``q``.

    Returns:
        La ruta ``output_json`` con las detecciones serializadas.

    Raises:
        RuntimeError: si no se puede abrir la fuente de video.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source!r}.")

    all_detections: list[dict] = []
    frame_idx = 0

    with estimator:
        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                detections = estimator.predict(frame, image_id=frame_idx)
                all_detections.extend(detections)

                if show_preview:
                    preview = draw_keypoints(frame, detections, COCO_SKELETON)
                    cv2.imshow("pose", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(all_detections, fh, indent=2)

    return output_json


def draw_keypoints(
    frame: np.ndarray,
    detections: list[dict],
    skeleton: list[tuple[int, int]],
) -> np.ndarray:
    """Dibuja keypoints y esqueleto COCO sobre una copia del frame.

    Args:
        frame: frame BGR de OpenCV, shape ``(H, W, 3)``.
        detections: detecciones COCO (cada una con ``keypoints`` de 51 valores).
        skeleton: pares de índices de keypoints a conectar (p. ej.
            :data:`~src.inference.keypoint_schema.COCO_SKELETON`).

    Returns:
        Copia del frame con las anotaciones dibujadas.
    """
    canvas = frame.copy()

    for det in detections:
        kps = det["keypoints"]
        # Reagrupar en (17, 3) -> (x, y, v).
        pts = [(kps[3 * i], kps[3 * i + 1], kps[3 * i + 2]) for i in range(NUM_COCO_KEYPOINTS)]

        # Conexiones del esqueleto (sólo si ambos extremos son visibles).
        for a, b in skeleton:
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va > 0 and vb > 0:
                cv2.line(
                    canvas,
                    (int(xa), int(ya)),
                    (int(xb), int(yb)),
                    _SKELETON_COLOR,
                    2,
                )

        # Keypoints visibles.
        for x, y, v in pts:
            if v > 0:
                cv2.circle(canvas, (int(x), int(y)), 3, _KEYPOINT_COLOR, -1)

    return canvas
