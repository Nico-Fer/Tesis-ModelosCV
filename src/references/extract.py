"""Extracción de keypoints cuadro a cuadro desde un video grabado.

Los cuadros sin persona detectada se **avisan y se saltan** en el sentido de
"sin datos": se preserva la base de tiempo con una fila de confianza 0 (queda
enmascarada aguas abajo, igual que un keypoint de baja confianza). Colapsar el
eje temporal desalinearía el ciclo con el fps real del video.

**Rotación sí, espejado nunca.** Un video de celular guarda la orientación como
metadata del contenedor y cv2 la ignora salvo que se le pida
(``CAP_PROP_ORIENTATION_AUTO``), así que un clip vertical llega acostado y el
detector — entrenado con personas de pie — trabaja en su peor caso. Aplicarla es
seguro: el ángulo es SIGNADO vía ``atan2(cross, dot)`` y una rotación preserva
ambos productos, así que el feature no cambia; además reproduce el encuadre que
recibe la app. Un **espejado** sí invertiría el signo, y por eso no se hace.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.references.backend import PoseBackend

logger = logging.getLogger(__name__)


def extract_keypoints(
    video_path: str | Path, backend: PoseBackend
) -> tuple[np.ndarray, float, list[int]]:
    """Corre el backend sobre todos los cuadros: ``(T, 17, 3)`` px, fps real, saltados.

    Se aplica la rotación declarada por el contenedor (invariante para el ángulo
    signado); nunca se espeja.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video no encontrado: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 no pudo abrir el video: {path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    rotation = float(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    if rotation:
        logger.info("%s: rotación del contenedor %.0f° aplicada", path.name, rotation)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError(f"fps inválido ({fps}) en {path}")

    frames: list[np.ndarray] = []
    skipped: list[int] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            kpts = backend.keypoints(frame)
            if kpts is None:
                skipped.append(idx)
                kpts = np.zeros((17, 3))  # conf 0: fila enmascarada, base de tiempo intacta
            frames.append(kpts)
            idx += 1
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"El video no tiene cuadros legibles: {path}")
    if skipped:
        logger.warning(
            "%s: %d/%d cuadros SIN persona detectada (saltados): %s",
            path.name,
            len(skipped),
            len(frames),
            _summarize(skipped),
        )
    return np.stack(frames), fps, skipped


def _summarize(indices: list[int], limit: int = 10) -> str:
    if len(indices) <= limit:
        return str(indices)
    return f"{indices[:limit]}… (+{len(indices) - limit})"
