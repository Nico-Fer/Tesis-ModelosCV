"""Extracción de frames de los clips de video UCO con OpenCV.

YOLOv8-pose entrena sobre **imágenes sueltas**, no video. Este módulo recorre
los ``camX.mp4`` y vuelca frames a disco, submuestreando (1 de cada ``stride``)
para no generar millones de imágenes casi idénticas.

Cada frame extraído debe quedar emparejado con su anotación 2D (ver
:mod:`src.data.uco_to_yolo`). La convención de nombre incluye sujeto, acción,
cámara y número de frame para que el conversor pueda reconstruir la clave hacia
``dataset_2d.json``::

    {sujeto}_{accion}_{cam}_{frame:06d}.jpg

Sólo depende de ``cv2`` (en requirements-base), así que importa en cualquier venv.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ExtractedFrame:
    """Un frame extraído y su procedencia, para emparejar con la anotación."""

    path: Path
    subject: str
    action: str
    cam: str
    frame_idx: int


def iter_clips(dataset_root: str | Path) -> Iterator[tuple[str, str, str, Path]]:
    """Itera los clips de video del dataset extraído.

    Args:
        dataset_root: raíz del dataset extraído (contiene ``clips_mp4/``).

    Yields:
        Tuplas ``(sujeto, accion, cam, ruta_mp4)``.
    """
    clips_dir = Path(dataset_root) / "clips_mp4"
    for mp4 in sorted(clips_dir.glob("*/*/cam*.mp4")):
        # clips_mp4/{sujeto}/{accion}/cam{N}.mp4
        action_dir = mp4.parent
        subject_dir = action_dir.parent
        cam = mp4.stem  # "cam0" … "cam4"
        yield subject_dir.name, action_dir.name, cam, mp4


def extract_frames_from_clip(
    mp4_path: Path,
    out_dir: Path,
    subject: str,
    action: str,
    cam: str,
    *,
    stride: int = 5,
    jpg_quality: int = 95,
) -> list[ExtractedFrame]:
    """Extrae frames de un clip, 1 de cada ``stride``.

    Args:
        mp4_path: ruta del ``.mp4``.
        out_dir: carpeta donde se escriben los ``.jpg``.
        subject, action, cam: metadatos de procedencia (para el nombre y el
            emparejamiento posterior con la anotación).
        stride: submuestreo temporal (stride=5 ⇒ 1 frame de cada 5).
        jpg_quality: calidad JPEG (0–100).

    Returns:
        Lista de :class:`ExtractedFrame` escritos.
    """
    import cv2  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4_path))
    extracted: list[ExtractedFrame] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                name = f"{subject}_{action}_{cam}_{idx:06d}.jpg"
                path = out_dir / name
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
                extracted.append(ExtractedFrame(path, subject, action, cam, idx))
            idx += 1
    finally:
        cap.release()
    return extracted


def extract_all(
    dataset_root: str | Path,
    out_dir: str | Path,
    *,
    stride: int = 5,
) -> list[ExtractedFrame]:
    """Extrae frames de todos los clips del dataset.

    Args:
        dataset_root: raíz del dataset extraído.
        out_dir: carpeta destino de las imágenes.
        stride: submuestreo temporal.

    Returns:
        Lista plana de todos los frames extraídos.
    """
    out_dir = Path(out_dir)
    all_frames: list[ExtractedFrame] = []
    for subject, action, cam, mp4 in iter_clips(dataset_root):
        # TODO(data): paralelizar por clip (multiprocessing) si la extracción
        # secuencial resulta muy lenta sobre los ~5000 clips.
        all_frames.extend(
            extract_frames_from_clip(mp4, out_dir, subject, action, cam, stride=stride)
        )
    return all_frames
