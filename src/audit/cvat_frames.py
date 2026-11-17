"""Mapa frame → job de CVAT y derivación del conjunto auditado.

"Auditado" NO se puede inferir de qué frames aparecen en un export: CVAT
exporta todos los frames con pre-anotación, se hayan revisado o no (el export
sin editar del gate de round-trip contiene todos los frames del pack). La señal
es el **estado del job**: el anotador marca el job ``completed`` al terminar de
revisar todos sus frames, y entonces un frame no editado significa "revisado y
correcto".

Este módulo es la parte pura de esa lógica (sin red): arma el mapa
``stem → (task, job, frame)`` a partir de lo que devuelve la API
(``GET /api/tasks/{id}/data/meta`` + ``GET /api/jobs``) y deriva la lista de
stems auditados. El cliente HTTP vive en ``scripts/cvat_client.py``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

FRAMES_FIELDS = ["stem", "task_id", "job_id", "frame"]


def build_frame_map(
    task_id: int, frame_names: Sequence[str], jobs: Iterable[Mapping[str, Any]]
) -> list[dict[str, str]]:
    """Asigna cada frame de la task a su job de anotación.

    Args:
        task_id: id de la task.
        frame_names: nombres de archivo en el orden de frames de la task
            (``data/meta`` → ``frames[i].name``); el índice es el número de frame.
        jobs: jobs de la task (``id``, ``start_frame``, ``stop_frame``, ``type``).
            Se ignoran los que no son de anotación (ground truth, consenso).

    Returns:
        Una fila por frame con las columnas de :data:`FRAMES_FIELDS`.

    Raises:
        ValueError: si un frame queda sin job o en dos jobs, o hay stems
            duplicados (el stem es la clave de todo el pipeline).
    """
    job_de_frame: dict[int, int] = {}
    for job in jobs:
        if job.get("type", "annotation") != "annotation":
            continue
        for frame in range(int(job["start_frame"]), int(job["stop_frame"]) + 1):
            if frame in job_de_frame:
                raise ValueError(
                    f"Frame {frame} de la task {task_id} en dos jobs "
                    f"({job_de_frame[frame]} y {job['id']})."
                )
            job_de_frame[frame] = int(job["id"])

    filas: list[dict[str, str]] = []
    vistos: set[str] = set()
    for frame, name in enumerate(frame_names):
        if frame not in job_de_frame:
            raise ValueError(f"Frame {frame} ({name}) de la task {task_id} sin job de anotación.")
        stem = Path(name).stem
        if stem in vistos:
            raise ValueError(f"Stem duplicado en la task {task_id}: {stem}")
        vistos.add(stem)
        filas.append(
            {
                "stem": stem,
                "task_id": str(task_id),
                "job_id": str(job_de_frame[frame]),
                "frame": str(frame),
            }
        )
    return filas


def write_frame_map(filas: list[dict[str, str]], path: Path) -> None:
    """Escribe el mapa en CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FRAMES_FIELDS)
        writer.writeheader()
        writer.writerows(filas)


def read_frame_map(path: Path) -> list[dict[str, str]]:
    """Lee el mapa escrito por :func:`write_frame_map`.

    Raises:
        FileNotFoundError: si no existe (la task no se montó con ``cvat_audit.py``).
        ValueError: si las columnas no son las esperadas.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}; se genera al montar la task con `cvat_audit.py montar`."
        )
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != FRAMES_FIELDS:
            raise ValueError(f"Columnas inesperadas en {path}: {reader.fieldnames}")
        return list(reader)


def stems_auditados(
    filas: Iterable[Mapping[str, str]],
    estado_por_job: Mapping[int, str],
    estado_auditado: str,
    frames_borrados: Mapping[int, set[int]] | None = None,
) -> list[str]:
    """Stems de los frames cuyo job está en ``estado_auditado``.

    Args:
        filas: mapa frame → job.
        estado_por_job: ``job_id → state`` actual en CVAT.
        estado_auditado: estado que cuenta como revisado (``completed``).
        frames_borrados: ``task_id → {frames}`` marcados como borrados en CVAT;
            un frame borrado no se revisó y no cuenta.

    Raises:
        ValueError: si un job del mapa ya no existe en CVAT (task recreada o
            re-segmentada: el mapa es de otra task).
    """
    borrados = frames_borrados or {}
    out: list[str] = []
    for fila in filas:
        job_id = int(fila["job_id"])
        if job_id not in estado_por_job:
            raise ValueError(
                f"El job {job_id} del mapa de frames no existe en CVAT; "
                "el mapa corresponde a otra task."
            )
        if int(fila["frame"]) in borrados.get(int(fila["task_id"]), set()):
            continue
        if estado_por_job[job_id] == estado_auditado:
            out.append(fila["stem"])
    return sorted(out)
