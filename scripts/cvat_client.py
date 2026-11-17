"""Cliente mínimo de la API REST de CVAT (v2.74.1) para la auditoría de pseudo-labels.

Solo ``requests`` + stdlib. Lo usan ``scripts/cvat_audit.py`` y
``scripts/cvat_roundtrip_gate.py``. Endpoints verificados contra el schema
OpenAPI del server desplegado (``GET /api/schema/``, 2026-09-28).

Dos trampas de la API, verificadas:

* upload en una sola request: headers ``Upload-Start`` + ``Upload-Finish`` y
  el archivo en el body; **sin** ``filename`` en la query (con él CVAT asume un
  upload TUS previo y responde "No such file were uploaded");
* todas las operaciones pesadas (import, export, backup, carga de datos) son
  asíncronas: devuelven ``rq_id`` y se sigue con ``GET /api/requests/{rq_id}``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("cvat_client")

POLL_S = 2.0
_UPLOAD_UNICO = {"Upload-Start": "true", "Upload-Finish": "true"}


def leer_credenciales(path: Path) -> dict[str, str]:
    """Lee ``cvat_credentials.local`` (KEY=VALUE por línea; NUNCA se commitea)."""
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}; crearlo con CVAT_URL/CVAT_USER/CVAT_PASS (ver docs/cvat_setup.md)."
        )
    creds: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            creds[key.strip()] = value.strip()
    for key in ("CVAT_URL", "CVAT_USER", "CVAT_PASS"):
        if key not in creds:
            raise ValueError(f"Falta {key} en {path}.")
    return creds


def sesion(credenciales: Path) -> tuple[requests.Session, str]:
    """Sesión autenticada (Basic) y URL base del server."""
    creds = leer_credenciales(credenciales)
    session = requests.Session()
    session.auth = (creds["CVAT_USER"], creds["CVAT_PASS"])
    return session, creds["CVAT_URL"].rstrip("/")


def version_server(session: requests.Session, base: str) -> str:
    """Versión de CVAT del server (para registrar en la metadata del export)."""
    resp = session.get(f"{base}/api/server/about", timeout=30)
    resp.raise_for_status()
    return str(resp.json().get("version", "?"))


def esperar_request(session: requests.Session, base: str, rq_id: str, timeout_s: float) -> dict:
    """Espera a que un request asíncrono de CVAT termine; devuelve su estado final.

    Raises:
        RuntimeError: si el job falla o no termina dentro del timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = session.get(f"{base}/api/requests/{rq_id}", timeout=30)
        resp.raise_for_status()
        estado = resp.json()
        if estado["status"] == "finished":
            return estado
        if estado["status"] == "failed":
            raise RuntimeError(f"Request {rq_id} falló en CVAT: {estado.get('message', estado)}")
        time.sleep(POLL_S)
    raise RuntimeError(f"Request {rq_id} no terminó en {timeout_s:.0f} s.")


def _listar(session: requests.Session, url: str, params: dict[str, Any]) -> list[dict]:
    """GET paginado: sigue ``next`` hasta juntar todos los resultados."""
    out: list[dict] = []
    siguiente: str | None = url
    query: dict[str, Any] | None = {**params, "page_size": 100}
    while siguiente:
        resp = session.get(siguiente, params=query, timeout=30)
        resp.raise_for_status()
        cuerpo = resp.json()
        out.extend(cuerpo["results"])
        siguiente = cuerpo.get("next")
        query = None  # `next` ya trae los parámetros
    return out


def _descargar(session: requests.Session, estado: dict, out: Path) -> None:
    """Baja el resultado de un request de export/backup terminado."""
    result_url = estado.get("result_url")
    if not result_url:
        raise RuntimeError(f"El request no devolvió result_url: {estado}")
    descarga = session.get(result_url, timeout=600)
    descarga.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(descarga.content)
    logger.info("Descargado: %s (%.1f KB).", out, len(descarga.content) / 1024)


# ---------------------------------------------------------------------------
# Proyecto y task
# ---------------------------------------------------------------------------


def crear_proyecto(session: requests.Session, base: str, nombre: str, labels: list[dict]) -> int:
    """Crea un proyecto con la definición de labels dada; devuelve su id."""
    resp = session.post(f"{base}/api/projects", json={"name": nombre, "labels": labels}, timeout=30)
    resp.raise_for_status()
    proyecto_id = int(resp.json()["id"])
    logger.info("Proyecto creado: '%s' (id %d).", nombre, proyecto_id)
    return proyecto_id


def crear_task(
    session: requests.Session, base: str, nombre: str, proyecto_id: int, segment_size: int
) -> int:
    """Crea una task vacía dentro del proyecto, con ``segment_size`` frames por job."""
    resp = session.post(
        f"{base}/api/tasks",
        json={"name": nombre, "project_id": proyecto_id, "segment_size": segment_size},
        timeout=30,
    )
    resp.raise_for_status()
    task_id = int(resp.json()["id"])
    logger.info("Task creada: '%s' (id %d, %d frames/job).", nombre, task_id, segment_size)
    return task_id


def subir_imagenes(
    session: requests.Session,
    base: str,
    task_id: int,
    imagenes: list[Path],
    image_quality: int,
    timeout_s: float,
) -> None:
    """Adjunta las imágenes a la task **en el orden dado** (``sorting_method: predefined``).

    Los bytes se leen a memoria en lugar de pasar file handles abiertos: con
    ~1000 archivos se evita el límite de descriptores del CRT de Windows.
    """
    files = [
        (f"client_files[{i}]", (img.name, img.read_bytes(), "image/jpeg"))
        for i, img in enumerate(imagenes)
    ]
    data: dict[str, Any] = {"image_quality": str(image_quality), "sorting_method": "predefined"}
    for i, img in enumerate(imagenes):
        data[f"upload_file_order[{i}]"] = img.name
    resp = session.post(
        f"{base}/api/tasks/{task_id}/data/",
        data=data,
        files=files,
        headers=_UPLOAD_UNICO,
        timeout=1800,
    )
    resp.raise_for_status()
    esperar_request(session, base, resp.json()["rq_id"], timeout_s)
    logger.info("Imágenes adjuntadas a la task %d: %d frames.", task_id, len(imagenes))


def importar_anotaciones(
    session: requests.Session,
    base: str,
    task_id: int,
    zip_path: Path,
    formato: str,
    timeout_s: float,
) -> None:
    """Importa un ZIP de anotaciones a la task (pre-anotaciones)."""
    with zip_path.open("rb") as fh:
        resp = session.post(
            f"{base}/api/tasks/{task_id}/annotations/",
            params={"format": formato},
            files={"annotation_file": (zip_path.name, fh, "application/zip")},
            headers=_UPLOAD_UNICO,
            timeout=600,
        )
    resp.raise_for_status()
    esperar_request(session, base, resp.json()["rq_id"], timeout_s)
    logger.info("Pre-anotaciones importadas a la task %d (%s).", task_id, zip_path.name)


def frames_de_task(
    session: requests.Session, base: str, task_id: int
) -> tuple[list[str], set[int]]:
    """Nombres de archivo en orden de frame y frames marcados como borrados."""
    resp = session.get(f"{base}/api/tasks/{task_id}/data/meta", timeout=30)
    resp.raise_for_status()
    meta = resp.json()
    return [f["name"] for f in meta["frames"]], {int(f) for f in meta.get("deleted_frames", [])}


def jobs_de_task(session: requests.Session, base: str, task_id: int) -> list[dict]:
    """Jobs de la task (incluye ``state``, ``stage``, ``start_frame``, ``stop_frame``, ``type``)."""
    return _listar(session, f"{base}/api/jobs", {"task_id": task_id})


# ---------------------------------------------------------------------------
# Export y backup
# ---------------------------------------------------------------------------


def exportar_job(
    session: requests.Session, base: str, job_id: int, formato: str, out: Path, timeout_s: float
) -> None:
    """Exporta las anotaciones de un job (sin imágenes) a un ZIP."""
    resp = session.post(
        f"{base}/api/jobs/{job_id}/dataset/export",
        params={"format": formato, "save_images": "false"},
        timeout=30,
    )
    resp.raise_for_status()
    _descargar(session, esperar_request(session, base, resp.json()["rq_id"], timeout_s), out)


def backup_proyecto(
    session: requests.Session, base: str, proyecto_id: int, out: Path, timeout_s: float
) -> None:
    """Backup completo del proyecto (anotaciones + media) a un ZIP restaurable."""
    resp = session.post(f"{base}/api/projects/{proyecto_id}/backup/export", timeout=30)
    resp.raise_for_status()
    _descargar(session, esperar_request(session, base, resp.json()["rq_id"], timeout_s), out)
