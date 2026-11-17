"""Operación de la auditoría en CVAT: montar la task, ver progreso, exportar y respaldar.

Subcomandos (cualquier venv; solo requests + stdlib + ``src.audit``)::

    # crea proyecto + task por jobs, sube imágenes en orden, importa pre-anotaciones
    # y escribe el mapa frame → job (cvat.frames_csv)
    python scripts/cvat_audit.py montar --config configs/audit_cvat.yaml --project auditoria-v2

    # progreso por job
    python scripts/cvat_audit.py estado --config configs/audit_cvat.yaml

    # exporta SOLO los jobs en cvat.estado_auditado (completed) → reconcile.export_dir
    python scripts/cvat_audit.py exportar --config configs/audit_cvat.yaml

    # backup completo del proyecto (las horas de anotación viven en volúmenes Docker)
    python scripts/cvat_audit.py backup --config configs/audit_cvat.yaml --project-id <id>

``auditados.txt`` se deriva del **estado de los jobs**, nunca de qué frames
aparecen en el export: CVAT exporta todos los frames pre-anotados, revisados o no.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cvat_client as cc  # noqa: E402

from src.audit.common import AuditConfig, build_config, load_config  # noqa: E402
from src.audit.cvat_frames import (  # noqa: E402
    build_frame_map,
    read_frame_map,
    stems_auditados,
    write_frame_map,
)
from src.audit.pack_for_cvat import anotaciones_zip  # noqa: E402

logger = logging.getLogger("cvat_audit")


# ---------------------------------------------------------------------------
# montar
# ---------------------------------------------------------------------------


def montar(
    cfg: AuditConfig,
    session,
    base: str,
    *,
    pack_dir: Path,
    proyecto_id: int | None,
    proyecto_nombre: str | None,
    task_nombre: str,
    segment_size: int,
    frames_csv: Path,
    sobrescribir_mapa: bool,
    timeout_s: float,
) -> tuple[int, int, list[dict[str, str]]]:
    """Crea (o reusa) el proyecto, crea la task y escribe el mapa frame → job.

    Returns:
        ``(proyecto_id, task_id, mapa)``.

    Raises:
        FileExistsError: si ``frames_csv`` ya existe y no se pidió sobrescribir
            (perder el mapa de una task real deja la auditoría sin señal de
            auditado).
        ValueError: si CVAT no quedó con exactamente los frames del pack.
    """
    if frames_csv.exists() and not sobrescribir_mapa:
        raise FileExistsError(
            f"{frames_csv} ya existe (mapa de otra task). Usar --sobrescribir-mapa solo si esa "
            "task se descartó, o --frames-csv con otra ruta."
        )
    orden = [s for s in (pack_dir / "orden.txt").read_text(encoding="utf-8").split() if s]
    imagenes = [pack_dir / "images" / cfg.subset / f"{stem}.jpg" for stem in orden]
    faltantes = [p for p in imagenes if not p.exists()]
    if faltantes:
        raise FileNotFoundError(f"Faltan {len(faltantes)} imágenes del pack, p.ej. {faltantes[0]}")
    zip_path = anotaciones_zip(pack_dir)
    if not zip_path.exists():
        raise FileNotFoundError(f"Falta {zip_path}; correr src.audit.pack_for_cvat.")

    if proyecto_id is None:
        if not proyecto_nombre:
            raise ValueError("Indicar --project <nombre> (nuevo) o --project-id <id> (existente).")
        spec = json.loads(
            (pack_dir.parent / "cvat_skeleton_labels.json").read_text(encoding="utf-8")
        )
        proyecto_id = cc.crear_proyecto(session, base, proyecto_nombre, spec)

    task_id = cc.crear_task(session, base, task_nombre, proyecto_id, segment_size)
    cc.subir_imagenes(session, base, task_id, imagenes, cfg.image_quality, timeout_s)
    cc.importar_anotaciones(session, base, task_id, zip_path, cfg.cvat_formato, timeout_s)

    nombres, _ = cc.frames_de_task(session, base, task_id)
    jobs = cc.jobs_de_task(session, base, task_id)
    mapa = build_frame_map(task_id, nombres, jobs)

    stems_cvat = [f["stem"] for f in mapa]
    if sorted(stems_cvat) != sorted(orden):
        raise ValueError(
            f"La task {task_id} no tiene exactamente los frames del pack "
            f"({len(stems_cvat)} en CVAT vs {len(orden)} en el pack)."
        )
    if stems_cvat != orden:
        logger.warning(
            "CVAT no respetó el orden mezclado del pack: el mapa sigue siendo correcto "
            "(sale de data/meta), pero el control podría quedar agrupado."
        )
    n_jobs = len({f["job_id"] for f in mapa})
    esperados = math.ceil(len(orden) / segment_size)
    if n_jobs != esperados:
        raise ValueError(f"La task {task_id} tiene {n_jobs} jobs; se esperaban {esperados}.")

    write_frame_map(mapa, frames_csv)
    logger.info(
        "Montado: proyecto %d, task %d, %d frames en %d jobs → %s/tasks/%d. Mapa: %s",
        proyecto_id,
        task_id,
        len(mapa),
        n_jobs,
        base,
        task_id,
        frames_csv,
    )
    return proyecto_id, task_id, mapa


# ---------------------------------------------------------------------------
# estado / exportar
# ---------------------------------------------------------------------------


def _jobs_del_mapa(session, base: str, mapa: list[dict[str, str]]) -> dict[int, dict]:
    """Jobs de anotación actuales de las tasks del mapa, por id."""
    jobs: dict[int, dict] = {}
    for task_id in sorted({int(f["task_id"]) for f in mapa}):
        for job in cc.jobs_de_task(session, base, task_id):
            if job.get("type", "annotation") == "annotation":
                jobs[int(job["id"])] = job
    return jobs


def estado(cfg: AuditConfig, session, base: str, frames_csv: Path) -> None:
    """Imprime el estado de cada job y el avance de la auditoría."""
    mapa = read_frame_map(frames_csv)
    jobs = _jobs_del_mapa(session, base, mapa)
    frames_por_job: dict[int, int] = {}
    for fila in mapa:
        frames_por_job[int(fila["job_id"])] = frames_por_job.get(int(fila["job_id"]), 0) + 1
    print(f"{'job':>6} {'frames':>6}  {'stage':<11} {'state':<12} asignado")
    for job_id in sorted(frames_por_job):
        job = jobs.get(job_id, {})
        asignado = (job.get("assignee") or {}).get("username", "-")
        print(
            f"{job_id:>6} {frames_por_job[job_id]:>6}  {job.get('stage', '?'):<11} "
            f"{job.get('state', 'NO EXISTE'):<12} {asignado}"
        )
    auditados = stems_auditados(mapa, {j: v["state"] for j, v in jobs.items()}, cfg.estado_auditado)
    print(f"\nAuditados ({cfg.estado_auditado}): {len(auditados)}/{len(mapa)} frames.")


def exportar(
    cfg: AuditConfig,
    session,
    base: str,
    *,
    frames_csv: Path,
    out_dir: Path,
    todos: bool,
    timeout_s: float,
) -> dict:
    """Exporta los jobs completados a ``out_dir`` y escribe ``auditados.txt``.

    Con ``todos`` exporta también los jobs no completados (lo usa el gate de
    round-trip); ``auditados.txt`` sigue listando solo los completados.

    Returns:
        La metadata escrita en ``export_meta.json``.
    """
    mapa = read_frame_map(frames_csv)
    jobs = _jobs_del_mapa(session, base, mapa)
    estado_por_job = {j: v["state"] for j, v in jobs.items()}

    borrados: dict[int, set[int]] = {}
    for task_id in sorted({int(f["task_id"]) for f in mapa}):
        _, borrados[task_id] = cc.frames_de_task(session, base, task_id)
        if borrados[task_id]:
            logger.warning(
                "Task %d: %d frames borrados en CVAT; no cuentan como auditados.",
                task_id,
                len(borrados[task_id]),
            )
    auditados = stems_auditados(mapa, estado_por_job, cfg.estado_auditado, borrados)

    jobs_mapa = sorted({int(f["job_id"]) for f in mapa})
    a_exportar = [j for j in jobs_mapa if todos or estado_por_job.get(j) == cfg.estado_auditado]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    labels_dir = out_dir / "labels"
    zips_dir = out_dir / "zips"
    labels_dir.mkdir(parents=True)
    zips_dir.mkdir()

    stems_mapa = {f["stem"] for f in mapa}
    n_labels = 0
    for job_id in a_exportar:
        zip_path = zips_dir / f"job{job_id}.zip"
        cc.exportar_job(session, base, job_id, cfg.cvat_formato, zip_path, timeout_s)
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            for label in Path(tmp).rglob("*.txt"):
                if "labels" not in label.parts:
                    continue
                if label.stem not in stems_mapa:
                    raise ValueError(
                        f"El export del job {job_id} trae {label.stem}, ausente del mapa "
                        f"{frames_csv}: el mapa es de otra task."
                    )
                shutil.copyfile(label, labels_dir / label.name)
                n_labels += 1

    (out_dir / "auditados.txt").write_text("".join(f"{s}\n" for s in auditados), encoding="utf-8")
    meta = {
        "generado": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cvat_version": cc.version_server(session, base),
        "estado_auditado": cfg.estado_auditado,
        "frames_csv": str(frames_csv),
        "n_frames_mapa": len(mapa),
        "n_auditados": len(auditados),
        "n_labels_exportados": n_labels,
        "jobs": [
            {
                "id": j,
                "task_id": int(jobs[j]["task_id"]) if j in jobs else None,
                "state": estado_por_job.get(j, "NO EXISTE"),
                "stage": jobs.get(j, {}).get("stage"),
                "exportado": j in a_exportar,
            }
            for j in jobs_mapa
        ],
    }
    (out_dir / "export_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "Export: %d/%d jobs exportados, %d labels; auditados (%s): %d/%d frames → %s",
        len(a_exportar),
        len(jobs_mapa),
        n_labels,
        cfg.estado_auditado,
        len(auditados),
        len(mapa),
        out_dir,
    )
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    comunes = argparse.ArgumentParser(add_help=False)
    comunes.add_argument("--config", type=Path, required=True, help="YAML de auditoría.")
    comunes.add_argument(
        "--credentials",
        type=Path,
        default=Path("cvat_credentials.local"),
        help="Archivo KEY=VALUE con CVAT_URL/CVAT_USER/CVAT_PASS.",
    )
    comunes.add_argument(
        "--frames-csv", type=Path, default=None, help="Mapa frame → job (default: cvat.frames_csv)."
    )
    comunes.add_argument(
        "--timeout", type=float, default=1800.0, help="Timeout (s) por request asíncrono."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_montar = sub.add_parser("montar", parents=[comunes], help="Crear task y mapa frame → job.")
    p_montar.add_argument("--project", default=None, help="Nombre de un proyecto NUEVO.")
    p_montar.add_argument("--project-id", type=int, default=None, help="Proyecto existente.")
    p_montar.add_argument("--task", default="auditoria", help="Nombre de la task.")
    p_montar.add_argument("--pack-dir", type=Path, default=None, help="Default: pack.out_dir.")
    p_montar.add_argument(
        "--segment-size", type=int, default=None, help="Default: cvat.segment_size."
    )
    p_montar.add_argument(
        "--sobrescribir-mapa",
        action="store_true",
        help="Pisar un frames_csv existente (solo si su task se descartó).",
    )

    sub.add_parser("estado", parents=[comunes], help="Progreso por job.")

    p_exp = sub.add_parser("exportar", parents=[comunes], help="Exportar jobs completados.")
    p_exp.add_argument("--out", type=Path, default=None, help="Default: reconcile.export_dir.")
    p_exp.add_argument(
        "--todos", action="store_true", help="Exportar también jobs no completados (gate)."
    )

    p_bak = sub.add_parser("backup", parents=[comunes], help="Backup del proyecto.")
    p_bak.add_argument("--project-id", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = build_config(load_config(args.config))
    frames_csv = args.frames_csv or cfg.frames_csv
    session, base = cc.sesion(args.credentials)

    if args.cmd == "montar":
        montar(
            cfg,
            session,
            base,
            pack_dir=args.pack_dir or cfg.pack_dir,
            proyecto_id=args.project_id,
            proyecto_nombre=args.project,
            task_nombre=args.task,
            segment_size=args.segment_size or cfg.segment_size,
            frames_csv=frames_csv,
            sobrescribir_mapa=args.sobrescribir_mapa,
            timeout_s=args.timeout,
        )
    elif args.cmd == "estado":
        estado(cfg, session, base, frames_csv)
    elif args.cmd == "exportar":
        exportar(
            cfg,
            session,
            base,
            frames_csv=frames_csv,
            out_dir=args.out or cfg.export_dir,
            todos=args.todos,
            timeout_s=args.timeout,
        )
    elif args.cmd == "backup":
        out = cfg.backups_dir / f"proyecto{args.project_id}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        cc.backup_proyecto(session, base, args.project_id, out, args.timeout)


if __name__ == "__main__":
    main()
