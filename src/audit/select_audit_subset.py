"""Selección del subconjunto de auditoría humana sobre el manifest de pseudo-labels.

Entran a la auditoría:

* los frames con ``baja_confianza=1`` del teacher (sin detección, sin
  keypoints válidos, score bajo);
* los frames donde el pseudo-label **discrepa del GT de UCO** en las
  articulaciones del ejercicio (error > ``gt.discrepancia_thr_px`` o una
  articulación visible en el GT perdida a ``v=0``). Es el criterio que detecta
  errores reales: medido el 2026-09-28, ``baja_confianza`` marcaba solo 3 de
  los 339 frames con error > 100 px;
* una muestra **control** de frames sanos, estratificada por ejercicio × sujeto:
  sin control, la tasa de corrección medida después no tiene línea base.

Los ``multi_persona`` sin discrepancia se excluyen (con
``seleccion.excluir_multi_sin_discrepancia``) salvo una muestra estratificada:
con ``max_persons: 1`` se escribe un solo esqueleto y la única duda es si es el
paciente, cosa que el GT ya confirma cuando coincide. Su distribución de error
contra el GT es la del control (p99 48 vs 47 px).

Cada frame queda clasificado en dos ejes de estrato:

* ``motivo`` — por qué se audita, con precedencia ``sin_deteccion`` >
  ``sin_kpts_validos`` > ``discrepancia_gt`` > ``score_bajo`` >
  ``multi_persona`` (o ``multi_persona_muestra``) > ``control``;
* ``postura`` — régimen del ejercicio según ``estratos.postura`` del YAML
  (``sedente`` | ``brazo``).

Uso::

    python -m src.audit.select_audit_subset --config configs/audit_cvat.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.audit.common import (
    SELECTION_FIELDS,
    AuditConfig,
    DiscrepanciaGT,
    build_config,
    discrepancia_gt,
    image_size,
    load_config,
    postura_de,
    read_label_file,
)
from src.data.pseudo_label_vitpose import stratified_sample
from src.data.uco_to_yolo import parse_stem

logger = logging.getLogger(__name__)

# El manifest guarda mean_kpt_score redondeado a 6 decimales: en esa banda
# alrededor del umbral, recalcular el flag no es confiable y decide el manifest.
_BANDA_REDONDEO = 1e-6


def review_thresholds(run_metadata: Path) -> tuple[float, int]:
    """Umbrales del flag ``baja_confianza`` con los que corrió el pseudo-labeling.

    Se leen de ``run_metadata.json`` (la config efectiva de esa corrida) y no del
    YAML actual, que pudo cambiar después.

    Raises:
        FileNotFoundError: si no existe el ``run_metadata.json``.
        ValueError: si no tiene la sección ``review``.
    """
    if not run_metadata.exists():
        raise FileNotFoundError(f"No existe {run_metadata} (metadata del pseudo-labeling).")
    meta = json.loads(run_metadata.read_text(encoding="utf-8"))
    review = (meta.get("config") or {}).get("review")
    if not review:
        raise ValueError(f"{run_metadata} no registra config.review (umbrales del flag).")
    return float(review["mean_score_thr"]), int(review["min_visible_kpts"])


def _score_bajo(row: dict[str, str], mean_thr: float, min_vis: int) -> bool:
    """Condición de score bajo del flag, tolerante al redondeo del manifest."""
    mean = float(row["mean_kpt_score"])
    if int(row["n_v2"]) < min_vis:
        return True
    if abs(mean - mean_thr) <= _BANDA_REDONDEO:
        # Ambiguo por redondeo: si el manifest lo marcó y ninguna otra regla lo
        # explica, fue el score.
        return int(row["baja_confianza"]) == 1 and int(row["n_personas_detectadas"]) == 1
    return mean < mean_thr


def motivo_de(row: dict[str, str], discrepante: bool, mean_thr: float, min_vis: int) -> str | None:
    """Clasifica una fila del manifest por su causa de auditoría (``None`` = sano).

    La precedencia importa: un frame sin detección también tiene score 0, pero
    su corrección es la más cara (se anota de cero) y debe contarse aparte; la
    discrepancia con el GT va antes que los motivos de confianza porque es
    evidencia de error, no una sospecha.
    """
    if int(row["n_personas_detectadas"]) == 0:
        return "sin_deteccion"
    if int(row["n_personas_escritas"]) == 0:
        return "sin_kpts_validos"
    if discrepante:
        return "discrepancia_gt"
    if _score_bajo(row, mean_thr, min_vis):
        return "score_bajo"
    if int(row["n_personas_detectadas"]) > 1:
        return "multi_persona"
    return None


def _verificar_consistencia(row: dict[str, str], mean_thr: float, min_vis: int) -> None:
    """Aborta si el flag recalculado no coincide con el ``baja_confianza`` del manifest.

    Una diferencia significa que los umbrales leídos no son los de la corrida
    (o que el manifest está corrupto): la clasificación por motivo sería falsa.
    """
    recalculado = (
        int(row["n_personas_detectadas"]) != 1
        or int(row["n_personas_escritas"]) == 0
        or _score_bajo(row, mean_thr, min_vis)
    )
    if recalculado != (int(row["baja_confianza"]) == 1):
        raise ValueError(
            f"{row['image']}: baja_confianza={row['baja_confianza']} en el manifest pero las "
            f"reglas (mean_score_thr={mean_thr}, min_visible_kpts={min_vis}) dan "
            f"{int(recalculado)}. ¿run_metadata de otra corrida?"
        )


def _verificar_img_size(cfg: AuditConfig, stem: str) -> None:
    """``gt.img_size`` debe coincidir con las imágenes reales (se escala el error a px)."""
    real = image_size(cfg.images_dir / f"{stem}.jpg")
    if real != cfg.gt_img_size:
        raise ValueError(
            f"gt.img_size={list(cfg.gt_img_size)} pero {stem}.jpg mide {list(real)}; "
            "el error en px contra el GT saldría mal escalado."
        )


def _discrepancia_de(cfg: AuditConfig, stem: str) -> DiscrepanciaGT:
    """Lee GT + pseudo de un frame y los compara.

    Raises:
        FileNotFoundError: si falta el label GT o el pseudo (pools desalineados).
    """
    gt_path = cfg.gt_labels_dir / f"{stem}.txt"
    pseudo_path = cfg.labels_dir / f"{stem}.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Falta el label GT de {stem}: {gt_path}")
    if not pseudo_path.exists():
        raise FileNotFoundError(f"Falta el pseudo-label de {stem}: {pseudo_path}")
    w, h = cfg.gt_img_size
    return discrepancia_gt(read_label_file(gt_path), read_label_file(pseudo_path), w, h)


def select(cfg: AuditConfig, workers: int = 16) -> list[dict[str, str]]:
    """Arma la selección: motivos de auditoría + muestras estratificadas.

    Args:
        cfg: configuración de la auditoría.
        workers: hilos de lectura de labels (148k archivos chicos: domina la
            latencia de apertura en NTFS, no la CPU).

    Returns:
        Filas con las columnas de :data:`~src.audit.common.SELECTION_FIELDS`,
        ordenadas por stem (determinista).

    Raises:
        FileNotFoundError: si falta el manifest, el run_metadata o algún label.
        ValueError: ejercicio sin postura, flag inconsistente o img_size errado.
    """
    if not cfg.manifest.exists():
        raise FileNotFoundError(f"No existe el manifest de pseudo-labels: {cfg.manifest}")

    with cfg.manifest.open("r", encoding="utf-8", newline="") as fh:
        manifest = list(csv.DictReader(fh))
    if not manifest:
        raise ValueError(f"Manifest vacío: {cfg.manifest}")
    logger.info("Manifest leído: %d frames en %s.", len(manifest), cfg.manifest)

    mean_thr, min_vis = review_thresholds(cfg.run_metadata)
    stems = [Path(row["image"]).stem for row in manifest]
    _verificar_img_size(cfg, stems[0])

    logger.info("Comparando pseudo-labels contra el GT de UCO (%s)...", cfg.gt_labels_dir)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        discrepancias = list(pool.map(lambda s: _discrepancia_de(cfg, s), stems))

    filas: list[dict[str, str]] = []
    sanos: list[str] = []
    multi: list[str] = []
    por_stem: dict[str, tuple[dict[str, str], DiscrepanciaGT]] = {}
    for row, stem, disc in zip(manifest, stems, discrepancias):
        discrepante = disc.es_discrepante(cfg.discrepancia_thr_px)
        motivo = motivo_de(row, discrepante, mean_thr, min_vis)
        _verificar_consistencia(row, mean_thr, min_vis)
        por_stem[stem] = (row, disc)
        if motivo is None:
            sanos.append(stem)
        elif motivo == "multi_persona" and cfg.excluir_multi_sin_discrepancia:
            multi.append(stem)
        else:
            filas.append(_fila(cfg, row, motivo, disc, discrepante))

    control = stratified_sample(sanos, cfg.stratify_by, cfg.control_size, cfg.seed)
    for stem in control:
        row, disc = por_stem[stem]
        filas.append(_fila(cfg, row, "control", disc, False))

    if cfg.excluir_multi_sin_discrepancia:
        muestra = stratified_sample(multi, cfg.stratify_by, cfg.muestra_multi_persona, cfg.seed)
        for stem in muestra:
            row, disc = por_stem[stem]
            filas.append(_fila(cfg, row, "multi_persona_muestra", disc, False))
        logger.info(
            "multi_persona sin discrepancia con el GT: %d frames, se audita una muestra de %d "
            "(el resto queda excluido: el GT confirma que el esqueleto es del paciente).",
            len(multi),
            len(muestra),
        )

    logger.info(
        "Selección: %d frames (control de %d sanos, estratificado por %s, seed %d).",
        len(filas),
        len(sanos),
        " × ".join(cfg.stratify_by),
        cfg.seed,
    )
    filas.sort(key=lambda f: f["stem"])

    seleccionados = [f["stem"] for f in filas]
    if len(seleccionados) != len(set(seleccionados)):
        raise ValueError("Stems duplicados en la selección; el manifest está corrupto.")
    return filas


def _fila(
    cfg: AuditConfig,
    row: dict[str, str],
    motivo: str,
    disc: DiscrepanciaGT,
    discrepante: bool,
) -> dict[str, str]:
    """Proyecta una fila del manifest a las columnas de selection.csv."""
    stem = Path(row["image"]).stem
    parts = parse_stem(stem)
    return {
        "stem": stem,
        "motivo": motivo,
        "postura": postura_de(cfg, parts.exercise),
        "subject": parts.subject,
        "exercise": parts.exercise,
        "cam": parts.cam,
        "frame": str(parts.frame),
        "det_score": row["det_score"],
        "mean_kpt_score": row["mean_kpt_score"],
        "n_personas_escritas": row["n_personas_escritas"],
        "n_v2": row["n_v2"],
        "n_v1": row["n_v1"],
        "n_v0": row["n_v0"],
        "err_gt_px": "" if disc.err_px is None else f"{disc.err_px:.1f}",
        "kpt_perdido": str(int(disc.kpt_perdido)),
        "discrepancia_gt": str(int(discrepante)),
    }


def write_selection(filas: list[dict[str, str]], out_csv: Path) -> None:
    """Escribe ``selection.csv`` y loguea el desglose por estrato."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        writer.writerows(filas)

    por_motivo = Counter(f["motivo"] for f in filas)
    por_postura = Counter(f["postura"] for f in filas)
    logger.info("selection.csv escrito: %d frames → %s", len(filas), out_csv)
    logger.info("Por motivo: %s", dict(sorted(por_motivo.items())))
    logger.info("Por postura: %s", dict(sorted(por_postura.items())))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML de auditoría.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = build_config(load_config(args.config))
    write_selection(select(cfg), cfg.selection_csv)


if __name__ == "__main__":
    main()
