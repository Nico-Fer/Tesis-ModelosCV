"""Reconcilia las correcciones humanas de CVAT contra los pseudo-labels originales.

Produce el entregable que valida el pseudo-labeling en la tesis:

* **métrica de auditoría** (``audit_report.json``): tasa de corrección (% de
  keypoints modificados sobre el total auditado), desplazamiento medio en px y
  normalizado (fracción de la diagonal de la imagen) de los corregidos, y el
  desglose por estrato en dos ejes — ``motivo`` (sin_deteccion / multi_persona /
  score_bajo / control / …) y ``postura`` (sedente / brazo) — donde se ve si el
  teacher falla sistemáticamente en algún régimen;
* **gold set** (``<gold_root>/``): los frames auditados, verificados por
  humano, en una ubicación separada y trackeada. Es el único set con GT humano
  del proyecto. Cómo se usa lo decide ``aplicar.usar_en_train``
  (:mod:`src.audit.apply_corrections`): con ``true`` entra al train de los folds
  donde su sujeto es train y se evalúa en el fold donde es val (held-out **por
  sujeto**); con ``false`` se excluye de todo train (``gold_stems.txt`` es la
  lista de exclusión).

Definiciones de la métrica (documentadas acá porque el JSON las hereda):

* un keypoint cuenta como **corregido** si se movió más de
  ``reconcile.corregido_thr_px`` px, o si el humano lo agregó (``v: 0 → >0``) o
  lo eliminó (``v: >0 → 0``);
* los flips ``v=1 ↔ v=2`` se reportan aparte (``cambios_v1v2``) y NO entran en
  la tasa: Ultralytics binariza ``v>0``, así que no cambian lo que aprende el
  student;
* el denominador son los 17 slots de cada persona emparejada: dejar un slot en
  ``v=0`` también es una decisión de auditoría;
* las personas sin par (eliminadas o anotadas de cero, p.ej. frames sin
  detección) se reportan por separado: no hay pseudo contra el cual medir
  desplazamiento.

**Qué cuenta como auditado** lo dice ``<export>/auditados.txt`` (escrito por
``scripts/cvat_audit.py exportar`` a partir del estado de los jobs), nunca la
presencia en el export: CVAT exporta todos los frames pre-anotados, revisados o
no, y un frame revisado al que el anotador le borró todas las personas NO
aparece en el export (verificado el 2026-09-28). Por eso:

* frame de la selección fuera de ``auditados.txt`` ⇒ ``sin_auditar``, aunque
  esté en el export;
* frame auditado sin label en el export ⇒ gold vacío (``personas_eliminadas``).

Acepta auditorías **parciales** (jobs terminados de a tandas). El gold se
reconstruye de cero en cada corrida: si un job se reabre, sus frames salen.

Uso::

    python -m src.audit.reconcile_corrections --config configs/audit_cvat.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from src.audit.common import (
    AuditConfig,
    PoseLabel,
    build_config,
    image_size,
    load_config,
    match_persons,
    read_label_file,
)
from src.data.pseudo_label_vitpose import link_image
from src.data.uco_to_yolo import PersonAnnotation, yolo_label_line

logger = logging.getLogger(__name__)

GOLD_MANIFEST_FIELDS = [
    "stem",
    "subject",
    "exercise",
    "cam",
    "frame",
    "motivo",
    "postura",
    "n_personas_gold",
    "n_kpts_corregidos",
    "despl_medio_px",
    "verificado_por_humano",
]


def _nuevo_acumulador() -> dict[str, Any]:
    """Acumulador de un estrato (o del global)."""
    return {
        "frames": 0,
        "kpts_auditados": 0,
        "kpts_corregidos": 0,
        "kpts_movidos": 0,
        "kpts_agregados": 0,
        "kpts_eliminados": 0,
        "cambios_v1v2": 0,
        "personas_agregadas": 0,
        "personas_eliminadas": 0,
        "kpts_anotados_de_cero": 0,
        "despl_px": [],
        "despl_norm": [],
    }


def _finalizar(acum: dict[str, Any]) -> dict[str, Any]:
    """Convierte un acumulador en las métricas finales del reporte."""
    despl_px = acum.pop("despl_px")
    despl_norm = acum.pop("despl_norm")
    auditados = acum["kpts_auditados"]
    acum["tasa_correccion"] = round(acum["kpts_corregidos"] / auditados, 4) if auditados else 0.0
    acum["despl_medio_px"] = round(statistics.mean(despl_px), 2) if despl_px else 0.0
    acum["despl_mediana_px"] = round(statistics.median(despl_px), 2) if despl_px else 0.0
    acum["despl_max_px"] = round(max(despl_px), 2) if despl_px else 0.0
    acum["despl_medio_norm"] = round(statistics.mean(despl_norm), 5) if despl_norm else 0.0
    return acum


class FrameResult:
    """Resultado de reconciliar un frame: qué cambió y las personas gold."""

    def __init__(self) -> None:
        self.kpts_auditados = 0
        self.kpts_corregidos = 0
        self.kpts_movidos = 0
        self.kpts_agregados = 0
        self.kpts_eliminados = 0
        self.cambios_v1v2 = 0
        self.personas_agregadas = 0
        self.personas_eliminadas = 0
        self.kpts_anotados_de_cero = 0
        self.despl_px: list[float] = []
        self.despl_norm: list[float] = []


def reconcile_frame(
    originales: list[PoseLabel],
    corregidas: list[PoseLabel],
    img_w: int,
    img_h: int,
    thr_px: float,
) -> FrameResult:
    """Compara las dos versiones de un frame según las definiciones del módulo."""
    res = FrameResult()
    diag = math.hypot(img_w, img_h)
    pares, sin_par_orig, sin_par_corr = match_persons(originales, corregidas)

    for i, j in pares:
        for ka, kb in zip(originales[i].kpts, corregidas[j].kpts):
            res.kpts_auditados += 1
            if ka[2] == 0 and kb[2] == 0:
                continue
            if ka[2] == 0 and kb[2] > 0:
                res.kpts_agregados += 1
                res.kpts_corregidos += 1
                continue
            if ka[2] > 0 and kb[2] == 0:
                res.kpts_eliminados += 1
                res.kpts_corregidos += 1
                continue
            # Ambos anotados: se mide el desplazamiento en píxeles.
            dist_px = math.hypot((ka[0] - kb[0]) * img_w, (ka[1] - kb[1]) * img_h)
            if ka[2] != kb[2]:
                res.cambios_v1v2 += 1
            if dist_px > thr_px:
                res.kpts_movidos += 1
                res.kpts_corregidos += 1
                res.despl_px.append(dist_px)
                res.despl_norm.append(dist_px / diag)

    res.personas_eliminadas = len(sin_par_orig)
    res.personas_agregadas = len(sin_par_corr)
    for j in sin_par_corr:
        res.kpts_anotados_de_cero += sum(1 for k in corregidas[j].kpts if k[2] > 0)
    return res


def gold_label_lines(corregidas: list[PoseLabel], img_w: int, img_h: int) -> list[str]:
    """Reconstruye las líneas gold con el bbox convención GT (hull + 5 %).

    CVAT recalcula el bbox desde el skeleton al exportar (el del detector no
    sobrevive el round-trip), así que se rehace con la misma convención que
    :func:`src.data.uco_to_yolo.yolo_label_line`: byte-compatible con el GT.
    """
    lines: list[str] = []
    for persona in corregidas:
        kpts_px = [
            (x * img_w, y * img_h, v) if v > 0 else (0.0, 0.0, 0) for x, y, v in persona.kpts
        ]
        if all(v == 0 for _, _, v in kpts_px):
            logger.warning("Persona corregida sin keypoints visibles: descartada del gold.")
            continue
        ann = PersonAnnotation(image_path=Path(""), img_w=img_w, img_h=img_h, keypoints=kpts_px)
        lines.append(yolo_label_line(ann))
    return lines


def read_export(export_dir: Path) -> tuple[set[str], dict[str, Path]]:
    """Lee el directorio escrito por ``scripts/cvat_audit.py exportar``.

    Returns:
        ``(auditados, labels)`` — los stems revisados por humano y los labels
        exportados por stem (un frame auditado puede no tener label: el
        anotador borró todas las personas).

    Raises:
        FileNotFoundError: si no es un directorio de export o falta
            ``auditados.txt`` (sin esa lista no hay forma de saber qué se revisó).
    """
    if not export_dir.is_dir():
        raise FileNotFoundError(
            f"{export_dir} no es un directorio de export; generarlo con "
            "`python scripts/cvat_audit.py exportar`."
        )
    lista = export_dir / "auditados.txt"
    if not lista.exists():
        raise FileNotFoundError(
            f"Falta {lista}: sin la lista de jobs completados no se sabe qué frames se "
            "revisaron (CVAT exporta también los no revisados)."
        )
    auditados = {s for s in lista.read_text(encoding="utf-8").split() if s}
    labels = {p.stem: p for p in (export_dir / "labels").glob("*.txt")}
    return auditados, labels


def _reset_gold(gold_root: Path) -> tuple[Path, Path]:
    """Borra y recrea el pool del gold: es derivado y debe reflejar solo esta corrida."""
    gold_img_dir = gold_root / "images" / "all"
    gold_lbl_dir = gold_root / "labels" / "all"
    for d in (gold_img_dir, gold_lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    return gold_img_dir, gold_lbl_dir


def reconcile(cfg: AuditConfig, export_dir: Path) -> dict[str, Any]:
    """Corre la reconciliación completa y escribe el gold set + el reporte.

    Returns:
        El reporte (mismo contenido que ``audit_report.json``).

    Raises:
        FileNotFoundError: si el export no tiene ``auditados.txt``.
        ValueError: si un stem auditado no está en la selección (export de otro
            proyecto) o un label gold no cumple los invariantes del formato.
    """
    from src.audit.common import read_selection  # noqa: PLC0415
    from tests.test_pseudo_label_format import check_label_line  # noqa: PLC0415

    seleccion = read_selection(cfg.selection_csv)
    auditados, exportadas = read_export(export_dir)

    ajenos = sorted(auditados - {f["stem"] for f in seleccion})
    if ajenos:
        raise ValueError(
            f"{len(ajenos)} stems auditados no están en {cfg.selection_csv} "
            f"(p.ej. {ajenos[0]}): el export es de otro proyecto o de otra selección."
        )

    gold_img_dir, gold_lbl_dir = _reset_gold(cfg.gold_root)

    global_acum = _nuevo_acumulador()
    por_motivo: dict[str, dict[str, Any]] = {}
    por_postura: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, str]] = []
    sin_auditar: list[str] = []
    auditados_vacios: list[str] = []

    for fila in seleccion:
        stem = fila["stem"]
        if stem not in auditados:
            sin_auditar.append(stem)
            continue

        img_w, img_h = image_size(cfg.images_dir / f"{stem}.jpg")
        originales = read_label_file(cfg.labels_dir / f"{stem}.txt")
        if stem in exportadas:
            corregidas = read_label_file(exportadas[stem])
        else:
            corregidas = []
            auditados_vacios.append(stem)
        res = reconcile_frame(originales, corregidas, img_w, img_h, cfg.corregido_thr_px)

        for acum in (
            global_acum,
            por_motivo.setdefault(fila["motivo"], _nuevo_acumulador()),
            por_postura.setdefault(fila["postura"], _nuevo_acumulador()),
        ):
            acum["frames"] += 1
            for key in (
                "kpts_auditados",
                "kpts_corregidos",
                "kpts_movidos",
                "kpts_agregados",
                "kpts_eliminados",
                "cambios_v1v2",
                "personas_agregadas",
                "personas_eliminadas",
                "kpts_anotados_de_cero",
            ):
                acum[key] += getattr(res, key)
            acum["despl_px"].extend(res.despl_px)
            acum["despl_norm"].extend(res.despl_norm)

        lines = gold_label_lines(corregidas, img_w, img_h)
        for line in lines:
            check_label_line(line)
        (gold_lbl_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        link_image(cfg.images_dir / f"{stem}.jpg", gold_img_dir / f"{stem}.jpg", cfg.link_mode)

        manifest_rows.append(
            {
                "stem": stem,
                "subject": fila["subject"],
                "exercise": fila["exercise"],
                "cam": fila["cam"],
                "frame": fila["frame"],
                "motivo": fila["motivo"],
                "postura": fila["postura"],
                "n_personas_gold": str(len(lines)),
                "n_kpts_corregidos": str(res.kpts_corregidos),
                "despl_medio_px": (
                    f"{statistics.mean(res.despl_px):.2f}" if res.despl_px else "0.00"
                ),
                "verificado_por_humano": "1",
            }
        )

    if auditados_vacios:
        # UCO siempre tiene un paciente en cuadro: un frame auditado sin persona
        # es raro y vale la pena revisarlo, pero es una decisión válida.
        logger.warning(
            "%d frames auditados quedaron SIN personas (p.ej. %s): el anotador borró todo.",
            len(auditados_vacios),
            auditados_vacios[0],
        )

    manifest_rows.sort(key=lambda r: r["stem"])
    with (cfg.gold_root / "gold_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=GOLD_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (cfg.gold_root / "gold_stems.txt").write_text(
        "\n".join(r["stem"] for r in manifest_rows) + "\n", encoding="utf-8"
    )

    report = {
        "generado": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "export": str(export_dir),
        "corregido_thr_px": cfg.corregido_thr_px,
        "definiciones": {
            "corregido": "desplazamiento > thr, o v 0→>0 (agregado), o v >0→0 (eliminado)",
            "cambios_v1v2": "flips v=1↔v=2; NO cuentan en la tasa (Ultralytics binariza v>0)",
            "denominador": "17 slots por persona emparejada",
            "despl_medio_norm": "fracción de la diagonal de la imagen",
        },
        "n_frames_seleccionados": len(seleccion),
        "n_frames_auditados": global_acum["frames"],
        "n_frames_sin_auditar": len(sin_auditar),
        "n_frames_auditados_sin_personas": len(auditados_vacios),
        "global": _finalizar(global_acum),
        "por_motivo": {k: _finalizar(v) for k, v in sorted(por_motivo.items())},
        "por_postura": {k: _finalizar(v) for k, v in sorted(por_postura.items())},
    }
    report_path = cfg.gold_root / "audit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    g = report["global"]
    logger.info(
        "Reconciliación: %d/%d frames auditados (%d pendientes).",
        report["n_frames_auditados"],
        report["n_frames_seleccionados"],
        report["n_frames_sin_auditar"],
    )
    logger.info(
        "Tasa de corrección global: %.2f %% (%d de %d kpts auditados); "
        "desplazamiento medio de los corregidos: %.2f px (%.4f de la diagonal).",
        100 * g["tasa_correccion"],
        g["kpts_corregidos"],
        g["kpts_auditados"],
        g["despl_medio_px"],
        g["despl_medio_norm"],
    )
    logger.info(
        "Gold set: %d frames → %s (lista: gold_stems.txt).",
        len(manifest_rows),
        cfg.gold_root,
    )
    logger.info("Reporte de tesis: %s", report_path)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML de auditoría.")
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Directorio de `cvat_audit.py exportar` (default: reconcile.export_dir).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = build_config(load_config(args.config))
    reconcile(cfg, args.export or cfg.export_dir)


if __name__ == "__main__":
    main()
