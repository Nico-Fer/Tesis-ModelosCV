"""Config y helpers compartidos de la auditoría de pseudo-labels vía CVAT.

Centraliza el parseo de ``configs/audit_cvat.yaml``, la lectura de labels
YOLO-pose (byte-compatibles con el resto del repo) y el matching de personas
entre dos versiones del mismo frame. Framework-free: stdlib + cv2.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.inference.keypoint_schema import NUM_COCO_KEYPOINTS

logger = logging.getLogger(__name__)

# 1 clase + 4 bbox + 17×(x, y, v) = 56 tokens por línea de label.
EXPECTED_TOKENS = 5 + NUM_COCO_KEYPOINTS * 3


@dataclass(frozen=True)
class AuditConfig:
    """Configuración completa de la auditoría, con defaults aplicados."""

    manifest: Path
    images_dir: Path
    labels_dir: Path
    folds_dir: Path
    run_metadata: Path

    gt_labels_dir: Path
    gt_img_size: tuple[int, int]
    discrepancia_thr_px: float

    control_size: int
    stratify_by: tuple[str, ...]
    seed: int
    selection_csv: Path
    excluir_multi_sin_discrepancia: bool
    muestra_multi_persona: int

    postura: dict[str, str]

    pack_dir: Path
    link_mode: str
    subset: str

    cvat_url: str
    cvat_formato: str
    label_name: str
    segment_size: int
    image_quality: int
    estado_auditado: str
    frames_csv: Path
    backups_dir: Path

    roundtrip_tol_px: float

    export_dir: Path
    corregido_thr_px: float
    gold_root: Path

    corrected_root: Path
    corrected_yaml: Path
    usar_en_train: bool
    n_folds: int
    coco_manifest: Path | None
    coco_fold: int


def load_config(path: Path) -> dict[str, Any]:
    """Lee el YAML crudo (ver ``configs/audit_cvat.yaml``)."""
    import yaml  # noqa: PLC0415

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_config(raw: dict[str, Any]) -> AuditConfig:
    """Aplica defaults y valida el YAML crudo.

    Raises:
        ValueError: si falta una clave obligatoria o un valor está fuera de rango.
    """
    pseudo = raw.get("pseudo") or {}
    gt = raw.get("gt") or {}
    sel = raw.get("seleccion") or {}
    estratos = raw.get("estratos") or {}
    pack = raw.get("pack") or {}
    cvat = raw.get("cvat") or {}
    rt = raw.get("roundtrip") or {}
    rec = raw.get("reconcile") or {}
    aplicar = raw.get("aplicar") or {}
    coco = aplicar.get("mix_coco") or {}

    postura = {str(k): str(v) for k, v in (estratos.get("postura") or {}).items()}
    if not postura:
        raise ValueError("Falta estratos.postura en la configuración (mapa ejercicio → postura).")

    link_mode = str(pack.get("link_mode", "hardlink"))
    if link_mode not in ("hardlink", "copy", "symlink"):
        raise ValueError(f"link_mode inválido: {link_mode!r} (hardlink | copy | symlink).")

    img_size = tuple(int(v) for v in gt.get("img_size", (1280, 720)))
    if len(img_size) != 2 or min(img_size) <= 0:
        raise ValueError(f"gt.img_size debe ser [ancho, alto] positivos (recibido {img_size}).")

    segment_size = int(cvat.get("segment_size", 100))
    image_quality = int(cvat.get("image_quality", 95))
    if segment_size <= 0:
        raise ValueError(f"cvat.segment_size debe ser > 0 (recibido {segment_size}).")
    if not 1 <= image_quality <= 100:
        raise ValueError(f"cvat.image_quality debe estar en [1, 100] (recibido {image_quality}).")

    labels_dir = Path(pseudo.get("labels_dir", "data/processed/uco_pseudo/labels/all"))
    return AuditConfig(
        manifest=Path(pseudo.get("manifest", "data/processed/uco_pseudo/manifest.csv")),
        images_dir=Path(pseudo.get("images_dir", "data/processed/uco_pseudo/images/all")),
        labels_dir=labels_dir,
        folds_dir=Path(pseudo.get("folds_dir", "data/processed/uco_pseudo/folds")),
        run_metadata=Path(
            pseudo.get("run_metadata", "data/processed/uco_pseudo/run_metadata.json")
        ),
        gt_labels_dir=Path(gt.get("labels_dir", "data/processed/uco_yolo/labels/all")),
        gt_img_size=img_size,  # type: ignore[arg-type]
        discrepancia_thr_px=float(gt.get("discrepancia_thr_px", 100.0)),
        control_size=int(sel.get("control_size", 150)),
        stratify_by=tuple(sel.get("stratify_by", ("exercise", "subject"))),
        seed=int(sel.get("seed", 42)),
        selection_csv=Path(sel.get("out_csv", "data/processed/uco_audit/selection.csv")),
        excluir_multi_sin_discrepancia=bool(sel.get("excluir_multi_sin_discrepancia", True)),
        muestra_multi_persona=int(sel.get("muestra_multi_persona", 50)),
        postura=postura,
        pack_dir=Path(pack.get("out_dir", "data/processed/uco_audit/cvat_pack")),
        link_mode=link_mode,
        subset=str(pack.get("subset", "train")),
        cvat_url=str(cvat.get("url", "http://localhost:8080")),
        cvat_formato=str(cvat.get("formato", "Ultralytics YOLO Pose 1.0")),
        label_name=str(cvat.get("label_name", "person")),
        segment_size=segment_size,
        image_quality=image_quality,
        estado_auditado=str(cvat.get("estado_auditado", "completed")),
        frames_csv=Path(cvat.get("frames_csv", "data/processed/uco_audit/cvat_frames.csv")),
        backups_dir=Path(cvat.get("backups_dir", "data/processed/uco_audit/backups")),
        roundtrip_tol_px=float(rt.get("tol_px", 1.0)),
        export_dir=Path(rec.get("export_dir", "data/processed/uco_audit/cvat_export")),
        corregido_thr_px=float(rec.get("corregido_thr_px", 3.0)),
        gold_root=Path(rec.get("gold_root", "data/processed/uco_gold")),
        corrected_root=Path(aplicar.get("out_root", "data/processed/uco_pseudo_corrected")),
        corrected_yaml=Path(
            aplicar.get("out_yaml", "data/processed/pose_dataset_pseudo_corrected.yaml")
        ),
        usar_en_train=bool(aplicar.get("usar_en_train", True)),
        n_folds=int(aplicar.get("n_folds", 5)),
        coco_manifest=Path(coco["manifest"]) if coco.get("manifest") else None,
        coco_fold=int(coco.get("fold", 0)),
    )


def postura_de(cfg: AuditConfig, exercise: str) -> str:
    """Postura del ejercicio según el mapa del YAML; aborta si no está mapeado.

    Raises:
        ValueError: ejercicio fuera del mapa (un estrato inventado en silencio
            invalidaría el desglose de la métrica).
    """
    try:
        return cfg.postura[exercise]
    except KeyError as exc:
        raise ValueError(
            f"Ejercicio {exercise!r} sin postura asignada en estratos.postura."
        ) from exc


# ---------------------------------------------------------------------------
# Labels YOLO-pose
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseLabel:
    """Una persona de un label YOLO-pose, en coordenadas normalizadas.

    kpts: 17 tuplas ``(x, y, v)`` en orden COCO, ``v ∈ {0,1,2}``.
    """

    bbox: tuple[float, float, float, float]
    kpts: tuple[tuple[float, float, int], ...]


def parse_label_line(line: str) -> PoseLabel:
    """Parsea una línea de label YOLO-pose (56 tokens).

    Raises:
        ValueError: si el número de tokens o la visibilidad no cumplen el formato.
    """
    tokens = line.split()
    if len(tokens) != EXPECTED_TOKENS:
        raise ValueError(f"Línea con {len(tokens)} tokens (esperados {EXPECTED_TOKENS}): {line!r}")
    bbox = tuple(float(t) for t in tokens[1:5])
    kpts: list[tuple[float, float, int]] = []
    for i in range(NUM_COCO_KEYPOINTS):
        x, y, v = tokens[5 + 3 * i : 8 + 3 * i]
        vis = int(v)
        if vis not in (0, 1, 2):
            raise ValueError(f"Visibilidad inválida {v!r} en la línea: {line!r}")
        kpts.append((float(x), float(y), vis))
    return PoseLabel(bbox=bbox, kpts=tuple(kpts))  # type: ignore[arg-type]


def read_label_file(path: Path) -> list[PoseLabel]:
    """Lee un archivo de labels; vacío (frame sin detección) devuelve ``[]``."""
    labels: list[PoseLabel] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            labels.append(parse_label_line(line))
    return labels


def image_size(path: Path) -> tuple[int, int]:
    """``(ancho, alto)`` en píxeles de una imagen.

    Raises:
        ValueError: si la imagen no se puede leer.
    """
    import cv2  # noqa: PLC0415

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    h, w = img.shape[:2]
    return int(w), int(h)


# ---------------------------------------------------------------------------
# Matching de personas entre dos versiones del mismo frame
# ---------------------------------------------------------------------------


def _person_distance(a: PoseLabel, b: PoseLabel) -> float:
    """Distancia media (normalizada) entre keypoints visibles en ambas personas.

    Sin keypoints en común cae al centro del bbox: peor señal, pero suficiente
    para desambiguar personas espacialmente separadas.
    """
    comunes = [(ka, kb) for ka, kb in zip(a.kpts, b.kpts) if ka[2] > 0 and kb[2] > 0]
    if comunes:
        return sum(math.dist(ka[:2], kb[:2]) for ka, kb in comunes) / len(comunes)
    return math.dist(a.bbox[:2], b.bbox[:2])


def match_persons(
    originales: list[PoseLabel], corregidas: list[PoseLabel]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Empareja personas de dos versiones del frame, greedy por distancia.

    Returns:
        ``(pares, sin_par_original, sin_par_corregida)`` — índices en cada lista.
        Personas originales sin par = eliminadas por el anotador; corregidas
        sin par = agregadas (p.ej. un frame sin detección anotado de cero).
    """
    candidatos = sorted(
        (
            (_person_distance(a, b), i, j)
            for i, a in enumerate(originales)
            for j, b in enumerate(corregidas)
        ),
    )
    usados_a: set[int] = set()
    usados_b: set[int] = set()
    pares: list[tuple[int, int]] = []
    for _, i, j in candidatos:
        if i in usados_a or j in usados_b:
            continue
        pares.append((i, j))
        usados_a.add(i)
        usados_b.add(j)
    sin_par_a = [i for i in range(len(originales)) if i not in usados_a]
    sin_par_b = [j for j in range(len(corregidas)) if j not in usados_b]
    return pares, sin_par_a, sin_par_b


# ---------------------------------------------------------------------------
# Discrepancia pseudo-label vs GT de UCO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscrepanciaGT:
    """Desacuerdo entre el pseudo-label y el GT de UCO en un frame.

    UCO anota solo las 3 articulaciones del ejercicio (las que usa la app), así
    que la comparación cubre exactamente esas; las demás no tienen GT.

    err_px: distancia máxima (px) entre articulaciones visibles en ambos, o
        ``None`` si no aplica (frame sin GT, sin pseudo o sin articulaciones en
        común).
    kpt_perdido: alguna articulación visible en el GT quedó en ``v=0`` en el
        pseudo — ViTPose la descartó por score y el student no la aprende.
    """

    err_px: float | None
    kpt_perdido: bool

    def es_discrepante(self, thr_px: float) -> bool:
        """``True`` si el error supera el umbral o se perdió una articulación del GT."""
        return self.kpt_perdido or (self.err_px is not None and self.err_px > thr_px)


def discrepancia_gt(
    gt: list[PoseLabel], pseudo: list[PoseLabel], img_w: int, img_h: int
) -> DiscrepanciaGT:
    """Compara la primera persona del GT contra la primera del pseudo-label.

    UCO es single-subject (una línea de GT por frame) y el pseudo-labeling corre
    con ``max_persons: 1``, así que la primera persona de cada lado es la única.
    Sin persona en alguno de los dos lados la comparación no aplica: esos frames
    ya entran a la auditoría como ``sin_deteccion`` / ``sin_kpts_validos``.
    """
    if not gt or not pseudo:
        return DiscrepanciaGT(err_px=None, kpt_perdido=False)
    errores: list[float] = []
    perdido = False
    for kg, kp in zip(gt[0].kpts, pseudo[0].kpts):
        if kg[2] == 0:
            continue
        if kp[2] == 0:
            perdido = True
            continue
        errores.append(math.hypot((kg[0] - kp[0]) * img_w, (kg[1] - kp[1]) * img_h))
    return DiscrepanciaGT(err_px=max(errores) if errores else None, kpt_perdido=perdido)


# ---------------------------------------------------------------------------
# selection.csv
# ---------------------------------------------------------------------------

SELECTION_FIELDS = [
    "stem",
    "motivo",
    "postura",
    "subject",
    "exercise",
    "cam",
    "frame",
    "det_score",
    "mean_kpt_score",
    "n_personas_escritas",
    "n_v2",
    "n_v1",
    "n_v0",
    "err_gt_px",
    "kpt_perdido",
    "discrepancia_gt",
]


def read_selection(path: Path) -> list[dict[str, str]]:
    """Lee ``selection.csv`` producido por :mod:`src.audit.select_audit_subset`.

    Raises:
        FileNotFoundError: si aún no se corrió la selección.
        ValueError: si el CSV no tiene las columnas esperadas.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}; correr primero src.audit.select_audit_subset.")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != SELECTION_FIELDS:
            raise ValueError(
                f"Columnas inesperadas en {path}: {reader.fieldnames} != {SELECTION_FIELDS}"
            )
        return list(reader)
