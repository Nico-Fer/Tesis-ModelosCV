"""Auto-etiquetado (pseudo-labeling) con ViTPose-H como teacher → YOLO-pose.

Los labels de UCO son **estructuralmente pobres**: cada clip anota sólo los ~3
joints relevantes al ejercicio y **nunca la cara**, así que 14 de los 17 slots
COCO salen con ``v=0``. Este módulo genera un dataset paralelo con los 17
keypoints completos, corriendo un teacher fuerte sobre las **mismas imágenes**
ya extraídas por :mod:`src.data.uco_to_yolo`.

Pipeline **top-down** (dos etapas, como exige ViTPose):

1. **RTMDet-m** detecta personas y devuelve bounding boxes;
2. **ViTPose-H** estima 17 keypoints COCO sobre el recorte de cada persona;
3. se serializa una línea de label YOLO-pose por persona.

Salida (idéntica en forma a la de :mod:`src.data.uco_to_yolo`, para que los
runners de fase la consuman sin cambios)::

    <out>/images/all/<stem>.jpg     hardlinks al pool original (no se copia)
    <out>/labels/all/<stem>.txt     class cx cy w h  px py v ×17
    <out>/folds/fold{k}_{train,val}.txt
    <out>/manifest.csv              una fila por imagen, para revisión humana
    <out>/summary.json              contadores globales de la corrida
    <out>/run_metadata.json         versiones, seeds y hashes (reproducibilidad)
    pose_dataset_pseudo_fold{k}.yaml

Diferencias **deliberadas** respecto de :func:`src.data.uco_to_yolo.yolo_label_line`:

* el bbox sale del **detector**, no de la caja envolvente de los keypoints;
* las coordenadas se **clampean** a ``[0,1]`` (ViTPose puede extrapolar fuera
  del recorte y devolver coords negativas o mayores al ancho de la imagen).

El resto del formato es byte-compatible (clase literal ``0``, ``.6f`` en bbox y
keypoints, visibilidad entera), así que ambos datasets son intercambiables.

Uso (en ``venv-mmpose``, **no** en ``venv-torch``)::

    # 1. calibrar los umbrales con datos (obligatorio antes de la corrida larga)
    python -m src.data.pseudo_label_vitpose --config configs/pseudo_label_vitpose.yaml \\
        --dump-scores

    # 2. corrida completa (~2-4 h sobre 74k imágenes en una RTX 3090)
    python -m src.data.pseudo_label_vitpose --config configs/pseudo_label_vitpose.yaml

Los umbrales **no son simétricos** y conviene tenerlo presente al calibrar:

* ``det_thr`` es el más caro. Si el detector no dispara se pierde el **frame
  entero**, no un keypoint, y es irrecuperable sin rehacer la corrida.
* ``vis_thr_low`` es el único umbral que decide qué aprende el student: marca la
  frontera entre "el keypoint entra a la loss" y "se descarta como no anotado".
* ``vis_thr_high`` **no afecta el entrenamiento**: Ultralytics binariza la
  visibilidad (usa ``v>0`` como máscara), así que ``v=1`` y ``v=2`` entrenan
  idéntico. Sólo sirve para priorizar la revisión humana en el manifest.

Los tres se calibran con ``--dump-scores``, leyendo primero el detector: cambiar
``det_thr`` cambia la población de recortes y por lo tanto la distribución de
scores de ViTPose sobre la que se calibra ``vis_thr_low``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from src.data.uco_to_yolo import parse_stem, write_fold_manifests
from src.inference.keypoint_schema import COCO_KEYPOINT_NAMES, NUM_COCO_KEYPOINTS

logger = logging.getLogger(__name__)

# Se avisa una sola vez si el enlace duro no está disponible y se cae a copiar.
_LINK_FALLBACK_WARNED = False


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PseudoLabelConfig:
    """Configuración completa de una corrida, ya con defaults aplicados."""

    images_dir: Path
    pattern: str
    out_root: Path
    dataset_yaml: Path
    link_mode: str

    det_config: Path
    det_checkpoint: Path
    det_thr: float
    nms_thr: float
    person_cat_id: int
    max_persons: int | None

    pose_config: Path
    pose_checkpoint: Path
    vis_thr_low: float
    vis_thr_high: float

    device: str
    seed: int
    resume: bool
    flush_every: int

    n_folds: int
    fold_seed: int
    skip_empty: bool
    verify_against: Path | None

    sample_size: int
    stratify_by: tuple[str, ...]
    det_thr_candidates: tuple[float, ...]

    review_mean_score_thr: float
    review_min_visible_kpts: int


def load_config(path: Path) -> dict[str, Any]:
    """Lee el YAML de configuración.

    Args:
        path: ruta al YAML (ver ``configs/pseudo_label_vitpose.yaml``).

    Returns:
        El diccionario crudo, sin defaults aplicados.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    import yaml  # noqa: PLC0415

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return raw


def build_config(raw: dict[str, Any]) -> PseudoLabelConfig:
    """Aplica defaults y valida el YAML crudo.

    Args:
        raw: diccionario devuelto por :func:`load_config`.

    Returns:
        La configuración normalizada.

    Raises:
        ValueError: si un valor está fuera de rango o falta una clave obligatoria.
    """
    inp = raw.get("input") or {}
    out = raw.get("output") or {}
    det = raw.get("detector") or {}
    pose = raw.get("pose") or {}
    run = raw.get("runtime") or {}
    folds = raw.get("folds") or {}
    calib = raw.get("calibration") or {}
    review = raw.get("review") or {}

    for section, key, value in (
        ("detector", "config", det.get("config")),
        ("detector", "checkpoint", det.get("checkpoint")),
        ("pose", "config", pose.get("config")),
        ("pose", "checkpoint", pose.get("checkpoint")),
    ):
        if not value:
            raise ValueError(f"Falta la clave obligatoria {section}.{key} en la configuración.")

    vis_low = float(pose.get("vis_thr_low", 0.3))
    vis_high = float(pose.get("vis_thr_high", 0.6))
    if not 0.0 <= vis_low <= vis_high <= 1.0:
        raise ValueError(
            f"Se espera 0 ≤ vis_thr_low ≤ vis_thr_high ≤ 1 (recibido {vis_low} / {vis_high})."
        )

    link_mode = str(out.get("link_mode", "hardlink"))
    if link_mode not in ("hardlink", "copy", "symlink"):
        raise ValueError(f"link_mode inválido: {link_mode!r} (hardlink | copy | symlink).")

    max_persons = det.get("max_persons", 1)
    verify_against = folds.get("verify_against")

    return PseudoLabelConfig(
        images_dir=Path(inp.get("images_dir", "data/processed/uco_yolo/images/all")),
        pattern=str(inp.get("pattern", "*.jpg")),
        out_root=Path(out.get("root", "data/processed/uco_pseudo")),
        dataset_yaml=Path(out.get("dataset_yaml", "data/processed/pose_dataset_pseudo.yaml")),
        link_mode=link_mode,
        det_config=Path(det["config"]),
        det_checkpoint=Path(det["checkpoint"]),
        det_thr=float(det.get("det_thr", 0.5)),
        nms_thr=float(det.get("nms_thr", 0.3)),
        person_cat_id=int(det.get("person_cat_id", 0)),
        max_persons=None if max_persons in (None, 0) else int(max_persons),
        pose_config=Path(pose["config"]),
        pose_checkpoint=Path(pose["checkpoint"]),
        vis_thr_low=vis_low,
        vis_thr_high=vis_high,
        device=str(run.get("device", "cuda:0")),
        seed=int(run.get("seed", 42)),
        resume=bool(run.get("resume", True)),
        flush_every=int(run.get("flush_every", 500)),
        n_folds=int(folds.get("n_folds", 5)),
        fold_seed=int(folds.get("seed", 42)),
        skip_empty=bool(folds.get("skip_empty", True)),
        verify_against=Path(verify_against) if verify_against else None,
        sample_size=int(calib.get("sample_size", 500)),
        stratify_by=tuple(calib.get("stratify_by", ("exercise", "subject"))),
        det_thr_candidates=tuple(
            float(x) for x in calib.get("det_thr_candidates", (0.2, 0.3, 0.4, 0.5))
        ),
        review_mean_score_thr=float(review.get("mean_score_thr", 0.5)),
        review_min_visible_kpts=int(review.get("min_visible_kpts", 10)),
    )


# ---------------------------------------------------------------------------
# Serialización de labels
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    """Recorta un valor normalizado al rango ``[0, 1]``."""
    return min(1.0, max(0.0, value))


def derive_visibility(
    kpt_scores: Sequence[float], vis_thr_low: float, vis_thr_high: float
) -> list[int]:
    """Traduce los scores por keypoint de ViTPose a la visibilidad COCO.

    * ``score < vis_thr_low``            → ``0`` (no anotado; la coord se descarta)
    * ``vis_thr_low ≤ score < vis_thr_high`` → ``1`` (anotado, ocluido)
    * ``score ≥ vis_thr_high``           → ``2`` (visible)

    Recordar que Ultralytics **binariza** ``v>0``: ``1`` y ``2`` entrenan igual,
    así que sólo ``vis_thr_low`` cambia lo que aprende el student.

    Args:
        kpt_scores: score por keypoint, en orden COCO.
        vis_thr_low: umbral por debajo del cual el keypoint se descarta.
        vis_thr_high: umbral por encima del cual se marca visible.

    Returns:
        Lista de 17 enteros ``v ∈ {0,1,2}``.
    """
    out: list[int] = []
    for score in kpt_scores:
        if score < vis_thr_low:
            out.append(0)
        elif score < vis_thr_high:
            out.append(1)
        else:
            out.append(2)
    return out


def pseudo_label_line(
    bbox_xyxy: Sequence[float],
    kpts_px: Sequence[Sequence[float]],
    visibility: Sequence[int],
    img_w: int,
    img_h: int,
) -> str:
    """Serializa una persona a una línea de label YOLO-pose.

    A diferencia de :func:`src.data.uco_to_yolo.yolo_label_line`, el bbox viene
    del **detector** (no de la envolvente de keypoints) y todas las coordenadas
    se **clampean** a ``[0,1]``.

    Args:
        bbox_xyxy: ``(x0, y0, x1, y1)`` en píxeles, del detector.
        kpts_px: 17 pares ``(x, y)`` en píxeles.
        visibility: 17 enteros ``v`` (ver :func:`derive_visibility`).
        img_w: ancho de la imagen en píxeles.
        img_h: alto de la imagen en píxeles.

    Returns:
        Línea ``"0 cx cy w h px1 py1 v1 …"`` con 56 tokens.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy)
    cx = _clamp01(((x0 + x1) / 2) / img_w)
    cy = _clamp01(((y0 + y1) / 2) / img_h)
    bw = _clamp01((x1 - x0) / img_w)
    bh = _clamp01((y1 - y0) / img_h)

    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for (x, y), v in zip(kpts_px, visibility):
        if v == 0:
            # No anotado: se descarta la coordenada, igual que hace el converter.
            parts += ["0.000000", "0.000000", "0"]
        else:
            parts += [f"{_clamp01(x / img_w):.6f}", f"{_clamp01(y / img_h):.6f}", str(v)]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Backends (mmdet + mmpose) — importados en diferido
# ---------------------------------------------------------------------------


def _import_backends() -> SimpleNamespace:
    """Importa la API de mmdet/mmpose con un error claro si falta el entorno.

    Se difiere el import para que el módulo se pueda importar (y testear el
    formateo de labels) sin OpenMMLab instalado.

    Raises:
        RuntimeError: si el entorno ``venv-mmpose`` no está armado.
    """
    try:
        from mmdet.apis import inference_detector, init_detector  # noqa: PLC0415
        from mmengine.registry import init_default_scope  # noqa: PLC0415
        from mmpose.apis import inference_topdown, init_model  # noqa: PLC0415
        from mmpose.evaluation.functional import nms  # noqa: PLC0415
        from mmpose.structures import merge_data_samples  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError(
            "No se pudo importar mmdet/mmpose. Este script corre en venv-mmpose: "
            "crealo con `make setup-mmpose` (ver requirements-mmpose.txt)."
        ) from exc
    return SimpleNamespace(
        inference_detector=inference_detector,
        init_detector=init_detector,
        init_default_scope=init_default_scope,
        inference_topdown=inference_topdown,
        init_model=init_model,
        nms=nms,
        merge_data_samples=merge_data_samples,
    )


def _seed_everything(seed: int) -> None:
    """Fija las semillas de random/numpy/torch y desactiva el no-determinismo de cuDNN."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_persons(
    backends: SimpleNamespace,
    det_model: Any,
    img: Any,
    cfg: PseudoLabelConfig,
    *,
    det_thr: float | None = None,
) -> Any:
    """Corre el detector y devuelve las cajas de persona, ordenadas por score.

    Args:
        backends: namespace de :func:`_import_backends`.
        det_model: detector ya inicializado.
        img: imagen BGR (ndarray).
        cfg: configuración de la corrida.
        det_thr: umbral a aplicar; si es ``None`` usa ``cfg.det_thr``. El modo
            ``--dump-scores`` pasa ``0.0`` para poder perfilar la cola baja.

    Returns:
        Array ``(N, 5)`` con ``[x0, y0, x1, y1, score]``, score descendente.
    """
    import numpy as np  # noqa: PLC0415

    thr = cfg.det_thr if det_thr is None else det_thr
    backends.init_default_scope(det_model.cfg.get("default_scope", "mmdet"))
    result = backends.inference_detector(det_model, img)
    pred = result.pred_instances.cpu().numpy()

    keep = np.logical_and(pred.labels == cfg.person_cat_id, pred.scores > thr)
    bboxes = np.concatenate((pred.bboxes, pred.scores[:, None]), axis=1)[keep]
    if len(bboxes) == 0:
        return bboxes
    bboxes = bboxes[backends.nms(bboxes, cfg.nms_thr)]
    # Descartar cajas degeneradas (área ~0) antes de recortar.
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    degenerate = areas <= 1.0
    if degenerate.any():
        logger.warning("Se descartaron %d cajas de área ~0.", int(degenerate.sum()))
        bboxes = bboxes[~degenerate]
    order = np.argsort(-bboxes[:, 4])
    return bboxes[order]


def estimate_pose(backends: SimpleNamespace, pose_model: Any, img: Any, bboxes: Any) -> Any:
    """Corre ViTPose sobre los recortes indicados.

    Args:
        backends: namespace de :func:`_import_backends`.
        pose_model: estimador ya inicializado.
        img: imagen BGR (ndarray).
        bboxes: array ``(P, 4+)``; sólo se usan las 4 primeras columnas (xyxy).

    Returns:
        Tupla ``(keypoints (P,17,2) px, scores (P,17))``.
    """
    backends.init_default_scope(pose_model.cfg.get("default_scope", "mmpose"))
    results = backends.inference_topdown(pose_model, img, bboxes[:, :4], bbox_format="xyxy")
    merged = backends.merge_data_samples(results)
    pred = merged.pred_instances
    return pred.keypoints, pred.keypoint_scores


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class ImageRecord:
    """Una fila del ``manifest.csv``: el resultado del teacher sobre una imagen."""

    image: str
    subject: str
    exercise: str
    cam: str
    frame: int
    n_personas_detectadas: int
    n_personas_escritas: int
    det_score: float
    mean_kpt_score: float
    n_v2: int
    n_v1: int
    n_v0: int
    baja_confianza: int


MANIFEST_FIELDS: list[str] = [f.name for f in fields(ImageRecord)]


def write_manifest(path: Path, records: Sequence[ImageRecord]) -> None:
    """Vuelca el manifest completo a CSV (se reescribe entero en cada flush)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(path: Path) -> dict[str, ImageRecord]:
    """Relee un ``manifest.csv`` parcial para poder reanudar una corrida.

    Returns:
        Mapa ``image -> ImageRecord``; vacío si el archivo no existe o está roto.
    """
    if not path.exists():
        return {}
    out: dict[str, ImageRecord] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                out[row["image"]] = ImageRecord(
                    image=row["image"],
                    subject=row["subject"],
                    exercise=row["exercise"],
                    cam=row["cam"],
                    frame=int(row["frame"]),
                    n_personas_detectadas=int(row["n_personas_detectadas"]),
                    n_personas_escritas=int(row["n_personas_escritas"]),
                    det_score=float(row["det_score"]),
                    mean_kpt_score=float(row["mean_kpt_score"]),
                    n_v2=int(row["n_v2"]),
                    n_v1=int(row["n_v1"]),
                    n_v0=int(row["n_v0"]),
                    baja_confianza=int(row["baja_confianza"]),
                )
    except (KeyError, ValueError) as exc:
        logger.warning("Manifest previo ilegible (%s); se reprocesa todo.", exc)
        return {}
    return out


# ---------------------------------------------------------------------------
# Enlazado de imágenes
# ---------------------------------------------------------------------------


def link_image(src: Path, dst: Path, mode: str) -> None:
    """Materializa la imagen en el pool de salida sin moverla del origen.

    ``hardlink`` es el default: en NTFS sobre el mismo volumen no ocupa espacio
    extra ni requiere permisos de administrador (a diferencia de ``symlink``).
    Si el enlace falla —típicamente por cruzar de volumen— se cae a copiar y se
    avisa **una sola vez**.
    """
    global _LINK_FALLBACK_WARNED

    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        if mode == "symlink":
            os.symlink(src, dst)
        else:
            os.link(src, dst)
    except OSError as exc:
        if not _LINK_FALLBACK_WARNED:
            logger.warning("No se pudo enlazar (%s); se copian las imágenes.", exc)
            _LINK_FALLBACK_WARNED = True
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Procesamiento de una imagen
# ---------------------------------------------------------------------------


def process_image(
    img_path: Path,
    backends: SimpleNamespace,
    det_model: Any,
    pose_model: Any,
    cfg: PseudoLabelConfig,
) -> tuple[ImageRecord, list[str]] | None:
    """Etiqueta una imagen: detecta personas, estima pose y arma las líneas.

    Args:
        img_path: ruta de la imagen de entrada.
        backends: namespace de :func:`_import_backends`.
        det_model: detector inicializado.
        pose_model: estimador de pose inicializado.
        cfg: configuración de la corrida.

    Returns:
        ``(registro, líneas)`` — ``líneas`` vacío si no hubo persona válida — o
        ``None`` si la imagen no se pudo leer.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning("Imagen ilegible, se saltea: %s", img_path)
        return None

    height, width = img.shape[:2]
    parts = parse_stem(img_path.stem)
    bboxes = detect_persons(backends, det_model, img, cfg)
    n_detectadas = int(len(bboxes))

    lines: list[str] = []
    n_v2 = n_v1 = n_v0 = 0
    det_score = 0.0
    mean_kpt_score = 0.0

    if n_detectadas:
        det_score = float(bboxes[0, 4])
        selected = bboxes if cfg.max_persons is None else bboxes[: cfg.max_persons]
        keypoints, kpt_scores = estimate_pose(backends, pose_model, img, selected)
        per_person_means: list[float] = []
        for person_idx in range(len(selected)):
            scores = kpt_scores[person_idx]
            visibility = derive_visibility(scores, cfg.vis_thr_low, cfg.vis_thr_high)
            if all(v == 0 for v in visibility):
                # Caja sin ningún keypoint utilizable: una línea así no aporta
                # señal de pose y la kpt-loss no puede usarla. Se descarta.
                continue
            lines.append(
                pseudo_label_line(
                    selected[person_idx, :4], keypoints[person_idx], visibility, width, height
                )
            )
            n_v2 += sum(1 for v in visibility if v == 2)
            n_v1 += sum(1 for v in visibility if v == 1)
            n_v0 += sum(1 for v in visibility if v == 0)
            per_person_means.append(float(np.mean(scores)))
        if per_person_means:
            mean_kpt_score = float(np.mean(per_person_means))

    baja_confianza = (
        n_detectadas == 0
        or n_detectadas > 1
        or not lines
        or mean_kpt_score < cfg.review_mean_score_thr
        or n_v2 < cfg.review_min_visible_kpts
    )
    record = ImageRecord(
        image=img_path.name,
        subject=parts.subject,
        exercise=parts.exercise,
        cam=parts.cam,
        frame=parts.frame,
        n_personas_detectadas=n_detectadas,
        n_personas_escritas=len(lines),
        det_score=round(det_score, 6),
        mean_kpt_score=round(mean_kpt_score, 6),
        n_v2=n_v2,
        n_v1=n_v1,
        n_v0=n_v0,
        baja_confianza=int(baja_confianza),
    )
    return record, lines


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def _subjects_by_fold_from_manifests(folds_dir: Path, n_folds: int) -> dict[int, set[str]]:
    """Lee ``folds/fold{k}_val.txt`` y devuelve el set de sujetos de cada fold."""
    out: dict[int, set[str]] = {}
    for k in range(n_folds):
        manifest = folds_dir / f"fold{k}_val.txt"
        if not manifest.exists():
            continue
        subjects: set[str] = set()
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                subjects.add(parse_stem(Path(line.strip()).stem).subject)
        out[k] = subjects
    return out


def verify_folds_against(pseudo_root: Path, gt_root: Path, n_folds: int) -> None:
    """Comprueba que el reparto de sujetos por fold coincida con el del ground truth.

    Sin esta verificación, "mismo seed ⇒ mismo reparto" es una suposición. Y si
    los repartos difieren, comparar el fold *k* del dataset pseudo contra el fold
    *k* del GT deja de ser válido — un error silencioso que invalidaría toda la
    comparación experimental.

    Raises:
        RuntimeError: si algún fold tiene un set de sujetos distinto.
    """
    gt_folds = _subjects_by_fold_from_manifests(gt_root / "folds", n_folds)
    if not gt_folds:
        logger.warning(
            "No hay manifests de fold en %s; se omite la verificación contra el GT.",
            gt_root / "folds",
        )
        return
    pseudo_folds = _subjects_by_fold_from_manifests(pseudo_root / "folds", n_folds)
    diffs: list[str] = []
    for k, gt_subjects in gt_folds.items():
        mine = pseudo_folds.get(k, set())
        if mine != gt_subjects:
            diffs.append(
                f"  fold{k}: GT={sorted(gt_subjects, key=str)} pseudo={sorted(mine, key=str)}"
            )
    if diffs:
        raise RuntimeError(
            "El reparto de sujetos por fold NO coincide con el dataset GT; la comparación "
            "1:1 por fold sería inválida:\n" + "\n".join(diffs)
        )
    logger.info("Reparto de folds verificado contra %s: idéntico.", gt_root)


# ---------------------------------------------------------------------------
# Modo calibración (--dump-scores)
# ---------------------------------------------------------------------------


def stratified_sample(stems: Sequence[str], keys: Sequence[str], size: int, seed: int) -> list[str]:
    """Muestrea ``size`` stems repartiendo round-robin entre los grupos de ``keys``.

    Args:
        stems: nombres base disponibles.
        keys: campos de :class:`~src.data.uco_to_yolo.StemParts` por los que
            estratificar (p.ej. ``("exercise", "subject")``).
        size: tamaño objetivo de la muestra.
        seed: semilla del shuffle intra-grupo.

    Returns:
        Lista ordenada de stems (determinista para un mismo ``seed``).
    """
    groups: dict[tuple[str, ...], list[str]] = {}
    for stem in stems:
        parts = parse_stem(stem)
        key = tuple(str(getattr(parts, k)) for k in keys)
        groups.setdefault(key, []).append(stem)

    rng = random.Random(seed)
    for bucket in groups.values():
        bucket.sort()
        rng.shuffle(bucket)

    ordered_keys = sorted(groups)
    picked: list[str] = []
    idx = 0
    while len(picked) < size and any(groups[k] for k in ordered_keys):
        bucket = groups[ordered_keys[idx % len(ordered_keys)]]
        if bucket:
            picked.append(bucket.pop())
        idx += 1
    return sorted(picked)


def _percentiles(values: Sequence[float]) -> dict[str, Any]:
    """Percentiles p5/p25/p50/p75/p95 + media de una muestra."""
    import numpy as np  # noqa: PLC0415

    if not len(values):
        return {"n": 0, "p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=float)
    p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    return {
        "n": int(arr.size),
        "p5": round(float(p5), 4),
        "p25": round(float(p25), 4),
        "p50": round(float(p50), 4),
        "p75": round(float(p75), 4),
        "p95": round(float(p95), 4),
        "mean": round(float(arr.mean()), 4),
    }


def run_dump_scores(cfg: PseudoLabelConfig) -> None:
    """Perfila los scores del detector y de ViTPose sin escribir labels.

    Escribe en ``<out>/calibration/``:

    * ``score_stats.csv`` — percentiles de ``det_score`` y de cada keypoint,
      global y por ejercicio (formato largo, listo para pandas);
    * ``det_thr_sweep.csv`` — cuántas imágenes quedarían sin detección con cada
      umbral candidato. **Esto es lo primero que hay que mirar**: cada frame
      perdido acá es irrecuperable sin rehacer la corrida completa;
    * ``score_samples.csv`` — los scores crudos por imagen y keypoint, para
      poder re-analizar sin volver a gastar GPU.
    """
    backends = _import_backends()
    _seed_everything(cfg.seed)

    stems = sorted(p.stem for p in cfg.images_dir.glob(cfg.pattern))
    if not stems:
        raise RuntimeError(f"No se encontraron imágenes en {cfg.images_dir} ({cfg.pattern}).")
    sample = stratified_sample(stems, cfg.stratify_by, cfg.sample_size, cfg.seed)
    logger.info(
        "Calibración: %d imágenes muestreadas de %d, estratificadas por %s.",
        len(sample),
        len(stems),
        ", ".join(cfg.stratify_by),
    )

    det_model, pose_model = _init_models(backends, cfg)
    out_dir = cfg.out_root / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    import cv2  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    for stem in tqdm(sample, desc="calibrando", unit="img"):
        img_path = cfg.images_dir / f"{stem}{_suffix(cfg.pattern)}"
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Imagen ilegible, se saltea: %s", img_path)
            continue
        parts = parse_stem(stem)
        # Umbral 0 a propósito: para dimensionar la cola baja hace falta ver las
        # detecciones que `det_thr` descartaría.
        bboxes = detect_persons(backends, det_model, img, cfg, det_thr=0.0)
        if len(bboxes) == 0:
            rows.append(
                {
                    "stem": stem,
                    "exercise": parts.exercise,
                    "subject": parts.subject,
                    "det_score": 0.0,
                    "kpt_scores": [0.0] * NUM_COCO_KEYPOINTS,
                }
            )
            continue
        _, kpt_scores = estimate_pose(backends, pose_model, img, bboxes[:1])
        rows.append(
            {
                "stem": stem,
                "exercise": parts.exercise,
                "subject": parts.subject,
                "det_score": float(bboxes[0, 4]),
                "kpt_scores": [float(s) for s in kpt_scores[0]],
            }
        )

    _write_calibration_outputs(out_dir, rows, cfg)
    print(f"Calibración completa. Resultados en {out_dir}")
    print(
        "Leer PRIMERO det_thr_sweep.csv (elegir det_thr), DESPUÉS score_stats.csv "
        "(elegir vis_thr_low): cambiar det_thr cambia la población de recortes."
    )


def _write_calibration_outputs(
    out_dir: Path, rows: list[dict[str, Any]], cfg: PseudoLabelConfig
) -> None:
    """Escribe los tres CSV del modo calibración."""
    # 1. barrido de det_thr: cuántos frames se perderían con cada candidato.
    with (out_dir / "det_thr_sweep.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["det_thr", "imagenes", "sin_deteccion", "pct_sin_deteccion"])
        for thr in cfg.det_thr_candidates:
            missing = sum(1 for r in rows if r["det_score"] < thr)
            pct = round(100.0 * missing / len(rows), 2) if rows else 0.0
            writer.writerow([thr, len(rows), missing, pct])

    # 2. percentiles en formato largo, global y por ejercicio.
    exercises = sorted({r["exercise"] for r in rows})
    with (out_dir / "score_stats.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scope", "group", "metric", "n", "p5", "p25", "p50", "p75", "p95", "mean"])

        def emit(scope: str, group: str, metric: str, values: list[float]) -> None:
            stats = _percentiles(values)
            writer.writerow(
                [
                    scope,
                    group,
                    metric,
                    stats["n"],
                    stats["p5"],
                    stats["p25"],
                    stats["p50"],
                    stats["p75"],
                    stats["p95"],
                    stats["mean"],
                ]
            )

        for group, subset in [("__all__", rows)] + [
            (ex, [r for r in rows if r["exercise"] == ex]) for ex in exercises
        ]:
            emit("detector", group, "det_score", [r["det_score"] for r in subset])
            for idx, name in enumerate(COCO_KEYPOINT_NAMES):
                emit("keypoint", group, name, [r["kpt_scores"][idx] for r in subset])

    # 3. scores crudos, para re-analizar sin volver a usar la GPU.
    with (out_dir / "score_samples.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stem", "subject", "exercise", "det_score", *COCO_KEYPOINT_NAMES])
        for r in rows:
            writer.writerow(
                [
                    r["stem"],
                    r["subject"],
                    r["exercise"],
                    round(r["det_score"], 6),
                    *[round(s, 6) for s in r["kpt_scores"]],
                ]
            )


def _suffix(pattern: str) -> str:
    """Extensión implicada por el glob de entrada (``"*.jpg"`` → ``".jpg"``)."""
    return Path(pattern).suffix or ".jpg"


# ---------------------------------------------------------------------------
# Corrida principal
# ---------------------------------------------------------------------------


def _init_models(backends: SimpleNamespace, cfg: PseudoLabelConfig) -> tuple[Any, Any]:
    """Inicializa detector y estimador de pose, validando que los archivos existan."""
    for label, path in (
        ("detector.config", cfg.det_config),
        ("detector.checkpoint", cfg.det_checkpoint),
        ("pose.config", cfg.pose_config),
        ("pose.checkpoint", cfg.pose_checkpoint),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"No existe {label}: {path}. Descargá los checkpoints con `make teachers` "
                "y copiá los nombres reales de models/teachers/ al YAML."
            )
    logger.info("Cargando detector desde %s", cfg.det_checkpoint)
    det_model = backends.init_detector(
        str(cfg.det_config), str(cfg.det_checkpoint), device=cfg.device
    )
    logger.info("Cargando ViTPose desde %s", cfg.pose_checkpoint)
    pose_model = backends.init_model(
        str(cfg.pose_config), str(cfg.pose_checkpoint), device=cfg.device
    )
    return det_model, pose_model


def _file_md5(path: Path) -> str:
    """MD5 de un archivo, leído por bloques (los checkpoints pesan GB)."""
    digest = hashlib.md5()  # noqa: S324 - identificación de artefacto, no seguridad
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_metadata(cfg: PseudoLabelConfig, raw_config: dict[str, Any], elapsed: float) -> dict:
    """Arma el snapshot de reproducibilidad de la corrida."""
    import cv2  # noqa: PLC0415
    import numpy  # noqa: PLC0415
    import torch  # noqa: PLC0415

    versions = {"torch": torch.__version__, "numpy": numpy.__version__, "cv2": cv2.__version__}
    for name in ("mmcv", "mmdet", "mmpose", "mmengine"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # pragma: no cover - informativo
            versions[name] = "no disponible"

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "device": cfg.device,
        "cuda_disponible": torch.cuda.is_available(),
        "seed": cfg.seed,
        "duracion_segundos": round(elapsed, 1),
        "versiones": versions,
        "checkpoints": {
            "detector": {"path": str(cfg.det_checkpoint), "md5": _file_md5(cfg.det_checkpoint)},
            "pose": {"path": str(cfg.pose_checkpoint), "md5": _file_md5(cfg.pose_checkpoint)},
        },
        "config": raw_config,
    }


@dataclass
class RunCounters:
    """Contadores globales de la corrida, volcados a ``summary.json``."""

    procesadas: int = 0
    reanudadas: int = 0
    ilegibles: int = 0
    sin_deteccion: int = 0
    sin_keypoints_validos: int = 0
    multi_persona: int = 0
    baja_confianza: int = 0
    personas_escritas: int = 0
    keypoints_por_imagen: dict[str, int] = field(default_factory=dict)


def run(
    cfg: PseudoLabelConfig, raw_config: dict[str, Any], limit: int | None = None
) -> Path | None:
    """Ejecuta el auto-etiquetado completo y genera el dataset pseudo-etiquetado.

    Args:
        cfg: configuración normalizada.
        raw_config: YAML crudo, para el snapshot de reproducibilidad.
        limit: si se indica, procesa sólo las primeras N imágenes (smoke test) y
            **se saltea la generación de folds**: un subconjunto arbitrario no
            cubre los ``n_folds`` sujetos que exige el GroupKFold.

    Returns:
        La ruta del primer dataset YAML de fold, o ``None`` en modo smoke.
    """
    import numpy as np  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    started = time.time()
    backends = _import_backends()
    _seed_everything(cfg.seed)

    image_paths = sorted(cfg.images_dir.glob(cfg.pattern))
    if not image_paths:
        raise RuntimeError(f"No se encontraron imágenes en {cfg.images_dir} ({cfg.pattern}).")
    if limit is not None:
        image_paths = image_paths[:limit]
    logger.info("Auto-etiquetado sobre %d imágenes de %s.", len(image_paths), cfg.images_dir)

    images_out = cfg.out_root / "images" / "all"
    labels_out = cfg.out_root / "labels" / "all"
    for directory in (images_out, labels_out, cfg.out_root / "folds"):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = cfg.out_root / "manifest.csv"
    previous = read_manifest(manifest_path) if cfg.resume else {}
    if previous:
        logger.info("Reanudando: %d imágenes ya registradas en el manifest.", len(previous))

    det_model, pose_model = _init_models(backends, cfg)

    counters = RunCounters()
    records: list[ImageRecord] = []
    stems_by_subject: dict[str, list[str]] = {}

    for processed, img_path in enumerate(tqdm(image_paths, desc="etiquetando", unit="img"), 1):
        stem = img_path.stem
        label_path = labels_out / f"{stem}.txt"
        record = previous.get(img_path.name) if cfg.resume else None

        if record is not None and label_path.exists():
            counters.reanudadas += 1
            lines_count = record.n_personas_escritas
        else:
            outcome = process_image(img_path, backends, det_model, pose_model, cfg)
            if outcome is None:
                counters.ilegibles += 1
                continue
            record, lines = outcome
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            lines_count = len(lines)

        link_image(img_path, images_out / img_path.name, cfg.link_mode)
        records.append(record)

        counters.procesadas += 1
        counters.personas_escritas += lines_count
        if record.n_personas_detectadas == 0:
            counters.sin_deteccion += 1
        elif lines_count == 0:
            counters.sin_keypoints_validos += 1
        if record.n_personas_detectadas > 1:
            counters.multi_persona += 1
        if record.baja_confianza:
            counters.baja_confianza += 1
        bucket = str(record.n_v2 + record.n_v1)
        counters.keypoints_por_imagen[bucket] = counters.keypoints_por_imagen.get(bucket, 0) + 1

        if lines_count or not cfg.skip_empty:
            stems_by_subject.setdefault(record.subject, []).append(stem)

        if processed % cfg.flush_every == 0:
            write_manifest(manifest_path, records)

    write_manifest(manifest_path, records)

    if not stems_by_subject and limit is None:
        raise RuntimeError(
            "Ninguna imagen produjo labels: revisá detector.det_thr y pose.vis_thr_low "
            "(corré primero --dump-scores)."
        )

    yaml_paths: list[Path] = []
    if limit is not None:
        # Smoke: --limit toma las primeras N imágenes del glob ordenado, que
        # caen todas en los primeros sujetos. Armar folds con eso fallaría (o,
        # peor, produciría un split no comparable con el del GT).
        logger.warning(
            "Modo smoke (--limit %d): se saltea la generación de folds. "
            "Los labels y el manifest sí se escribieron en %s.",
            limit,
            cfg.out_root,
        )
    else:
        yaml_paths = write_fold_manifests(
            cfg.out_root,
            cfg.dataset_yaml,
            stems_by_subject,
            n_folds=cfg.n_folds,
            seed=cfg.fold_seed,
        )
        if cfg.verify_against is not None and cfg.verify_against.exists():
            verify_folds_against(cfg.out_root, cfg.verify_against, cfg.n_folds)

    elapsed = time.time() - started
    mean_scores = [r.mean_kpt_score for r in records if r.mean_kpt_score > 0]
    summary = {
        **asdict(counters),
        "imagenes_totales": len(image_paths),
        "mean_kpt_score_global": round(float(np.mean(mean_scores)), 4) if mean_scores else 0.0,
        "sujetos": len(stems_by_subject),
        "duracion_segundos": round(elapsed, 1),
    }
    (cfg.out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (cfg.out_root / "run_metadata.json").write_text(
        json.dumps(_run_metadata(cfg, raw_config, elapsed), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "Listo: %d imágenes, %d personas, %d sin detección, %d sin keypoints válidos, "
        "%d marcadas para revisión.",
        counters.procesadas,
        counters.personas_escritas,
        counters.sin_deteccion,
        counters.sin_keypoints_validos,
        counters.baja_confianza,
    )
    return yaml_paths[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-etiquetado con ViTPose-H (teacher) → dataset YOLO-pose."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pseudo_label_vitpose.yaml"),
        help="YAML de configuración (ver configs/pseudo_label_vitpose.yaml).",
    )
    parser.add_argument(
        "--dump-scores",
        action="store_true",
        help="Sólo perfila scores de detector y pose para calibrar umbrales; no escribe labels.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Procesar sólo N imágenes (smoke).")
    parser.add_argument("--device", default=None, help="Override de runtime.device (p.ej. cpu).")
    parser.add_argument("--out-root", type=Path, default=None, help="Override de output.root.")
    parser.add_argument(
        "--no-resume", action="store_true", help="Reprocesa todo aunque existan labels previos."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw = load_config(args.config)

    # Overrides de CLI: sólo para iterar rápido; lo estable vive en el YAML.
    if args.device:
        raw.setdefault("runtime", {})["device"] = args.device
    if args.out_root:
        raw.setdefault("output", {})["root"] = str(args.out_root)
    if args.no_resume:
        raw.setdefault("runtime", {})["resume"] = False

    cfg = build_config(raw)
    if args.dump_scores:
        run_dump_scores(cfg)
        return

    dataset_yaml = run(cfg, raw, limit=args.limit)
    if dataset_yaml is None:
        print(f"Smoke completado (sin folds). Salida en {cfg.out_root}")
    else:
        print(f"Dataset pseudo-etiquetado generado. Primer fold: {dataset_yaml}")
    print(f"Revisar {cfg.out_root / 'summary.json'} antes de entrenar.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
