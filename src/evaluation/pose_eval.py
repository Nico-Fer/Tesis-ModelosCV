"""Evaluador unificado de keypoints para las 4 etapas de la Fase 5.

Calcula OKS y PCK **de la misma forma** para modelos ``.pt`` (base / podado /
distilado) y ``.tflite`` (INT8), de modo que las métricas de cada etapa sean
directamente comparables. La inferencia se corre siempre con Ultralytics (carga
tanto ``.pt`` como ``.tflite``), se extraen los keypoints predichos y se puntúan
contra el ground-truth de UCO con :mod:`src.evaluation.metrics`.

Decisiones de normalización (ver ``docs/requerimientos-de-datos-por-tecnica.md``):

* El bbox de UCO es del tamaño de un miembro (se calcula sólo sobre los 3
  keypoints anotados), así que **no** sirve para normalizar el área de OKS ni el
  ``reference_size`` de PCK. En su lugar se deriva un **bbox de persona completa**
  con el detector pre-entrenado base (yolov8n-pose) **una sola vez** por frame y
  se **reutiliza idéntico en las 4 etapas** → OKS/PCK quedan comparables entre
  etapas (misma normalización).
* Métrica primaria: **PCK@0.2** con ``reference_size`` = diagonal del bbox de
  persona. OKS se reporta además, normalizando por el área de ese mismo bbox.
* Se puntúan los keypoints con ``v > 0``: :mod:`src.evaluation.metrics` enmascara
  los ``v=0`` automáticamente. Sobre el dataset pseudo eso incluye los ``v=1``
  (ocluidos según ViTPose), que es el comportamiento correcto — Ultralytics
  también binariza la visibilidad, así que entrenamiento y evaluación coinciden.

Entorno requerido: **venv-torch** (Ultralytics). Sólo ``cv2``/``numpy`` fuera de
la inferencia, así que el parseo de labels es testeable sin framework.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.uco_to_yolo import parse_stem
from src.evaluation import metrics as metric_utils
from src.inference.keypoint_schema import NUM_COCO_KEYPOINTS
from src.utils import strict

logger = logging.getLogger(__name__)

# Ejercicios UCO por grupo muscular: 01–04 pierna izquierda, 05–08 pierna derecha,
# 09 brazo izquierdo, 13 brazo derecho. El grupo se deriva del **stem** y no de qué
# keypoints están visibles: esa heurística sólo funcionaba con las labels esparsas
# de UCO (3 kpts por clip). Con el dataset pseudo los 17 keypoints están anotados,
# así que mirar la visibilidad clasificaría todo como "pierna" y dejaría `pck_arm`
# permanentemente vacío.
_LEG_EXERCISES = frozenset({"01", "02", "03", "04", "05", "06", "07", "08"})
_ARM_EXERCISES = frozenset({"09", "13"})

# Fracción máxima de imágenes del split val que puede quedar sin label legible
# antes de considerar el dataset roto (en vez de saltearlas en silencio).
_MAX_MISSING_LABEL_RATIO = 0.01


@dataclass(slots=True)
class StageScore:
    """Puntajes de una etapa sobre el set de validación de un fold.

    Attributes:
        oks: Object Keypoint Similarity promedio (``None`` si no hay frames).
        pck: PCK@0.2 promedio.
        pck_leg: PCK@0.2 restringido a los clips de pierna.
        pck_arm: PCK@0.2 restringido a los clips de brazo.
        n_frames: cantidad de frames efectivamente puntuados.
    """

    oks: float | None = None
    pck: float | None = None
    pck_leg: float | None = None
    pck_arm: float | None = None
    n_frames: int = 0


@dataclass(slots=True)
class EvalContext:
    """Contexto compartido por las 4 etapas de un fold.

    Contiene el ground-truth y el bbox de persona (normalizador fijo) por frame,
    derivados una sola vez del stage base y reutilizados en todas las etapas.

    Attributes:
        gt_kpts: ``{path: (17, 3)}`` keypoints GT en píxeles (x, y, v).
        person_boxes: ``{path: (x, y, w, h)}`` bbox de persona en píxeles.
        dims: ``{path: (w, h)}`` tamaño de cada imagen en píxeles.
        img_paths: rutas de las imágenes con bbox de persona derivado.
        base_preds: predicciones del detector base por frame (formato de
            :func:`run_predictions`), reutilizables para puntuar la etapa base sin
            re-correr la inferencia.
    """

    gt_kpts: dict[Path, np.ndarray]
    person_boxes: dict[Path, np.ndarray]
    dims: dict[Path, tuple[int, int]]
    img_paths: list[Path]
    base_preds: dict[Path, dict[str, Any]]


def read_val_image_paths(data_yaml: str | Path) -> list[Path]:
    """Devuelve las rutas de imágenes del split ``val`` de un dataset YAML de fold.

    El ``val`` de ``pose_dataset_fold{k}.yaml`` es un ``.txt`` de rutas absolutas
    (una por línea). Se resuelve relativo a ``path`` si hiciera falta.

    Args:
        data_yaml: ruta al YAML de dataset del fold.

    Returns:
        Lista de rutas de imágenes existentes; ``[]`` sólo en smoke test (fuera de
        él, cualquier eslabón irresoluble aborta la fase en vez de devolver una
        lista vacía que degradaría la evaluación a silencio).
    """
    import yaml  # noqa: PLC0415

    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        strict.fail(f"No existe el dataset YAML del fold: {data_yaml}.")
        return []
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(spec.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    val = spec.get("val")
    if not val:
        strict.fail(f"{data_yaml} no declara un split 'val'; no hay sobre qué evaluar.")
        return []
    val_path = Path(val)
    if not val_path.is_absolute():
        val_path = root / val
    if not val_path.exists():
        strict.fail(f"No existe el manifest del split val: {val_path}.")
        return []

    lines = [ln.strip() for ln in val_path.read_text(encoding="utf-8").splitlines()]
    paths = [Path(ln) for ln in lines if ln]
    existing = [p for p in paths if p.exists()]
    if paths and len(existing) < len(paths) * (1.0 - _MAX_MISSING_LABEL_RATIO):
        strict.fail(
            f"{len(paths) - len(existing)}/{len(paths)} imágenes del split val de "
            f"{val_path.name} no existen en disco: el dataset está incompleto o los "
            "manifests apuntan a otra ruta."
        )
    return existing


def _label_path_for(image_path: Path) -> Path:
    """Ruta del label YOLO de una imagen (``images/all`` → ``labels/all``, ``.txt``)."""
    parts = [("labels" if part == "images" else part) for part in image_path.parts]
    return Path(*parts).with_suffix(".txt")


def load_gt_normalized(image_paths: list[Path]) -> dict[Path, np.ndarray]:
    """Carga los keypoints GT normalizados (0–1) de cada imagen desde su label.

    Cada label YOLO-pose es ``class cx cy w h`` seguido de 17 tripletas
    ``px py v`` normalizadas. Se toma la **primera** persona (UCO = 1 por frame).

    Args:
        image_paths: rutas de las imágenes cuyos labels cargar.

    Los frames sin label legible se omiten, pero si superan
    :data:`_MAX_MISSING_LABEL_RATIO` del total se aborta la fase: perder una
    fracción significativa del GT sesga OKS/PCK sin dejar rastro.

    Returns:
        ``{path: (17, 3)}`` con ``x, y`` normalizados 0–1 y ``v`` (0, 1 ó 2).
    """
    out: dict[Path, np.ndarray] = {}
    missing: list[Path] = []
    for img in image_paths:
        label = _label_path_for(img)
        if not label.exists():
            missing.append(label)
            continue
        first = next(
            (ln for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()),
            None,
        )
        if first is None:
            # Label vacío = frame sin persona detectada por el maestro (no es un
            # error de formato); no aporta GT pero tampoco invalida el dataset.
            continue
        vals = first.split()
        # 5 (class + bbox) + 17*3 keypoints = 56 columnas.
        expected = 5 + NUM_COCO_KEYPOINTS * 3
        if len(vals) < expected:
            missing.append(label)
            continue
        kpt_vals = np.array(vals[5:expected], dtype=np.float64).reshape(NUM_COCO_KEYPOINTS, 3)
        out[img] = kpt_vals

    if image_paths and len(missing) > len(image_paths) * _MAX_MISSING_LABEL_RATIO:
        strict.fail(
            f"{len(missing)}/{len(image_paths)} labels faltantes o con formato inesperado "
            f"(p. ej. {missing[0]}): el GT del fold está incompleto y las métricas "
            "quedarían sesgadas."
        )
    return out


def _pick_best_detection(result: Any) -> tuple[np.ndarray | None, np.ndarray | None]:  # noqa: ANN401
    """Extrae keypoints y bbox de la detección de mayor confianza de un Result.

    Args:
        result: un ``ultralytics.engine.results.Results``.

    Returns:
        Tupla ``(kpts_px (17,3), box_xywh_px (4,))``; ``(None, None)`` si no hay
        detección.
    """
    boxes = getattr(result, "boxes", None)
    kpts = getattr(result, "keypoints", None)
    if boxes is None or kpts is None or len(boxes) == 0:
        return None, None

    conf = boxes.conf.cpu().numpy()
    best = int(np.argmax(conf))

    kpt_data = kpts.data.cpu().numpy()  # (num_det, 17, 3)
    box_xywh = boxes.xywh.cpu().numpy()  # (num_det, 4) center x,y,w,h
    return kpt_data[best], box_xywh[best]


def run_predictions(
    model: Any,  # noqa: ANN401
    image_paths: list[Path],
    imgsz: int,
    conf: float = 0.25,
) -> dict[Path, dict[str, Any]]:
    """Corre inferencia con Ultralytics y devuelve la mejor detección por frame.

    Acepta una ruta (``.pt`` o ``.tflite``) o un objeto ``YOLO`` ya en memoria. Un
    ``YOLO`` en memoria **nunca se modifica**: se predice sobre una instantánea de su
    módulo recargada de disco. ``YOLO.predict`` arma un ``AutoBackend(fuse=True)``
    sobre el propio ``self.model`` y fusiona Conv+BN *in-place*, así que predecir
    sobre el wrapper del llamador le borraba los BatchNorm. En la Fase 5 eso dejaba
    al estudiante podado sin BN justo antes del fine-tune y la distillation (los
    −4,147 params entre ``params_pruned`` y ``params_pruned_ft``).

    Args:
        model: ruta al modelo o instancia ``ultralytics.YOLO``.
        image_paths: imágenes sobre las que predecir.
        imgsz: tamaño de imagen de inferencia.
        conf: umbral de confianza de detección.

    Returns:
        ``{path: {"kpts": (17,3)|None, "box": (4,)|None, "wh": (w, h)}}``.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    if isinstance(model, YOLO):
        import torch  # noqa: PLC0415

        # Instantánea en disco (mismo formato que ``pruning.save_pruned_checkpoint``) y
        # recarga en un wrapper nuevo: el fuse in-place cae sobre la copia, no sobre
        # el módulo del llamador (ver docstring).
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot.pt"
            torch.save({"model": model.model}, str(snapshot))
            return _predict_each(YOLO(str(snapshot), task="pose"), image_paths, imgsz, conf)

    # ``task="pose"`` es OBLIGATORIO para un ``.tflite``: a diferencia de un ``.pt``,
    # el TFLite no lleva la tarea en sus metadatos, así que Ultralytics avisa
    # "Unable to automatically guess model task, assuming 'task=detect'" y devuelve
    # 300 cajas SIN keypoints. La evaluación puntuaba entonces todos los frames como
    # fallo (OKS 0.0000) y el número parecía pérdida por cuantización en vez de un
    # error de carga. Para un ``.pt`` de pose es redundante pero inocuo, y este
    # módulo evalúa exclusivamente modelos de pose.
    return _predict_each(YOLO(str(model), task="pose"), image_paths, imgsz, conf)


def _predict_each(
    yolo: Any,  # noqa: ANN401
    image_paths: list[Path],
    imgsz: int,
    conf: float,
) -> dict[Path, dict[str, Any]]:
    """Mejor detección por frame con un ``YOLO`` ya cargado (propio de este módulo)."""
    # Se predice **una imagen por vez** (batch=1): un backend ``.tflite`` INT8 tiene
    # la dimensión de batch fija en 1, así que pasar la lista completa como una sola
    # fuente (que Ultralytics apila en un batch de N) rompe el intérprete. El loop
    # por imagen es correcto para ``.pt`` y ``.tflite`` por igual.
    out: dict[Path, dict[str, Any]] = {}
    for path in image_paths:
        result = yolo.predict(source=str(path), imgsz=imgsz, conf=conf, verbose=False)[0]
        h, w = result.orig_shape
        kpts, box = _pick_best_detection(result)
        out[path] = {"kpts": kpts, "box": box, "wh": (int(w), int(h))}
    return out


def build_eval_context(
    data_yaml: str | Path,
    base_model: Any,
    imgsz: int,  # noqa: ANN401
) -> EvalContext | None:
    """Construye el contexto de evaluación de un fold (GT + bbox de persona fijo).

    Corre el detector base una vez para derivar, por frame, el **bbox de persona
    completa** que normaliza OKS/PCK en las 4 etapas, y carga el GT en píxeles.

    Args:
        data_yaml: YAML de dataset del fold.
        base_model: ruta o ``YOLO`` del modelo base (yolov8n-pose sin optimizar).
        imgsz: tamaño de imagen de inferencia.

    Returns:
        :class:`EvalContext`, o ``None`` si no hay imágenes/GT evaluables.
    """
    image_paths = read_val_image_paths(data_yaml)
    if not image_paths:
        strict.fail(f"Sin imágenes de validación en {data_yaml}; no hay evaluación posible.")
        return None

    gt_norm = load_gt_normalized(image_paths)
    if not gt_norm:
        strict.fail(f"Sin labels GT legibles para {data_yaml}; no hay evaluación posible.")
        return None

    base_preds = run_predictions(base_model, list(gt_norm), imgsz)

    gt_kpts: dict[Path, np.ndarray] = {}
    person_boxes: dict[Path, np.ndarray] = {}
    dims: dict[Path, tuple[int, int]] = {}
    kept_preds: dict[Path, dict[str, Any]] = {}
    kept: list[Path] = []
    skipped = 0
    for path, gt in gt_norm.items():
        pred = base_preds.get(path)
        if pred is None or pred["box"] is None:
            skipped += 1  # el detector base no encontró persona → sin normalizador
            continue
        w, h = pred["wh"]
        gt_px = gt.copy()
        gt_px[:, 0] *= w
        gt_px[:, 1] *= h
        gt_kpts[path] = gt_px
        person_boxes[path] = np.asarray(pred["box"], dtype=np.float64)
        dims[path] = (w, h)
        kept_preds[path] = pred
        kept.append(path)

    if skipped:
        logger.info(
            "%d/%d frames sin bbox de persona base; excluidos de OKS/PCK.", skipped, len(gt_norm)
        )
    if not kept:
        strict.fail(
            f"El detector base no encontró persona en ninguno de los {len(gt_norm)} frames "
            f"de {data_yaml}: sin bbox normalizador no se pueden computar OKS/PCK."
        )
        return None
    return EvalContext(
        gt_kpts=gt_kpts,
        person_boxes=person_boxes,
        dims=dims,
        img_paths=kept,
        base_preds=kept_preds,
    )


def _group_of(image_path: Path) -> str | None:
    """Grupo muscular del ejercicio de un frame, derivado del stem del archivo.

    Usa :func:`~src.data.uco_to_yolo.parse_stem`, la única definición del formato
    del stem, para recuperar el nº de ejercicio y mapearlo a pierna o brazo.

    Args:
        image_path: ruta de la imagen en el pool plano.

    Returns:
        ``"leg"``, ``"arm"``, o ``None`` si el ejercicio no pertenece a ninguno
        de los dos grupos o el stem no respeta el formato.
    """
    try:
        exercise = parse_stem(image_path.stem).exercise
    except ValueError:
        logger.warning("Stem fuera de formato, sin desagregado pierna/brazo: %s", image_path.name)
        return None
    if exercise in _LEG_EXERCISES:
        return "leg"
    if exercise in _ARM_EXERCISES:
        return "arm"
    return None


def _pck_over(
    paths: list[Path],
    pred_map: dict[Path, dict[str, Any]],
    ctx: EvalContext,
    threshold: float = 0.2,
) -> float | None:
    """PCK@threshold sobre un subconjunto de frames, o ``None`` si está vacío."""
    if not paths:
        return None
    pred_arr = np.zeros((len(paths), NUM_COCO_KEYPOINTS, 3), dtype=np.float64)
    gt_arr = np.zeros_like(pred_arr)
    ref = np.zeros(len(paths), dtype=np.float64)
    for i, path in enumerate(paths):
        gt_arr[i] = ctx.gt_kpts[path]
        box = ctx.person_boxes[path]
        ref[i] = float(np.hypot(box[2], box[3]))  # diagonal del bbox de persona
        kpts = pred_map.get(path, {}).get("kpts")
        if kpts is not None:
            pred_arr[i] = kpts
    return metric_utils.compute_pck(pred_arr, gt_arr, threshold=threshold, reference_size=ref)


def score_stage(pred_map: dict[Path, dict[str, Any]], ctx: EvalContext) -> StageScore:
    """Puntúa una etapa (OKS + PCK, con desagregado pierna/brazo).

    Arma los arrays ``(N, 17, 3)`` alineados frame a frame contra el GT y el bbox
    de persona fijos del contexto y llama a :func:`metrics.compute_oks` /
    :func:`metrics.compute_pck`. Frames sin detección de la etapa quedan con
    keypoints en cero (puntúan como error, que es el comportamiento correcto).

    Args:
        pred_map: salida de :func:`run_predictions` para el modelo de la etapa.
        ctx: contexto de evaluación del fold.

    Returns:
        :class:`StageScore` con las métricas de la etapa.
    """
    paths = ctx.img_paths
    n = len(paths)
    pred_arr = np.zeros((n, NUM_COCO_KEYPOINTS, 3), dtype=np.float64)
    gt_arr = np.zeros_like(pred_arr)
    boxes = np.zeros((n, 4), dtype=np.float64)
    leg_paths: list[Path] = []
    arm_paths: list[Path] = []
    for i, path in enumerate(paths):
        gt = ctx.gt_kpts[path]
        gt_arr[i] = gt
        boxes[i] = ctx.person_boxes[path]
        kpts = pred_map.get(path, {}).get("kpts")
        if kpts is not None:
            pred_arr[i] = kpts
        group = _group_of(path)
        if group == "leg":
            leg_paths.append(path)
        elif group == "arm":
            arm_paths.append(path)

    oks = metric_utils.compute_oks(pred_arr, gt_arr, boxes)
    pck = _pck_over(paths, pred_map, ctx)
    pck_leg = _pck_over(leg_paths, pred_map, ctx)
    pck_arm = _pck_over(arm_paths, pred_map, ctx)
    return StageScore(oks=oks, pck=pck, pck_leg=pck_leg, pck_arm=pck_arm, n_frames=n)


def count_params(model: Any) -> int | None:  # noqa: ANN401
    """Nº de parámetros del modelo (``.pt`` o ``YOLO`` en memoria); ``None`` para ``.tflite``.

    Args:
        model: ruta al modelo o instancia ``ultralytics.YOLO``.

    Returns:
        Cantidad de parámetros, o ``None`` si no se puede introspeccionar
        (p. ej. un ``.tflite``, que no expone un ``nn.Module``).
    """
    from ultralytics import YOLO  # noqa: PLC0415

    try:
        if isinstance(model, YOLO):
            yolo = model
        else:
            if Path(str(model)).suffix.lower() == ".tflite":
                return None
            yolo = YOLO(str(model))
        return int(sum(p.numel() for p in yolo.model.parameters()))  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        logger.info("No se pudo contar parámetros de %r: %s", model, exc)
        return None
