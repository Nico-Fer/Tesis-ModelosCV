"""Runner de la FASE 5 (pruning + distillation + cuantización INT8) del pipeline experimental.

Combina las técnicas de optimización en orden: (1) pruning estructurado del
estudiante, (2) fine-tuning de recuperación post-poda, (3) distillation con el
maestro sobre el estudiante ya podado, (4) cuantización INT8 del resultado.

La poda va **antes** del entrenamiento a propósito: así tanto el fine-tuning
como la distillation entrenan ya la arquitectura reducida (ambos usan trainers
que preservan los canales podados; con ``YOLO.train`` a secas Ultralytics
reconstruiría el modelo desde el YAML y desharía la poda).

Uso:
    python -m src.training.combined --config configs/combined_yolov8n.yaml

El YAML debe incluir ``teacher_model`` (checkpoint del maestro) además de
``model`` (checkpoint inicial del estudiante), y las secciones ``pruning`` y
``distillation``.

Por defecto la fase **falla duro** ante cualquier etapa que no produzca su
resultado (ver :mod:`src.utils.strict`); ``smoke_test: true`` en el YAML es la
única vía para correr el esqueleto sin dataset ni pesos reales.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml

from src.evaluation import metrics as metric_utils
from src.evaluation import pose_eval
from src.models.base import ModelMetrics
from src.models.registry import build_model
from src.optimization import distillation as distill_utils
from src.optimization import pruning as pruning_utils
from src.optimization import quantization as quant_utils
from src.training import _kfold as kfold
from src.utils import checkpoints, mlflow_utils, strict

logger = logging.getLogger(__name__)

# Marca de run NO válido; sólo se loguea en smoke test (ver train_baseline).
_SMOKE_TEST_METRIC = "pipeline_smoke_test"


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con el atributo ``config`` (ruta al YAML de la fase).
    """
    parser = argparse.ArgumentParser(description="Runner Fase 5 (combined) del pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Ruta al YAML de configuración (ver configs/).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Carga y valida mínimamente un YAML de configuración.

    Args:
        path: ruta al archivo YAML.

    Returns:
        Configuración como diccionario.

    Raises:
        FileNotFoundError: si el archivo no existe.
        KeyError: si falta alguna clave obligatoria.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
    with path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}
    for required in ("family", "model", "teacher_model"):
        if required not in config:
            raise KeyError(f"Falta la clave obligatoria {required!r} en {path}.")
    return config


def _flatten_params(config: dict[str, Any]) -> dict[str, Any]:
    """Aplana la config a un solo nivel para loguearla como parámetros.

    Args:
        config: configuración cargada.

    Returns:
        Diccionario de parámetros aplanado.
    """
    flat: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def _log_stage_metrics(
    stage: str,
    *,
    score: pose_eval.StageScore | None = None,
    size_mb: float | None = None,
    params: int | None = None,
    latency_ms: float | None = None,
) -> dict[str, float]:
    """Loguea en MLflow las métricas de una etapa con nombres sufijados (``*_{stage}``).

    Los sufijos distintos (``oks_base``, ``oks_pruned``, …) permiten comparar la
    misma métrica a lo largo de las 5 etapas en un mismo run.

    Args:
        stage: identificador de etapa
            (``base``/``pruned``/``pruned_ft``/``distilled``/``int8``).
        score: puntajes OKS/PCK de la etapa (si se pudo evaluar contra el GT).
        size_mb: tamaño del artefacto en MB.
        params: nº de parámetros del modelo (muestra el efecto de la poda).
        latency_ms: latencia media de inferencia sobre el ``.tflite`` de la etapa
            (fp32 para base/podado/distilado, INT8 para la etapa final).

    Returns:
        Las métricas logueadas (para registrarlas también en el checkpoint del fold).
    """
    m: dict[str, float] = {}
    if size_mb is not None:
        m[f"size_mb_{stage}"] = float(size_mb)
    if params is not None:
        m[f"params_{stage}"] = float(params)
    if latency_ms is not None:
        m[f"latency_ms_{stage}"] = float(latency_ms)
    if score is not None:
        if score.oks is not None:
            m[f"oks_{stage}"] = float(score.oks)
        if score.pck is not None:
            m[f"pck_{stage}"] = float(score.pck)
        if score.pck_leg is not None:
            m[f"pck_leg_{stage}"] = float(score.pck_leg)
        if score.pck_arm is not None:
            m[f"pck_arm_{stage}"] = float(score.pck_arm)
        m[f"n_frames_{stage}"] = float(score.n_frames)
    if m:
        mlflow.log_metrics(m)
    return m


def _size_mb_or_none(path: str | Path | None) -> float | None:
    """Tamaño en MB de un artefacto si existe en disco; ``None`` si no."""
    if not path:
        return None
    p = Path(path)
    return metric_utils.model_size_mb(p) if p.exists() else None


def _tflite_latency_from_pt(pt_path: str | Path, imgsz: int, tag: str) -> float | None:
    """Exporta un checkpoint ``.pt`` a TFLite Float32 y mide su latencia media (ms).

    Medir la latencia de las 4 etapas **sobre TFLite** (el backend de deployment)
    las hace comparables entre sí: un ``.pt`` (PyTorch) y un ``.tflite`` no lo son.
    Las etapas fp32 (base/podado/distilado) se exportan sin cuantizar; la etapa
    INT8 mide sobre su propio ``.tflite`` cuantizado (ver :func:`_run_once`).

    La latencia es una de las tres métricas de deployment de la tesis, así que un
    fallo de export propaga; sólo en smoke test devuelve ``None``.

    Args:
        pt_path: checkpoint ``.pt`` de la etapa a exportar.
        imgsz: tamaño de imagen (define el shape de entrada del TFLite).
        tag: identificador de etapa para los logs.

    Returns:
        Latencia media en ms, o ``None`` si el export/medición falla en smoke test.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    try:
        # Se exporta una COPIA en un dir temporal corto (bajo el repo, por el límite
        # MAX_PATH de Windows). Ultralytics escribe el ``*_saved_model/`` junto al ``.pt``
        # de origen y arranca con un ``rmtree`` de ese dir: exportando el
        # ``models/yolov8n-pose.pt`` original, la etapa base borraba en cada fold el INT8
        # de control de ``models/yolov8n-pose_saved_model/``.
        exports_dir = Path("models/exports")
        exports_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=exports_dir) as tmp:
            pt_copy = Path(tmp) / f"{tag}.pt"
            shutil.copy2(pt_path, pt_copy)
            artifact = Path(YOLO(str(pt_copy)).export(format="tflite", imgsz=imgsz))
            return metric_utils.measure_latency(artifact, (1, imgsz, imgsz, 3))
    except Exception as exc:  # noqa: BLE001
        if not strict.is_smoke():
            raise
        logger.warning("[smoke] No se pudo medir la latencia TFLite (%s): %s", tag, exc)
        return None


def _metrics_from_score(
    score: pose_eval.StageScore | None,
    size_mb: float | None,
    latency_ms: float | None = None,
) -> ModelMetrics:
    """Arma un :class:`ModelMetrics` a partir del puntaje de una etapa."""
    return ModelMetrics(
        oks=score.oks if score else None,
        pck=score.pck if score else None,
        latency_ms=latency_ms,
        size_mb=size_mb,
    )


def _eval_context(
    data: str | None,
    base_path: str,
    imgsz: int,
    ckpt: checkpoints.FoldCheckpoint | None,
) -> pose_eval.EvalContext | None:
    """Contexto de evaluación del fold: del checkpoint si existe, si no se construye.

    Un fallo al construirlo deja las 5 etapas sin OKS/PCK, es decir sin la
    comparación que justifica la fase: propaga salvo en smoke test.
    """
    if not data:
        return None
    if ckpt is not None:
        cached = ckpt.load_ctx()
        if cached is not None:
            logger.info("Contexto de evaluación reusado del checkpoint (%s).", ckpt.dir)
            return cached
    try:
        ctx = pose_eval.build_eval_context(data, base_path, imgsz)
    except Exception as exc:  # noqa: BLE001
        if not strict.is_smoke():
            raise
        logger.warning("[smoke] No se pudo construir el contexto de evaluación: %s", exc)
        return None
    if ckpt is not None and ctx is not None:
        ckpt.save_ctx(ctx)
    return ctx


def _mark(
    ckpt: checkpoints.FoldCheckpoint | None,
    stage: str,
    metrics: ModelMetrics,
    logged: dict[str, float],
    **artifacts: str,
) -> None:
    """Registra la etapa como terminada en el checkpoint del fold (si lo hay)."""
    if ckpt is not None:
        ckpt.mark(stage, metrics=metrics, logged=logged, **artifacts)


def _run_once(config: dict[str, Any]) -> ModelMetrics:
    """Trabajo de la Fase 5 con un run MLflow ya activo (un fold o el run único).

    Evalúa el modelo en **5 etapas comparables** (base → podado → podado+recuperado
    → distilado → INT8), logueando OKS/PCK/size_mb/params/latencia con nombres
    sufijados por etapa. La latencia se mide sobre TFLite (fp32 para las etapas
    en float, INT8 en la última) para que sea comparable entre sí; ``pruned_ft``
    no la re-mide porque comparte arquitectura con ``pruned``. La evaluación
    de keypoints usa un **evaluador unificado**
    (:mod:`src.evaluation.pose_eval`) con el mismo bbox de persona normalizador en
    todas las etapas, de modo que las métricas sean directamente comparables. Lo
    llama :func:`run_combined` vía :mod:`src.training._kfold`, una vez por fold.

    Con ``checkpoint_dir`` (lo fija ``_kfold`` por fold) cada etapa terminada queda
    registrada con su artefacto en disco, y una reanudación tras un corte la saltea
    (ver :mod:`src.utils.checkpoints`).

    Args:
        config: configuración de la fase/modelo (con ``data`` ya apuntando al fold).

    Returns:
        Las :class:`~src.models.base.ModelMetrics` del estudiante INT8 final (o de
        la última etapa alcanzada si la cuantización se omite), para que la
        agregación k-fold tenga métricas comparables entre folds.
    """
    strict.set_smoke_mode(bool(config.get("smoke_test", False)))

    teacher_config: dict[str, Any] = {**config, "model": config["teacher_model"]}
    teacher = build_model(teacher_config)
    student = build_model(config)

    params = _flatten_params(config)
    params.update(distill_utils.build_distillation_loss(config))
    mlflow_utils.log_params(params)

    imgsz = int(config.get("imgsz", 640))
    data = config.get("data")
    base_path = config["model"]

    # Checkpoints por etapa (sólo con ``checkpoint_dir``, que _kfold fija por fold; ver
    # src.utils.checkpoints): una etapa ya registrada se saltea y la siguiente arranca
    # del artefacto que dejó en disco, así un corte de luz no obliga a repetir el fold.
    ckpt = checkpoints.FoldCheckpoint.from_config(config)
    # Dir corto (bajo el repo) para los .pt intermedios: evita el límite MAX_PATH de
    # Windows que rompe el export en rutas muy largas.
    work_dir = ckpt.dir if ckpt is not None else Path("models/exports")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Contexto de evaluación (GT + bbox de persona fijo por frame). Se deriva una
    # sola vez del detector base y se reutiliza en las 5 etapas (comparabilidad).
    ctx = _eval_context(data, base_path, imgsz, ckpt)

    final_metrics = ModelMetrics()

    # --- Etapa 1: base (estudiante sin optimizar) ---
    if ckpt is not None and ckpt.done("base"):
        final_metrics = ckpt.metrics("base")
    else:
        base_size = _size_mb_or_none(base_path)
        base_params = pose_eval.count_params(base_path)
        base_latency = _tflite_latency_from_pt(base_path, imgsz, "base") if data else None
        base_score = None
        if ctx is not None:
            # Reusa las predicciones del detector base ya calculadas al armar el
            # contexto (evita una pasada de inferencia redundante sobre el val).
            base_score = pose_eval.score_stage(ctx.base_preds, ctx)
        logged = _log_stage_metrics(
            "base", score=base_score, size_mb=base_size, params=base_params, latency_ms=base_latency
        )
        final_metrics = _metrics_from_score(base_score, base_size, base_latency)
        _mark(ckpt, "base", final_metrics, logged)

    # --- Etapa 2: poda estructurada del estudiante ---
    # El podado se persiste y el fine-tune lo RECARGA de disco: es el mismo camino en una
    # corrida de corrido y en una reanudada, así un bug de la reanudación aparece en el
    # primer fold y no recién después de un corte.
    pruned_pt = work_dir / "student_pruned.pt"
    pruned_latency: float | None = None
    have_pruned = False
    if ckpt is not None and ckpt.done("pruned"):
        final_metrics = ckpt.metrics("pruned")
        pruned_latency = ckpt.get("pruned")["logged"].get("latency_ms_pruned")
        have_pruned = True
    else:
        student = pruning_utils.apply_structured_pruning(student, config)
        pruned_score = None
        pruned_params = None
        if student._model is not None:
            pruned_params = pose_eval.count_params(student._model)
            pruning_utils.save_pruned_checkpoint(student._model, pruned_pt)
            have_pruned = True
            if data:
                pruned_latency = _tflite_latency_from_pt(pruned_pt, imgsz, "pruned")
            if ctx is not None:
                pruned_score = pose_eval.score_stage(
                    pose_eval.run_predictions(pruned_pt, ctx.img_paths, imgsz), ctx
                )
        logged = _log_stage_metrics(
            "pruned", score=pruned_score, params=pruned_params, latency_ms=pruned_latency
        )
        if pruned_score is not None or pruned_latency is not None:
            final_metrics = _metrics_from_score(pruned_score, None, pruned_latency)
        _mark(ckpt, "pruned", final_metrics, logged, pt=str(pruned_pt))
    if have_pruned:
        student.config["model"] = str(pruned_pt)
        student._model = None

    # --- Etapa 3: fine-tuning de recuperación post-poda ---
    # Se evalúa como etapa aparte de 'pruned' a propósito: 'pruned' mide el DAÑO de
    # la poda y 'pruned_ft' la RECUPERACIÓN. Colapsarlas en una sola métrica haría
    # indistinguible una poda benigna de una poda agresiva bien recuperada.
    ft_pt = work_dir / "student_pruned_ft.pt"
    have_ft = False
    if ckpt is not None and ckpt.done("pruned_ft"):
        final_metrics = ckpt.metrics("pruned_ft")
        have_ft = True
    else:
        pruning_utils.fine_tune_after_pruning(student, config)
        finetuned_score = None
        finetuned_params = None
        if student._model is not None:
            finetuned_params = pose_eval.count_params(student._model)
            # Se persiste el estudiante podado+recuperado y se RECARGA en un YOLO limpio.
            # Reusar el wrapper recién entrenado falla de dos formas distintas: la
            # inferencia revienta en NMS (``'list' object has no attribute 'shape'``) y un
            # segundo ``train()`` — la distillation — muere con ``KeyError: 'model'``,
            # porque el primer ``train()`` ya reemplazó ``overrides`` por los args del
            # checkpoint. Es el mismo idioma con el que ``run_distillation_training``
            # cierra su propia etapa.
            pruning_utils.save_pruned_checkpoint(student._model, ft_pt)
            have_ft = True
            if ctx is not None:
                finetuned_score = pose_eval.score_stage(
                    pose_eval.run_predictions(ft_pt, ctx.img_paths, imgsz), ctx
                )
        # Sin latencia ni tamaño propios: el fine-tuning ajusta pesos, no la arquitectura,
        # así que ambos son por construcción los de la etapa 'pruned'. Re-exportar a
        # TFLite sólo para reproducir el mismo número no agrega información.
        logged = _log_stage_metrics("pruned_ft", score=finetuned_score, params=finetuned_params)
        if finetuned_score is not None:
            final_metrics = _metrics_from_score(finetuned_score, None, pruned_latency)
        _mark(ckpt, "pruned_ft", final_metrics, logged, pt=str(ft_pt))
    if have_ft:
        student.config["model"] = str(ft_pt)
        student._model = None  # fuerza la recarga limpia en la etapa siguiente

    # --- Etapa 4: distillation sobre el estudiante YA PODADO Y RECUPERADO ---
    if ckpt is not None and ckpt.done("distilled"):
        final_metrics = ckpt.metrics("distilled")
        student.config["model"] = ckpt.artifact("distilled", "pt")
        student._model = None
    else:
        distill_utils.run_distillation_training(teacher, student, config)
        distilled_path = student.config.get("model")
        distilled_size = _size_mb_or_none(distilled_path)
        distilled_params = pose_eval.count_params(distilled_path) if distilled_path else None
        distilled_latency = None
        distilled_score = None
        if distilled_path and Path(distilled_path).exists():
            if data:
                distilled_latency = _tflite_latency_from_pt(distilled_path, imgsz, "distilled")
            if ctx is not None:
                distilled_score = pose_eval.score_stage(
                    pose_eval.run_predictions(distilled_path, ctx.img_paths, imgsz), ctx
                )
        logged = _log_stage_metrics(
            "distilled",
            score=distilled_score,
            size_mb=distilled_size,
            params=distilled_params,
            latency_ms=distilled_latency,
        )
        if distilled_score is not None or distilled_size is not None:
            final_metrics = _metrics_from_score(distilled_score, distilled_size, distilled_latency)
        _mark(ckpt, "distilled", final_metrics, logged, pt=str(distilled_path))

    if strict.is_smoke():
        mlflow.log_metric(_SMOKE_TEST_METRIC, 1.0)

    # --- Etapa 5: cuantización INT8 del estudiante optimizado ---
    if ckpt is not None and ckpt.done("int8"):
        return ckpt.metrics("int8")

    # Acota las imágenes de calibración (mismo guard que la Fase 3): sin él,
    # calibrar con los ~16k frames del val agota la memoria.
    if data:
        n_val = len(ctx.img_paths) if ctx is not None else len(pose_eval.read_val_image_paths(data))
        quant_utils.cap_calibration_fraction(student.config, n_val)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            quant_utils.prepare_calibration_dataset(config)
            artifact = quant_utils.quantize_to_int8(student, config, Path(tmp))

            int8_size = _size_mb_or_none(artifact)
            int8_latency: float | None = None
            try:
                int8_latency = metric_utils.measure_latency(artifact, (1, imgsz, imgsz, 3))
            except Exception as exc:  # noqa: BLE001
                if not strict.is_smoke():
                    raise
                logger.warning("[smoke] No se pudo medir la latencia del .tflite INT8: %s", exc)

            int8_score = None
            if ctx is not None:
                int8_score = pose_eval.score_stage(
                    pose_eval.run_predictions(artifact, ctx.img_paths, imgsz), ctx
                )
            logged = _log_stage_metrics(
                "int8", score=int8_score, size_mb=int8_size, latency_ms=int8_latency
            )
            final_metrics = _metrics_from_score(int8_score, int8_size, int8_latency)

            mlflow_utils.log_model_artifact(artifact)
            _mark(ckpt, "int8", final_metrics, logged, tflite=str(artifact))
        except Exception as exc:  # noqa: BLE001
            # El .tflite INT8 es EL entregable de la Fase 5 (el que va a la app):
            # si falla, el run no tiene resultado y no debe pasar por válido.
            if not strict.is_smoke():
                raise
            note = Path(tmp) / "export_skipped.txt"
            note.write_text(f"Export TFLite INT8 omitida: {exc}\n", encoding="utf-8")
            mlflow_utils.log_model_artifact(note, artifact_path="notes")

    return final_metrics


def run_combined(config: dict[str, Any]) -> ModelMetrics:
    """Ejecuta la Fase 5 (prune + distill + INT8), con GroupKFold si hay folds.

    Delega el ciclo de folds y el run padre a :func:`src.training._kfold.run_phase`.

    Args:
        config: configuración de la fase/modelo (requiere ``teacher_model``).

    Returns:
        Las :class:`~src.models.base.ModelMetrics` (del run único o promediadas).
    """
    mlflow_cfg: dict[str, Any] = config.get("mlflow", {})
    experiment: str = mlflow_cfg.get("experiment", "fase5-combined")
    return kfold.run_phase(
        config,
        _run_once,
        experiment=experiment,
        tags={"fase": "5", "familia": config["family"]},
    )


def main() -> None:
    """Punto de entrada CLI."""
    # Sin esto los logger.info de src/ no se ven (Ultralytics y MLflow tienen handlers
    # propios, el pipeline no): entre otros, los que dicen qué fold/etapa se reanuda.
    # Con ``python -m`` este módulo se llama ``__main__``, fuera del árbol ``src``.
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("src", __name__):
        logging.getLogger(name).setLevel(logging.INFO)
    args = parse_args()
    config = load_config(args.config)
    metrics = run_combined(config)
    print(f"Fase 5 completada. Métricas: {metrics.to_loggable()}")


if __name__ == "__main__":
    main()
