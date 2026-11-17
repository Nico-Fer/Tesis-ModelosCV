"""Runner de la FASE 3 (cuantización post-training FP32 → INT8) del pipeline experimental.

Toma el modelo base (o el podado en Fase 2), prepara el set de calibración,
aplica cuantización INT8 y registra el artefacto TFLite INT8 en MLflow.
La lógica de cuantización se delega a :mod:`src.optimization.quantization`.

Uso:
    python -m src.training.quantize --config configs/quantize_yolov8n.yaml

Por defecto la fase **falla duro** si falta el set de calibración o si la
exportación INT8 no produce el artefacto (ver :mod:`src.utils.strict`). Para
correr el esqueleto sin datos hay que declarar ``smoke_test: true`` en el YAML.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml

from src.evaluation import metrics as metric_utils
from src.evaluation import pose_eval
from src.models.base import ModelMetrics
from src.models.registry import build_model
from src.optimization import quantization as quant_utils
from src.training import _kfold as kfold
from src.utils import mlflow_utils, strict

logger = logging.getLogger(__name__)

# Marca de run NO válido; sólo se loguea en smoke test (ver train_baseline).
_SMOKE_TEST_METRIC = "pipeline_smoke_test"


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con el atributo ``config`` (ruta al YAML de la fase).
    """
    parser = argparse.ArgumentParser(description="Runner Fase 3 (cuantización INT8) del pipeline.")
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
        KeyError: si falta una clave obligatoria (``family``, ``model``).
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
    with path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}
    for required in ("family", "model"):
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


def _evaluate_int8(artifact: Path, config: dict[str, Any]) -> ModelMetrics | None:
    """Evalúa el ``.tflite`` INT8 ya exportado y loguea sus métricas con sufijo ``_int8``.

    Sin esto la fase logueaba ``oks``/``pck`` de :meth:`PoseModel.evaluate`, que son
    los del modelo **FP32** — en un experimento llamado ``fase3-quantization``, o sea
    con toda la apariencia de ser los del INT8. La pérdida por cuantizar, que es lo
    único que esta fase existe para medir, no se medía.

    El bbox normalizador sale de ``eval_base_model`` (default: el propio ``model``).
    Apuntarlo al mismo detector base que usó la Fase 5 es lo que hace estas métricas
    comparables con las de sus 5 etapas; con el default sólo son comparables entre sí.

    Args:
        artifact: ``.tflite`` INT8 exportado.
        config: configuración de la fase (con ``data`` ya apuntando al fold).

    Returns:
        Las métricas del INT8, o ``None`` si no hay ``data`` (o falla en smoke test).
    """
    data = config.get("data")
    if not data:
        return None

    imgsz = int(config.get("imgsz", 640))
    base_model = config.get("eval_base_model", config["model"])
    try:
        ctx = pose_eval.build_eval_context(data, base_model, imgsz)
        if ctx is None:
            return None
        score = pose_eval.score_stage(
            pose_eval.run_predictions(artifact, ctx.img_paths, imgsz), ctx
        )
        latency = metric_utils.measure_latency(artifact, (1, imgsz, imgsz, 3))
        size_mb = metric_utils.model_size_mb(artifact)
    except Exception as exc:  # noqa: BLE001
        if not strict.is_smoke():
            raise
        logger.warning("[smoke] No se pudo evaluar el .tflite INT8: %s", exc)
        return None

    logged: dict[str, float] = {
        "latency_ms_int8": float(latency),
        "size_mb_int8": float(size_mb),
        "n_frames_int8": float(score.n_frames),
    }
    if score.oks is not None:
        logged["oks_int8"] = float(score.oks)
    if score.pck is not None:
        logged["pck_int8"] = float(score.pck)
    mlflow.log_metrics(logged)
    return ModelMetrics(oks=score.oks, pck=score.pck, latency_ms=latency, size_mb=size_mb)


def _run_once(config: dict[str, Any]) -> ModelMetrics:
    """Trabajo de la Fase 3 con un run MLflow ya activo (un fold o el run único).

    Loguea hiperparámetros, prepara el dataset de calibración, evalúa en Float32
    y exporta a TFLite INT8. Lo llama :func:`run_quantize` vía
    :mod:`src.training._kfold`, una vez por fold.

    Args:
        config: configuración de la fase/modelo (con ``data`` ya apuntando al fold).

    Returns:
        Las :class:`~src.models.base.ModelMetrics` obtenidas tras la cuantización.
    """
    strict.set_smoke_mode(bool(config.get("smoke_test", False)))

    mlflow_utils.log_params(_flatten_params(config))

    model = build_model(config)

    # Preparar dataset de calibración (None en smoke test).
    quant_utils.prepare_calibration_dataset(config)

    model.load()
    metrics = model.evaluate()
    mlflow_utils.log_metrics(metrics)

    if strict.is_smoke():
        mlflow.log_metric(_SMOKE_TEST_METRIC, 1.0)

    # Acota las imágenes de calibración: sin este tope, calibrar con los ~16k
    # frames del split val de un fold agota la memoria (mismo guard que Fase 5).
    data = config.get("data")
    if data:
        quant_utils.cap_calibration_fraction(
            model.config, len(pose_eval.read_val_image_paths(data))
        )

    # La cuantización INT8 se aplica siempre en esta fase.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            artifact = quant_utils.quantize_to_int8(model, config, Path(tmp))
            mlflow_utils.log_model_artifact(artifact)
            int8_metrics = _evaluate_int8(artifact, config)
            if int8_metrics is not None:
                metrics = int8_metrics
        except Exception as exc:  # noqa: BLE001
            # El .tflite INT8 es el entregable de la Fase 3: si falla, el run no vale.
            if not strict.is_smoke():
                raise
            note = Path(tmp) / "export_skipped.txt"
            note.write_text(f"Export TFLite INT8 omitida: {exc}\n", encoding="utf-8")
            mlflow_utils.log_model_artifact(note, artifact_path="notes")

    return metrics


def run_quantize(config: dict[str, Any]) -> ModelMetrics:
    """Ejecuta la Fase 3, con GroupKFold por sujeto si el dataset tiene folds.

    Delega el ciclo de folds y el run padre a :func:`src.training._kfold.run_phase`.

    Args:
        config: configuración de la fase/modelo.

    Returns:
        Las :class:`~src.models.base.ModelMetrics` (del run único o promediadas).
    """
    mlflow_cfg: dict[str, Any] = config.get("mlflow", {})
    experiment: str = mlflow_cfg.get("experiment", "fase3-quantization")
    return kfold.run_phase(
        config,
        _run_once,
        experiment=experiment,
        tags={"fase": "3", "familia": config["family"]},
    )


def main() -> None:
    """Punto de entrada CLI."""
    args = parse_args()
    config = load_config(args.config)
    metrics = run_quantize(config)
    print(f"Fase 3 completada. Métricas: {metrics.to_loggable()}")


if __name__ == "__main__":
    main()
