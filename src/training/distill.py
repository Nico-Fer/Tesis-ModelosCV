"""Runner de la FASE 4 (knowledge distillation Maestro → Estudiante) del pipeline experimental.

Entrena un estudiante ligero (p. ej. yolov8n-pose) supervisado por las soft
targets de un maestro pre-entrenado (p. ej. yolov8x-pose). La lógica de
distillation se delega a :mod:`src.optimization.distillation`.

Uso:
    python -m src.training.distill --config configs/distill_yolov8n.yaml

El YAML debe incluir ``teacher_model`` (checkpoint del maestro) además de
``model`` (checkpoint inicial del estudiante).

El esqueleto corre end-to-end sin dataset: distillation y evaluate toleran
ausencia de datos al igual que el resto de los runners del pipeline.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml

from src.models.base import ModelMetrics
from src.models.registry import build_model
from src.optimization import distillation as distill_utils
from src.training import _kfold as kfold
from src.utils import mlflow_utils, strict

_SMOKE_TEST_METRIC = "pipeline_smoke_test"


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con el atributo ``config`` (ruta al YAML de la fase).
    """
    parser = argparse.ArgumentParser(description="Runner Fase 4 (distillation) del pipeline.")
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


def _run_once(config: dict[str, Any]) -> ModelMetrics:
    """Trabajo de la Fase 4 con un run MLflow ya activo (un fold o el run único).

    Loguea hiperparámetros (incluidos los de la loss de distillation), construye
    maestro y estudiante, corre el ciclo de distillation, evalúa el estudiante y
    exporta a TFLite Float32. Lo llama :func:`run_distill` vía
    :mod:`src.training._kfold`, una vez por fold.

    Args:
        config: configuración de la fase/modelo (con ``data`` ya apuntando al fold).

    Returns:
        Las :class:`~src.models.base.ModelMetrics` del estudiante tras la distillation.
    """
    strict.set_smoke_mode(bool(config.get("smoke_test", False)))
    export_cfg: dict[str, Any] = config.get("export", {})

    # El maestro usa la misma familia pero un checkpoint diferente.
    teacher_config: dict[str, Any] = {**config, "model": config["teacher_model"]}
    teacher = build_model(teacher_config)
    student = build_model(config)

    params = _flatten_params(config)
    # Incluir hiperparámetros de la loss de distillation en el run.
    params.update(distill_utils.build_distillation_loss(config))
    mlflow_utils.log_params(params)

    # Ciclo de entrenamiento con distillation (attention transfer, ver el módulo).
    distill_utils.run_distillation_training(teacher, student, config)

    metrics = student.evaluate()
    mlflow_utils.log_metrics(metrics)

    if strict.is_smoke():
        mlflow.log_metric(_SMOKE_TEST_METRIC, 1.0)

    if export_cfg.get("format") == "tflite":
        with tempfile.TemporaryDirectory() as tmp:
            try:
                artifact = student.export_tflite(
                    output_dir=Path(tmp),
                    quantize=bool(export_cfg.get("quantize", False)),
                )
                mlflow_utils.log_model_artifact(artifact)
            except Exception as exc:  # noqa: BLE001
                if not strict.is_smoke():
                    raise
                note = Path(tmp) / "export_skipped.txt"
                note.write_text(f"Export TFLite omitida: {exc}\n", encoding="utf-8")
                mlflow_utils.log_model_artifact(note, artifact_path="notes")

    return metrics


def run_distill(config: dict[str, Any]) -> ModelMetrics:
    """Ejecuta la Fase 4, con GroupKFold por sujeto si el dataset tiene folds.

    Delega el ciclo de folds y el run padre a :func:`src.training._kfold.run_phase`.

    Args:
        config: configuración de la fase/modelo (requiere ``teacher_model``).

    Returns:
        Las :class:`~src.models.base.ModelMetrics` (del run único o promediadas).
    """
    mlflow_cfg: dict[str, Any] = config.get("mlflow", {})
    experiment: str = mlflow_cfg.get("experiment", "fase4-distillation")
    return kfold.run_phase(
        config,
        _run_once,
        experiment=experiment,
        tags={"fase": "4", "familia": config["family"]},
    )


def main() -> None:
    """Punto de entrada CLI."""
    args = parse_args()
    config = load_config(args.config)
    metrics = run_distill(config)
    print(f"Fase 4 completada. Métricas: {metrics.to_loggable()}")


if __name__ == "__main__":
    main()
