"""Runner de la FASE 1 (baseline) del pipeline experimental.

Entrena/evalúa un modelo en su configuración original, exporta el artefacto
TFLite (Float32 por defecto) y registra un run de MLflow con hiperparámetros,
métricas comparables y el artefacto exportado.

Diseñado como plantilla extensible: las Fases 2–5 añaden runners hermanos
(``prune.py``, ``quantize.py``, ``distill.py``, ``combined.py``) que reusan
``src.utils.mlflow_utils`` y la abstracción :class:`~src.models.base.PoseModel`.

Uso:
    python -m src.training.train_baseline --config configs/baseline_yolov8n.yaml

Por defecto la fase **falla duro** si falta el dataset o si la exportación no
produce el artefacto (ver :mod:`src.utils.strict`). Para correr el esqueleto
end-to-end sin datos hay que declararlo con ``smoke_test: true`` en el YAML.
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
from src.training import _kfold as kfold
from src.utils import mlflow_utils, strict

# Marca de run NO válido: se loguea únicamente cuando la fase corre como smoke
# test. Antes se logueaba siempre (para que un run sin dataset "tuviera" una
# métrica), lo que hacía indistinguible un run vacío de uno real; ahora su sola
# presencia en MLflow significa "las métricas de este run no sirven".
_SMOKE_TEST_METRIC = "pipeline_smoke_test"


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con el atributo ``config`` (ruta al YAML de la fase).
    """
    parser = argparse.ArgumentParser(description="Runner Fase 1 (baseline) del pipeline.")
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

    Los valores anidados (p. ej. ``export``, ``mlflow``) se serializan con
    claves separadas por punto, ya que MLflow espera parámetros escalares.

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
    """Trabajo de la Fase 1 con un run MLflow ya activo (un fold o el run único).

    Loguea hiperparámetros, construye el modelo, entrena, evalúa, loguea métricas
    y exporta a TFLite (Float32 por defecto). Lo llama :func:`run_baseline` a
    través de :mod:`src.training._kfold`, una vez por fold del GroupKFold.

    Args:
        config: configuración de la fase/modelo (con ``data`` ya apuntando al fold).

    Returns:
        Las :class:`ModelMetrics` obtenidas en la evaluación.
    """
    strict.set_smoke_mode(bool(config.get("smoke_test", False)))
    export_cfg: dict[str, Any] = config.get("export", {})

    mlflow_utils.log_params(_flatten_params(config))

    model = build_model(config)

    model.train()
    metrics = model.evaluate()
    mlflow_utils.log_metrics(metrics)

    if strict.is_smoke():
        mlflow.log_metric(_SMOKE_TEST_METRIC, 1.0)

    # Exportar artefacto TFLite y subirlo (Float32 por defecto en Fase 1).
    if export_cfg.get("format") == "tflite":
        with tempfile.TemporaryDirectory() as tmp:
            try:
                artifact = model.export_tflite(
                    output_dir=Path(tmp),
                    quantize=bool(export_cfg.get("quantize", False)),
                )
                mlflow_utils.log_model_artifact(artifact)
            except Exception as exc:  # noqa: BLE001
                # El artefacto exportado ES el entregable de la fase: un fallo de
                # export invalida el run. Sólo el esqueleto (smoke) lo degrada a nota.
                if not strict.is_smoke():
                    raise
                note = Path(tmp) / "export_skipped.txt"
                note.write_text(f"Export TFLite omitida: {exc}\n", encoding="utf-8")
                mlflow_utils.log_model_artifact(note, artifact_path="notes")

    return metrics


def run_baseline(config: dict[str, Any]) -> ModelMetrics:
    """Ejecuta la Fase 1, con GroupKFold por sujeto si el dataset tiene folds.

    Delega el ciclo de folds y el run padre a :func:`src.training._kfold.run_phase`
    (con folds → promedia media ± desvío entre folds; sin folds → un único run,
    preservando el smoke test).

    Args:
        config: configuración de la fase/modelo.

    Returns:
        Las :class:`ModelMetrics` (del run único o promediadas entre folds).
    """
    mlflow_cfg: dict[str, Any] = config.get("mlflow", {})
    experiment: str = mlflow_cfg.get("experiment", "fase1-baseline")
    return kfold.run_phase(
        config,
        _run_once,
        experiment=experiment,
        tags={"fase": "1", "familia": config["family"]},
    )


def main() -> None:
    """Punto de entrada CLI."""
    args = parse_args()
    config = load_config(args.config)
    metrics = run_baseline(config)
    print(f"Fase 1 completada. Métricas: {metrics.to_loggable()}")


if __name__ == "__main__":
    main()
