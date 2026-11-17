"""Runner de la FASE 2 (pruning estructurado + fine-tuning) del pipeline experimental.

Aplica pruning estructurado al modelo base, realiza fine-tuning para recuperar
precisión perdida, evalúa y exporta a TFLite Float32. La lógica de pruning se
delega a :mod:`src.optimization.pruning`.

Uso:
    python -m src.training.prune --config configs/prune_yolov8n.yaml

El esqueleto corre end-to-end sin dataset (smoke test): train/evaluate toleran
ausencia de datos igual que en :mod:`src.training.train_baseline`.
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
from src.optimization import pruning as pruning_utils
from src.training import _kfold as kfold
from src.utils import mlflow_utils, strict

_SMOKE_TEST_METRIC = "pipeline_smoke_test"


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con el atributo ``config`` (ruta al YAML de la fase).
    """
    parser = argparse.ArgumentParser(description="Runner Fase 2 (pruning) del pipeline.")
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

    Los valores anidados (p. ej. ``pruning``, ``mlflow``) se serializan con
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
    """Trabajo de la Fase 2 con un run MLflow ya activo (un fold o el run único).

    Loguea hiperparámetros, construye el modelo, aplica pruning estructurado +
    fine-tuning, evalúa, loguea métricas y exporta a TFLite Float32. Lo llama
    :func:`run_prune` vía :mod:`src.training._kfold`, una vez por fold.

    Args:
        config: configuración de la fase/modelo (con ``data`` ya apuntando al fold).

    Returns:
        Las :class:`~src.models.base.ModelMetrics` obtenidas post-pruning.
    """
    strict.set_smoke_mode(bool(config.get("smoke_test", False)))
    export_cfg: dict[str, Any] = config.get("export", {})

    mlflow_utils.log_params(_flatten_params(config))

    model = build_model(config)

    # Pruning estructurado (torch_pruning / DepGraph) + fine-tuning de recuperación.
    model = pruning_utils.apply_structured_pruning(model, config)
    pruning_utils.fine_tune_after_pruning(model, config)

    # Se persiste el podado+recuperado y se RECARGA en un wrapper limpio (idioma de
    # combined.py): reusar el YOLO recién entrenado revienta en NMS ('list' object
    # has no attribute 'shape'). Además fija el deliverable en un path estable
    # (``pruning.save_checkpoint``) en vez de runs/pose/pruned_finetune*/. En smoke
    # test se omite para no pisar el checkpoint de un run real.
    if config.get("family") == "yolo" and model._model is not None and not strict.is_smoke():
        ft_pt = Path(
            config.get("pruning", {}).get("save_checkpoint", "models/exports/student_pruned_ft.pt")
        )
        pruning_utils.save_pruned_checkpoint(model._model, ft_pt)
        model.config["model"] = str(ft_pt)
        model._model = None  # fuerza recarga limpia en evaluate()/export

    metrics = model.evaluate()
    mlflow_utils.log_metrics(metrics)

    if strict.is_smoke():
        mlflow.log_metric(_SMOKE_TEST_METRIC, 1.0)

    if export_cfg.get("format") == "tflite":
        with tempfile.TemporaryDirectory() as tmp:
            try:
                artifact = model.export_tflite(
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


def run_prune(config: dict[str, Any]) -> ModelMetrics:
    """Ejecuta la Fase 2, con GroupKFold por sujeto si el dataset tiene folds.

    Delega el ciclo de folds y el run padre a :func:`src.training._kfold.run_phase`.

    Args:
        config: configuración de la fase/modelo.

    Returns:
        Las :class:`~src.models.base.ModelMetrics` (del run único o promediadas).
    """
    mlflow_cfg: dict[str, Any] = config.get("mlflow", {})
    experiment: str = mlflow_cfg.get("experiment", "fase2-pruning")
    return kfold.run_phase(
        config,
        _run_once,
        experiment=experiment,
        tags={"fase": "2", "familia": config["family"]},
    )


def main() -> None:
    """Punto de entrada CLI."""
    args = parse_args()
    config = load_config(args.config)
    metrics = run_prune(config)
    print(f"Fase 2 completada. Métricas: {metrics.to_loggable()}")


if __name__ == "__main__":
    main()
