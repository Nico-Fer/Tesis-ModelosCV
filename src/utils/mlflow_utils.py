"""Helpers de MLflow reutilizables por todas las fases del pipeline.

Centraliza el boilerplate de tracking para que cada runner de fase
(``train_baseline``, ``prune``, ``quantize``, ``distill``, ``combined``)
loguee parámetros, métricas y artefactos de forma consistente y comparable.

El tracking URI se toma de la variable de entorno ``MLFLOW_TRACKING_URI`` si
está definida (p. ej. el servidor local o el MLflow gestionado de DagsHub);
en caso contrario MLflow escribe en ``./mlruns``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow

from src.models.base import ModelMetrics


@contextmanager
def start_run(
    experiment: str,
    run_name: str,
    tags: dict[str, str] | None = None,
    run_id: str | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Context manager que abre un run de MLflow ya configurado.

    Args:
        experiment: nombre del experimento (p. ej. ``"fase1-baseline"``).
        run_name: nombre legible del run (p. ej. el nombre del modelo).
        tags: tags opcionales del run (p. ej. ``{"fase": "1", "familia": "yolo"}``).
        run_id: si se pasa, **reabre** ese run en vez de crear uno nuevo (reanudación
            tras un corte, ver :mod:`src.utils.checkpoints`); ``run_name`` se ignora.

    Yields:
        El :class:`mlflow.ActiveRun` activo.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_id=run_id, run_name=run_name, tags=tags) as run:
        yield run


def log_params(params: dict[str, Any]) -> None:
    """Loguea hiperparámetros en el run activo.

    Args:
        params: diccionario de hiperparámetros.
    """
    mlflow.log_params(params)


def log_metrics(metrics: ModelMetrics) -> None:
    """Loguea las métricas comparables (solo las presentes) en el run activo.

    Args:
        metrics: instancia de :class:`ModelMetrics`.
    """
    loggable = metrics.to_loggable()
    if loggable:
        mlflow.log_metrics(loggable)


def log_model_artifact(path: Path, artifact_path: str = "model") -> None:
    """Sube un artefacto de modelo (o cualquier archivo) al run activo.

    Args:
        path: ruta local al artefacto (p. ej. el ``.tflite`` exportado).
        artifact_path: subcarpeta dentro del run donde guardarlo.
    """
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
