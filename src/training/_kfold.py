"""Wrapper de GroupKFold (validación cruzada por sujeto) para los runners de fase.

El converter (:mod:`src.data.uco_to_yolo`) genera un dataset YAML por fold
(``pose_dataset_fold0.yaml`` … ``pose_dataset_fold{k-1}.yaml``, split por sujeto
sin fuga). Estos helpers descubren esos folds a partir del ``data`` de la config
y ejecutan la fase **una vez por fold**, agregando las métricas (media ± desvío)
en un run MLflow **padre**, con un run **anidado** por fold.

Cada fase provee una función ``run_once(config) -> ModelMetrics`` que hace su
trabajo (train/eval/export) **asumiendo que ya hay un run MLflow activo** y
loguea sus propias métricas; este módulo se encarga del ciclo de folds y del run
padre. Sin folds (dataset de fold único o smoke test) ejecuta ``run_once`` una
sola vez en un run normal, preservando el comportamiento previo.

Con ``cross_validation: false`` en la config, la fase corre sobre **un solo
fold** (``fold: k``, por defecto 0) en un run plano: sirve para producir el
modelo desplegable con un único entrenamiento, sin pagar el ~k× de la CV. Ese
run **no** emite métricas agregadas (``*_mean`` / ``*_std``) justamente para que
no se confunda con un resultado de validación cruzada.
"""

from __future__ import annotations

import copy
import logging
import statistics
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

import mlflow

from src.models.base import ModelMetrics
from src.utils import checkpoints, mlflow_utils, strict

logger = logging.getLogger(__name__)

RunOnce = Callable[[dict[str, Any]], ModelMetrics]


def discover_fold_yamls(data: str | Path) -> list[Path]:
    """Devuelve los YAML de fold hermanos de ``data`` (``*_fold{k}.yaml``), ordenados.

    Dado ``data/processed/pose_dataset.yaml`` busca en la misma carpeta
    ``pose_dataset_fold0.yaml``, ``…_fold1.yaml``, …  No requiere que el propio
    ``data`` exista (el converter genera sólo los YAML de fold).

    Args:
        data: ruta base del dataset YAML declarada en la config.

    Returns:
        Lista ordenada de rutas de fold; ``[]`` si no hay ninguna (fold único / smoke).
    """
    base = Path(data)
    return sorted(base.parent.glob(f"{base.stem}_fold*{base.suffix}"))


def _select_fold(folds: list[Path], data: str | Path, k: int) -> Path:
    """Devuelve el YAML del fold ``k`` entre los descubiertos, resuelto **por nombre**.

    Resuelve ``{stem}_fold{k}{suffix}`` en vez de indexar ``folds[k]``: el orden
    de :func:`discover_fold_yamls` es lexicográfico, así que con 10 o más folds
    ``fold10`` se ordena antes que ``fold2`` y el índice dejaría de coincidir con
    el número de fold. Pedir el fold 2 y entrenar sobre otro split es un error
    silencioso, no un fallo visible.

    Args:
        folds: YAML de fold descubiertos (no vacío).
        data: ruta base del dataset YAML declarada en la config.
        k: número de fold pedido.

    Returns:
        La ruta del fold ``k``.

    Raises:
        src.utils.strict.PipelineError: si no hay un fold ``k`` (salvo en smoke,
            donde degrada al primer fold disponible).
    """
    base = Path(data)
    wanted = base.with_name(f"{base.stem}_fold{k}{base.suffix}")
    for fold_yaml in folds:
        if fold_yaml.name == wanted.name:
            return fold_yaml
    disponibles = ", ".join(f.name for f in folds)
    strict.fail(
        f"No existe el fold {k} para {str(data)!r} (se buscó {wanted.name!r}). "
        f"Folds disponibles: {disponibles}."
    )
    return folds[0]


def _run_single_fold(
    config: dict[str, Any],
    run_once: RunOnce,
    folds: list[Path],
    *,
    experiment: str,
    tags: dict[str, str],
    run_name: str,
) -> ModelMetrics:
    """Ejecuta la fase sobre **un solo fold**, en un run MLflow plano.

    Es el camino de ``cross_validation: false``: entrena una vez en vez de k,
    para obtener el modelo desplegable sin el costo de la validación cruzada.

    A diferencia de la CV, **no** abre un run padre con runs anidados ni loguea
    ``{métrica}_mean`` / ``_std`` / ``_n_folds``. Una media y un desvío sobre un
    único split serían indistinguibles en la tabla de MLflow de un resultado de
    CV sano (el desvío daría 0.0), y esa confusión es exactamente lo que el
    conteo por métrica de :func:`_log_aggregate` existe para evitar. El tag
    ``cv=single-fold`` distingue estos runs de los ``cv=groupkfold``.

    Args:
        config: configuración de la fase.
        run_once: trabajo de la fase, asumiendo un run MLflow ya activo.
        folds: YAML de fold descubiertos (no vacío).
        experiment: nombre del experimento MLflow.
        tags: tags base del run (``fase``, ``familia``).
        run_name: nombre base del run (el modelo).

    Returns:
        Las métricas del único fold ejecutado.
    """
    k = int(config.get("fold", 0))
    fold_yaml = _select_fold(folds, str(config["data"]), k)
    logger.info(
        "cross_validation=false: fold único %s (%d folds disponibles).", fold_yaml.name, len(folds)
    )

    fold_config = copy.deepcopy(config)
    fold_config["data"] = str(fold_yaml)
    fold_config["fold"] = k

    single_tags = {**tags, "cv": "single-fold", "fold": str(k)}
    with mlflow_utils.start_run(
        experiment=experiment, run_name=f"{run_name}-fold{k}", tags=single_tags
    ):
        mlflow.log_param("n_folds", 1)
        metrics = run_once(fold_config)
        if not metrics.to_loggable():
            strict.fail(
                f"El fold {k} ({fold_yaml.name}) no produjo ninguna métrica: "
                "el run no tiene resultado que validar."
            )
        return metrics


def _aggregate(per_fold: list[ModelMetrics]) -> ModelMetrics:
    """Promedia (media) cada métrica presente a lo largo de los folds."""
    agg = ModelMetrics()
    for f in fields(ModelMetrics):
        vals = [getattr(m, f.name) for m in per_fold if getattr(m, f.name) is not None]
        if vals:
            setattr(agg, f.name, statistics.fmean(vals))
    return agg


def _log_aggregate(per_fold: list[ModelMetrics]) -> None:
    """Loguea media, desvío y **nº de folds que aportaron** cada métrica.

    El ``_n_folds`` por métrica no es decorativo: la media se computa sólo sobre
    los valores no-``None``, así que si 4 de 5 folds no produjeron OKS, el
    ``oks_mean`` es el de un único fold y el ``oks_std`` da 0.0 — indistinguible
    de un resultado sano y consistente. Logueando el conteo, un ``oks_n_folds``
    menor a ``n_folds`` delata el problema en la propia tabla de MLflow.
    """
    summary: dict[str, float] = {}
    for f in fields(ModelMetrics):
        vals = [getattr(m, f.name) for m in per_fold if getattr(m, f.name) is not None]
        if vals:
            summary[f"{f.name}_mean"] = statistics.fmean(vals)
            summary[f"{f.name}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            summary[f"{f.name}_n_folds"] = float(len(vals))
    if summary:
        mlflow.log_metrics(summary)


def _log_fold_runs_aggregate(run_ids: list[str]) -> None:
    """Media, desvío y nº de folds de **cada** métrica que loguearon los runs de fold.

    :func:`_log_aggregate` sólo cubre las :class:`ModelMetrics` que devuelve
    ``run_once`` (en la Fase 5, las de la etapa final). Las métricas por etapa
    (``oks_base``, ``oks_pruned``, …, ``latency_ms_int8``) son justamente la tabla
    comparativa de la tesis y quedaban sólo en los runs anidados. Se leen de MLflow
    (y no de memoria) para incluir los folds terminados antes de una reanudación.
    """
    already = {f.name for f in fields(ModelMetrics)}
    client = mlflow.tracking.MlflowClient()
    per_key: dict[str, list[float]] = {}
    for run_id in run_ids:
        for key, value in client.get_run(run_id).data.metrics.items():
            if key not in already:
                per_key.setdefault(key, []).append(float(value))
    summary: dict[str, float] = {}
    for key, vals in per_key.items():
        summary[f"{key}_mean"] = statistics.fmean(vals)
        summary[f"{key}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary[f"{key}_n_folds"] = float(len(vals))
    if summary:
        mlflow.log_metrics(summary)


def run_phase(
    config: dict[str, Any],
    run_once: RunOnce,
    *,
    experiment: str,
    tags: dict[str, str],
) -> ModelMetrics:
    """Ejecuta una fase con GroupKFold si hay folds; si no, aborta.

    - Con K folds: abre un run **padre** y, por cada fold, un run **anidado** que
      corre ``run_once`` sobre ``pose_dataset_fold{k}.yaml``; agrega media ±
      desvío de cada métrica en el padre y las devuelve.
    - Con ``config["cross_validation"] = False``: corre **un solo** fold
      (``config["fold"]``, por defecto 0) en un run plano, sin métricas
      agregadas — ver :func:`_run_single_fold`. Los folds igual tienen que
      existir: el chequeo de abajo es lo que delata un ``data`` mal escrito.
    - Sin ``data`` o sin YAML de folds: aborta con
      :class:`~src.utils.strict.PipelineError`, salvo en smoke test declarado
      (donde ejecuta ``run_once`` una vez, preservando el esqueleto end-to-end).

    Declara además el modo estricto del proceso a partir de
    ``config["smoke_test"]``, para toda la fase.

    Args:
        config: configuración de la fase.
        run_once: trabajo de la fase, asumiendo un run MLflow ya activo.
        experiment: nombre del experimento MLflow.
        tags: tags base del run (``fase``, ``familia``).

    Returns:
        Métricas del único run, o el promedio entre folds.
    """
    strict.set_smoke_mode(bool(config.get("smoke_test", False)))

    run_name = str(config.get("model", "model"))
    data = config.get("data")
    folds = discover_fold_yamls(data) if data else []

    if not folds:
        # Sin folds el run corre UNA vez en vez de k. Con un dataset real eso casi
        # siempre significa un typo en 'data' (el glob no matcheó), no un dataset
        # de fold único: degradar 5 folds a 1 en silencio invalidaría la media ± σ.
        strict.fail(
            f"No se encontraron YAML de fold hermanos de {data!r} "
            f"(patrón '{Path(str(data)).stem}_fold*{Path(str(data)).suffix}'). "
            "Verificá la ruta en la config o corré el converter."
        )
        with mlflow_utils.start_run(experiment=experiment, run_name=run_name, tags=tags):
            return run_once(config)

    # Después del guard de arriba a propósito: desactivar la CV no debe volverse
    # también una forma de saltear la validación de 'data'.
    if not bool(config.get("cross_validation", True)):
        return _run_single_fold(
            config, run_once, folds, experiment=experiment, tags=tags, run_name=run_name
        )

    logger.info("GroupKFold: %d folds detectados para %s.", len(folds), run_name)
    # Checkpoints reanudables (opt-in con ``checkpoint_dir``): un fold terminado no se
    # repite y los runs MLflow (padre y anidados) se REABREN por run_id, de modo que una
    # CV interrumpida termina en el mismo run padre, con su media ± σ sobre los k folds.
    cv_ckpt = checkpoints.CvCheckpoint.open(config) if config.get("checkpoint_dir") else None
    parent_tags = {**tags, "cv": "groupkfold", "n_folds": str(len(folds))}
    with mlflow_utils.start_run(
        experiment=experiment,
        run_name=run_name,
        tags=parent_tags,
        run_id=cv_ckpt.parent_run_id if cv_ckpt is not None else None,
    ) as parent:
        if cv_ckpt is not None:
            cv_ckpt.set_parent_run(parent.info.run_id)
        mlflow.log_param("n_folds", len(folds))
        per_fold: list[ModelMetrics] = []
        fold_run_ids: list[str] = []
        for k, fold_yaml in enumerate(folds):
            if cv_ckpt is not None and cv_ckpt.fold_done(k):
                logger.info("Fold %d ya terminado (checkpoint): se reusan sus métricas.", k)
                per_fold.append(cv_ckpt.fold_metrics(k))
                fold_run_ids.append(str(cv_ckpt.fold_run_id(k)))
                continue
            fold_config = copy.deepcopy(config)
            fold_config["data"] = str(fold_yaml)
            fold_config["fold"] = k
            if cv_ckpt is not None:
                fold_config["checkpoint_dir"] = str(cv_ckpt.fold_dir(k))
            with mlflow.start_run(
                run_id=cv_ckpt.fold_run_id(k) if cv_ckpt is not None else None,
                run_name=f"{run_name}-fold{k}",
                nested=True,
                tags={"fold": str(k)},
            ) as fold_run:
                if cv_ckpt is not None:
                    cv_ckpt.set_fold_run(k, fold_run.info.run_id, str(fold_yaml))
                fold_metrics = run_once(fold_config)
                if not fold_metrics.to_loggable():
                    strict.fail(
                        f"El fold {k} ({fold_yaml.name}) no produjo ninguna métrica: "
                        "agregarlo daría una media sobre menos folds de los declarados."
                    )
                per_fold.append(fold_metrics)
                fold_run_ids.append(fold_run.info.run_id)
                if cv_ckpt is not None:
                    cv_ckpt.mark_fold_done(k, fold_metrics)
        _log_aggregate(per_fold)
        _log_fold_runs_aggregate(fold_run_ids)
        if cv_ckpt is not None:
            cv_ckpt.mark_finished()
        return _aggregate(per_fold)
