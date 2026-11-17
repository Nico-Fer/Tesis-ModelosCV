"""Modo estricto del pipeline: falla duro salvo que el run sea un smoke test.

Motivación
----------
El esqueleto del pipeline se escribió cuando todavía no había un dataset bien
anotado, así que cada etapa tenía un *escape*: si faltaba ``data`` no entrenaba,
si la exportación fallaba se registraba una nota en vez de abortar, y todos los
runners logueaban ``pipeline_smoke_test = 1.0`` pase lo que pase. El resultado es
que un run degradado (sin entrenar, sin métricas de keypoints, sin artefacto) y
uno sano se ven **idénticos** en MLflow.

Con dataset real eso deja de ser una comodidad y pasa a ser un riesgo de publicar
resultados inválidos. Este módulo invierte el default: los escapes lanzan
:class:`PipelineError` salvo que el run se haya declarado explícitamente como
smoke test (``smoke_test: true`` en el YAML de la fase).

Uso
---
Cada runner declara el modo una vez, al entrar a la fase y a cada fold
(``_kfold`` deep-copia la config, así que el valor se re-declara por fold)::

    strict.set_smoke_mode(bool(config.get("smoke_test", False)))

y los puntos de escape llaman a :func:`fail` (precondición incumplida) o
consultan :func:`is_smoke` para decidir si re-lanzar (bloque frágil que además
tiene una acción de fallback, como registrar una nota)::

    if not data:
        strict.fail("train() requiere 'data'; sin dataset no hay entrenamiento.")

    try:
        artifact = model.export_tflite(...)
    except Exception:
        if not strict.is_smoke():
            raise
        ...  # fallback sólo en smoke

El flag es de proceso (no se pasa por parámetro) a propósito: módulos como
:mod:`src.evaluation.pose_eval` no reciben la config de la fase y aun así deben
respetar el modo. El pipeline es monoproceso por fase, así que un global alcanza.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Default estricto: un módulo importado sin que ningún runner declare el modo
# (p. ej. desde un notebook o un test) falla duro, que es el lado seguro.
_smoke_mode = False


class PipelineError(RuntimeError):
    """Fase abortada porque una precondición del pipeline no se cumple.

    Se lanza en lugar de degradar silenciosamente el run (no entrenar, no
    evaluar, no exportar). Si el run es un smoke test declarado, no se lanza.
    """


def set_smoke_mode(enabled: bool) -> None:
    """Declara si el run en curso es un smoke test.

    Args:
        enabled: ``True`` convierte los :func:`fail` en warnings (el pipeline
            corre end-to-end sin dataset ni pesos reales); ``False`` (default)
            los convierte en :class:`PipelineError`.
    """
    global _smoke_mode
    _smoke_mode = bool(enabled)
    if _smoke_mode:
        logger.warning(
            "SMOKE TEST activo: los fallos del pipeline se degradan a warnings. "
            "Las métricas de este run NO son válidas."
        )


def is_smoke() -> bool:
    """``True`` si el run en curso está declarado como smoke test."""
    return _smoke_mode


def fail(message: str) -> None:
    """Aborta la fase, salvo en smoke test (donde sólo advierte).

    Args:
        message: qué precondición falló y por qué importa.

    Raises:
        PipelineError: siempre, excepto si :func:`is_smoke` es ``True``.
    """
    if _smoke_mode:
        logger.warning("[smoke] %s", message)
        return
    raise PipelineError(message)
