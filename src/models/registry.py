"""Factory de modelos: mapea la familia declarada en la config a su clase.

Punto único de extensión para sumar familias al pipeline. El dispatch usa
**imports diferidos por familia**: cada wrapper concreto importa su framework
pesado (Ultralytics, TensorFlow, MediaPipe), que vive en un venv distinto, de
modo que construir una familia no obliga a tener instaladas las dependencias de
las otras. Así ``build_model`` es importable tanto en venv-torch como en venv-tf.
"""

from __future__ import annotations

from typing import Any

from src.models.base import PoseModel

# Familias soportadas (para el mensaje de error). Las clases se importan de
# forma diferida dentro de cada rama de build_model.
_SUPPORTED = ("yolo", "movenet", "blazepose")


def build_model(config: dict[str, Any]) -> PoseModel:
    """Construye el :class:`PoseModel` correspondiente a ``config["family"]``.

    El import de la clase concreta se hace dentro de la rama de la familia para
    no arrastrar dependencias de frameworks que viven en otro venv.

    Args:
        config: configuración de la fase/modelo (cargada desde un YAML).

    Returns:
        Instancia concreta de :class:`PoseModel`.

    Raises:
        KeyError: si falta la clave ``family``.
        ValueError: si la familia no está registrada.
    """
    family = config["family"]

    if family == "yolo":
        from src.models.yolo import YoloPoseModel  # noqa: PLC0415

        return YoloPoseModel(config)
    if family == "movenet":
        from src.models.movenet import MoveNetModel  # noqa: PLC0415

        return MoveNetModel(config)
    if family == "blazepose":
        from src.models.blazepose import BlazePoseModel  # noqa: PLC0415

        return BlazePoseModel(config)

    disponibles = ", ".join(_SUPPORTED)
    raise ValueError(f"Familia de modelo no soportada: {family!r}. Disponibles: {disponibles}.")
