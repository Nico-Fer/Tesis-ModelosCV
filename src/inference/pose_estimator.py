"""Inferencia unificada de pose para cualquier familia del pipeline.

:class:`PoseEstimator` envuelve un :class:`~src.models.base.PoseModel` ya
construido y expone un único método :meth:`PoseEstimator.predict` que toma un
frame BGR de OpenCV y devuelve detecciones en formato COCO JSON, delegando la
normalización a :class:`~src.inference.normalizer.KeypointNormalizer`.

Sólo depende de ``cv2`` y ``numpy`` (ambos en requirements-base); los imports de
los frameworks de cada familia ocurren de forma diferida al llamar
``model.load()`` dentro del wrapper concreto.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Literal

import cv2
import numpy as np

from src.inference.normalizer import KeypointNormalizer
from src.models.base import PoseModel

Source = Literal["yolo", "movenet", "blazepose"]


class PoseEstimator:
    """Adaptador de inferencia común sobre un :class:`PoseModel`.

    Úsese como context manager para garantizar que el modelo se cargue al entrar
    y se liberen los recursos al salir::

        with PoseEstimator(model, source="movenet") as estimator:
            detections = estimator.predict(frame)

    Args:
        model: instancia concreta de :class:`PoseModel` (YOLO, MoveNet o
            BlazePose) ya construida vía el registry.
        source: familia del modelo; determina cómo se ejecuta la inferencia y
            cómo se normaliza la salida.
    """

    def __init__(self, model: PoseModel, source: Source) -> None:
        self.model = model
        self.source = source

    def __enter__(self) -> PoseEstimator:
        """Carga el modelo subyacente y devuelve el estimador listo."""
        self.model.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Libera recursos del modelo (best-effort; p. ej. ``Pose.close()``)."""
        underlying = self.model._model
        close = getattr(underlying, "close", None)
        if callable(close):
            close()

    def predict(self, frame: np.ndarray, image_id: int = 0) -> list[dict]:
        """Corre inferencia sobre un frame y devuelve detecciones COCO JSON.

        Args:
            frame: frame BGR de OpenCV, shape ``(H, W, 3)``.
            image_id: identificador del frame para el JSON resultante.

        Returns:
            Lista de detecciones COCO (una por persona; ver
            :meth:`KeypointNormalizer.normalize`).
        """
        frame_h, frame_w = frame.shape[:2]
        backend: Any = self.model._model  # intérprete / YOLO / Pose ya cargado

        if self.source == "yolo":
            raw_output = backend.predict(frame, verbose=False)
        elif self.source == "movenet":
            raw_output = self._run_movenet(frame)
        elif self.source == "blazepose":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_output = backend.process(rgb)
        else:  # pragma: no cover - defensivo; el constructor ya restringe Source.
            raise ValueError(f"Fuente no soportada: {self.source!r}.")

        return KeypointNormalizer.normalize(
            raw_output,
            source=self.source,
            frame_w=frame_w,
            frame_h=frame_h,
            image_id=image_id,
        )

    def _run_movenet(self, frame: np.ndarray) -> np.ndarray:
        """Ejecuta el intérprete TFLite de MoveNet sobre un frame.

        Redimensiona al tamaño de la variante, convierte BGR→RGB y respeta el
        dtype de entrada esperado por el intérprete (``uint8`` para builds int8,
        ``float32`` para builds float). MoveNet espera valores en rango
        ``[0, 255]``, por lo que **no** se normaliza a ``[0, 1]``.

        Args:
            frame: frame BGR de OpenCV, shape ``(H, W, 3)``.

        Returns:
            Tensor de salida de MoveNet, shape ``(1, 1, 17, 3)`` ``(y, x, conf)``.
        """
        interpreter: Any = self.model._model
        size = int(getattr(self.model, "input_size"))

        resized = cv2.resize(frame, (size, size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        in_dtype = input_details[0]["dtype"]

        if np.issubdtype(in_dtype, np.integer):
            tensor = rgb.astype(in_dtype)
        else:
            tensor = rgb.astype(np.float32)
        tensor = np.expand_dims(tensor, axis=0)

        interpreter.set_tensor(input_details[0]["index"], tensor)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]["index"])
