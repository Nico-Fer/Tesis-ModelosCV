"""Implementación de :class:`PoseModel` para BlazePose (Google, MediaPipe).

BlazePose se consume vía el **MediaPipe Python SDK** (venv-tf). No se descarga
un ``.tflite`` manualmente: MediaPipe gestiona el modelo internamente. Variantes
por complejidad:

    * ``lite``  → ``model_complexity=0`` (más liviano).
    * ``full``  → ``model_complexity=1`` (mayor precisión).
    * ``heavy`` → ``model_complexity=2`` (máxima precisión, si disponible).

Entorno requerido: **venv-tf** (``mediapipe``). El import del SDK es **diferido
dentro de los métodos** para que el módulo sea importable en venv-torch sin
MediaPipe instalado.

Formato de salida del SDK (relevante para :mod:`src.inference`):
    ``mp.solutions.pose.Pose.process()`` devuelve ``results.pose_landmarks``,
    un ``NormalizedLandmarkList`` de **33 landmarks** (esqueleto completo
    BlazePose) con ``.x`` / ``.y`` normalizados 0.0–1.0 y ``.visibility``.
    Como BlazePose usa 33 keypoints vs los 17 de COCO, el normalizador de
    inferencia mapea los 33 al subconjunto COCO con :data:`BLAZEPOSE_TO_COCO`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.models.base import ModelMetrics, PoseModel

# Variantes soportadas y su mapeo a model_complexity de MediaPipe.
BlazePoseVariant = Literal["lite", "full", "heavy"]
_COMPLEXITY: dict[str, int] = {"lite": 0, "full": 1, "heavy": 2}

# Mapeo oficial COCO(17) → BlazePose(33): para cada índice COCO (posición en la
# lista), el índice del landmark BlazePose correspondiente. Fuente de verdad
# única; el normalizador de inferencia lo reutiliza desde aquí.
#   COCO idx -> BlazePose idx
#   0  nose           -> 0
#   1  left eye       -> 2
#   2  right eye      -> 5
#   3  left ear       -> 7
#   4  right ear      -> 8
#   5  left shoulder  -> 11
#   6  right shoulder -> 12
#   7  left elbow     -> 13
#   8  right elbow    -> 14
#   9  left wrist     -> 15
#   10 right wrist    -> 16
#   11 left hip       -> 23
#   12 right hip      -> 24
#   13 left knee      -> 25
#   14 right knee     -> 26
#   15 left ankle     -> 27
#   16 right ankle    -> 28
BLAZEPOSE_TO_COCO: list[int] = [
    0,
    2,
    5,
    7,
    8,
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
    25,
    26,
    27,
    28,
]


class BlazePoseModel(PoseModel):
    """Familia BlazePose (``lite`` / ``full`` / ``heavy``) vía MediaPipe SDK.

    Claves de ``config`` esperadas:
        family (str): ``"blazepose"``.
        variant (str): ``"lite"``, ``"full"`` o ``"heavy"`` (default ``"full"``).
        min_detection_confidence (float, opcional): umbral de detección (0.5).
        min_tracking_confidence (float, opcional): umbral de tracking (0.5).
    """

    @property
    def variant(self) -> BlazePoseVariant:
        """Variante configurada (``full`` por defecto)."""
        variant = str(self.config.get("variant", "full"))
        if variant not in _COMPLEXITY:
            raise ValueError(
                f"Variante BlazePose no soportada: {variant!r}. "
                f"Disponibles: {', '.join(sorted(_COMPLEXITY))}."
            )
        return variant  # type: ignore[return-value]

    def load(self) -> None:
        """Instancia ``mp.solutions.pose.Pose`` con la complejidad de la variante.

        Idempotente: no recrea la instancia si ``self._model`` ya está poblado.
        ``static_image_mode=False`` habilita el tracking entre frames de video.
        """
        if self._model is not None:
            return

        # Import diferido: sólo disponible en venv-tf.
        import mediapipe as mp  # noqa: PLC0415

        self._model = mp.solutions.pose.Pose(
            model_complexity=_COMPLEXITY[self.variant],
            static_image_mode=False,
            min_detection_confidence=float(self.config.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(self.config.get("min_tracking_confidence", 0.5)),
        )

    def train(self) -> None:
        """No-op: BlazePose es un modelo pre-entrenado de Google (MediaPipe).

        No se fine-tunea en este pipeline; se consume vía el SDK tal cual.
        Se mantiene el método por contrato de :class:`PoseModel`.
        """
        return

    def evaluate(self) -> ModelMetrics:
        """Evalúa BlazePose y devuelve las métricas disponibles.

        Sin ``config["data"]`` se devuelve un :class:`ModelMetrics` vacío: el SDK
        de MediaPipe no expone el ``.tflite`` subyacente, por lo que ``size_mb``
        no es medible por esta vía. Con dataset, la evaluación real requiere
        correr inferencia sobre el set de validación y compararla con ground
        truth, lo cual depende de :mod:`src.data` (pendiente).

        Returns:
            :class:`ModelMetrics` (vacío en el path sin dataset).
        """
        data = self.config.get("data")
        if not data:
            return ModelMetrics()

        # TODO(fase-eval): correr inferencia sobre el set de validación, mapear
        # los 33 landmarks a COCO-17 (BLAZEPOSE_TO_COCO) y comparar con
        # gt_keypoints vía metric_utils.compute_oks / compute_pck.
        # Depende de src/data/ (carga y parseo del dataset).
        return ModelMetrics()

    def export_tflite(self, output_dir: Path, quantize: bool) -> Path:
        """No soportado: MediaPipe no expone el TFLite subyacente.

        Args:
            output_dir: ignorado.
            quantize: ignorado.

        Raises:
            NotImplementedError: siempre.
        """
        raise NotImplementedError(
            "BlazePose/MediaPipe SDK no expone el modelo TFLite para "
            "re-exportación. Para deployment Android usar el modelo oficial de "
            "MediaPipe Tasks."
        )
