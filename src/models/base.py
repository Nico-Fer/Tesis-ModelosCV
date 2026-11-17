"""Abstracción base para modelos de pose estimation del pipeline.

Define el contrato común (`PoseModel`) que toda familia de modelos
(YOLOv8-pose, MoveNet, BlazePose, ...) debe implementar, y la estructura de
métricas comparables (`ModelMetrics`) que se loguea en MLflow en TODAS las
fases del pipeline (Fase 1 a Fase 5).

Diseñar contra esta interfaz permite que los runners de cada fase
(`train_baseline`, `prune`, `quantize`, `distill`, `combined`) sean
agnósticos a la familia de modelo concreta.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelMetrics:
    """Métricas comparables entre todos los runs del pipeline.

    Todos los campos son opcionales para tolerar fases o modelos que no
    computan una métrica determinada (p. ej. un baseline sin medición de
    latencia todavía). Solo se loguean en MLflow los campos no nulos.

    Attributes:
        map: mean Average Precision (detección de keypoints).
        oks: Object Keypoint Similarity promedio.
        pck: Percentage of Correct Keypoints.
        fps: cuadros por segundo en el hardware objetivo.
        latency_ms: latencia de inferencia por frame, en milisegundos.
        size_mb: peso del modelo exportado, en megabytes.
    """

    map: float | None = None
    oks: float | None = None
    pck: float | None = None
    fps: float | None = None
    latency_ms: float | None = None
    size_mb: float | None = None

    def to_loggable(self) -> dict[str, float]:
        """Devuelve solo las métricas presentes, listas para `mlflow.log_metrics`.

        Returns:
            Diccionario ``{nombre: valor}`` sin los campos en ``None``.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}


class PoseModel(ABC):
    """Contrato común para una familia de modelos de pose estimation.

    Cada implementación concreta (ver `src/models/yolo.py`) encapsula su
    framework (Ultralytics, TF, MediaPipe) detrás de estos métodos, de modo
    que el resto del pipeline no dependa de detalles del framework.

    Args:
        config: configuración del modelo/fase, típicamente cargada desde un
            YAML de ``configs/``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self._model: Any | None = None

    @property
    def name(self) -> str:
        """Nombre legible del modelo para nombrar runs y artefactos."""
        return str(self.config.get("model", self.__class__.__name__))

    @abstractmethod
    def load(self) -> None:
        """Carga o instancia el modelo subyacente en memoria.

        Debe poblar ``self._model``. Idempotente: invocarlo dos veces no debe
        recargar pesos innecesariamente.
        """

    @abstractmethod
    def train(self) -> None:
        """Entrena o hace fine-tuning según ``self.config``.

        En Fase 1 puede ser un entrenamiento desde el checkpoint base; en otras
        fases, el reentrenamiento posterior a pruning/distillation.
        """

    @abstractmethod
    def evaluate(self) -> ModelMetrics:
        """Evalúa el modelo y devuelve las métricas comparables.

        Returns:
            Instancia de :class:`ModelMetrics` con los campos que la familia
            sepa computar en esta fase.
        """

    @abstractmethod
    def export_tflite(self, output_dir: Path, quantize: bool) -> Path:
        """Exporta el modelo a TFLite.

        Args:
            output_dir: carpeta destino del artefacto exportado.
            quantize: si ``True``, aplica cuantización INT8 (Fases 3/5);
                si ``False``, exporta en Float32 (baseline Fase 1).

        Returns:
            Ruta al archivo ``.tflite`` generado.
        """
