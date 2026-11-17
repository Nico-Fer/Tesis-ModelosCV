"""Pipeline de inferencia: unifica la salida de las familias a COCO Keypoints JSON.

Expone los tres puntos de entrada del paquete:

    * :class:`~src.inference.pose_estimator.PoseEstimator` — adaptador de
      inferencia común sobre cualquier :class:`~src.models.base.PoseModel`.
    * :func:`~src.inference.video_pipeline.run_inference_on_video` — corre la
      inferencia sobre un video/cámara y vuelca las detecciones a JSON.
    * :class:`~src.inference.normalizer.KeypointNormalizer` — normaliza salidas
      crudas (YOLO / MoveNet / BlazePose) a COCO JSON.
"""

from src.inference.normalizer import KeypointNormalizer
from src.inference.pose_estimator import PoseEstimator
from src.inference.video_pipeline import run_inference_on_video

__all__ = ["PoseEstimator", "run_inference_on_video", "KeypointNormalizer"]
