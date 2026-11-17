"""Implementación de :class:`PoseModel` para MoveNet (Google, TF Hub).

MoveNet **no se entrena** en este pipeline: es un modelo pre-entrenado de
Google que se descarga directamente en formato TFLite y se evalúa / usa para
inferencia. Cubre dos variantes single-pose:

    * ``lightning`` → más liviano, mayor FPS, entrada 192×192.
    * ``thunder``   → mayor precisión, mayor latencia, entrada 256×256.

Entorno requerido: **venv-tf** (TensorFlow / ``tf.lite``). Igual que el resto
de las familias, los imports del framework son **diferidos dentro de los
métodos** para que el módulo sea importable en venv-torch sin TF.

Formato de salida del intérprete (relevante para :mod:`src.inference`):
    Tensor de salida shape ``(1, 1, 17, 3)`` = ``[batch, person, keypoint,
    (y, x, conf)]``. **MoveNet devuelve Y antes que X**; el normalizador de
    inferencia invierte el orden a ``(x, y)`` para COCO JSON. Coordenadas
    normalizadas 0.0–1.0 relativas al frame de entrada.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

from src.evaluation import metrics as metric_utils
from src.models.base import ModelMetrics, PoseModel

logger = logging.getLogger(__name__)

# Variantes single-pose soportadas.
MoveNetVariant = Literal["lightning", "thunder"]

# Tamaño de entrada (lado del cuadrado) esperado por cada variante.
_INPUT_SIZE: dict[str, int] = {"lightning": 192, "thunder": 256}

# URLs por defecto del .tflite por variante. ATENCIÓN: tfhub.dev migró a
# Kaggle Models, por lo que estas URLs pueden cambiar. Verificar / sobreescribir
# vía la clave de config ``model_url``.
_DEFAULT_MOVENET_URLS: dict[str, str] = {
    "lightning": (
        "https://tfhub.dev/google/lite-model/movenet/singlepose/"
        "lightning/tflite/float16/4?lite-format=tflite"
    ),
    "thunder": (
        "https://tfhub.dev/google/lite-model/movenet/singlepose/"
        "thunder/tflite/float16/4?lite-format=tflite"
    ),
}

# Directorio de caché por defecto para los .tflite descargados.
_DEFAULT_CACHE_DIR = Path("data/models")


class MoveNetModel(PoseModel):
    """Familia MoveNet single-pose (``lightning`` / ``thunder``) vía TFLite.

    Claves de ``config`` esperadas:
        family (str): ``"movenet"``.
        variant (str): ``"lightning"`` o ``"thunder"`` (default ``"lightning"``).
        data (str, opcional): ruta al dataset de validación (evaluación real).
        cache_dir (str, opcional): directorio de caché local de los ``.tflite``
            descargados (default ``data/models/``).
        model_url (str, opcional): URL directa del ``.tflite``; si se omite se
            usa :data:`_DEFAULT_MOVENET_URLS` para la variante.
    """

    @property
    def variant(self) -> MoveNetVariant:
        """Variante configurada (``lightning`` por defecto)."""
        variant = str(self.config.get("variant", "lightning"))
        if variant not in _INPUT_SIZE:
            raise ValueError(
                f"Variante MoveNet no soportada: {variant!r}. "
                f"Disponibles: {', '.join(sorted(_INPUT_SIZE))}."
            )
        return variant  # type: ignore[return-value]

    @property
    def input_size(self) -> int:
        """Lado del cuadrado de entrada esperado por la variante (192 o 256)."""
        return _INPUT_SIZE[self.variant]

    @property
    def cache_path(self) -> Path:
        """Ruta local del ``.tflite`` cacheado para la variante configurada."""
        cache_dir = Path(self.config.get("cache_dir", _DEFAULT_CACHE_DIR))
        return cache_dir / f"movenet_{self.variant}.tflite"

    def load(self) -> None:
        """Descarga (si hace falta) e instancia el intérprete TFLite de MoveNet.

        Descarga el ``.tflite`` desde ``config["model_url"]`` (o la URL por
        defecto de la variante) hacia :attr:`cache_path` si el archivo aún no
        existe, y luego crea un ``tf.lite.Interpreter`` con los tensores ya
        asignados. Idempotente: no redescarga si el archivo está en caché ni
        recrea el intérprete si ``self._model`` ya está poblado.
        """
        if self._model is not None:
            return

        # Imports diferidos: sólo disponibles en venv-tf.
        import tensorflow as tf  # noqa: PLC0415

        cache_path = self.cache_path
        if not cache_path.exists():
            self._download(cache_path)

        interpreter = tf.lite.Interpreter(model_path=str(cache_path))
        interpreter.allocate_tensors()
        self._model = interpreter

    def _download(self, dest: Path) -> None:
        """Descarga el ``.tflite`` de la variante a ``dest`` vía ``requests``.

        Args:
            dest: ruta destino del archivo en la caché local.
        """
        import requests  # noqa: PLC0415

        url = str(self.config.get("model_url", _DEFAULT_MOVENET_URLS[self.variant]))
        dest.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)

    def train(self) -> None:
        """No-op: MoveNet es un modelo pre-entrenado de Google.

        No se fine-tunea en este pipeline; se descarga y se evalúa tal cual.
        Se mantiene el método por contrato de :class:`PoseModel`.
        """
        return

    def evaluate(self) -> ModelMetrics:
        """Evalúa MoveNet y devuelve las métricas disponibles.

        Sin ``config["data"]`` sólo se reporta ``size_mb`` del ``.tflite``
        cacheado (si ya fue descargado). Con dataset, la evaluación real
        requiere correr inferencia sobre el set de validación y compararla con
        ground truth, lo cual depende de :mod:`src.data` (pendiente).

        Returns:
            :class:`ModelMetrics` con ``size_mb`` cuando el modelo está en disco.
        """
        cache_path = self.cache_path
        size_mb: float | None = None
        if cache_path.exists():
            size_mb = metric_utils.model_size_mb(cache_path)

        data = self.config.get("data")
        if not data:
            return ModelMetrics(size_mb=size_mb)

        # TODO(fase-eval): correr inferencia sobre el set de validación, extraer
        # pred_keypoints y compararlos con gt_keypoints vía metric_utils.compute_oks
        # / compute_pck. Depende de src/data/ (carga y parseo del dataset).
        return ModelMetrics(size_mb=size_mb)

    def export_tflite(self, output_dir: Path, quantize: bool) -> Path:
        """Exporta el modelo TFLite a ``output_dir``.

        MoveNet ya ES un artefacto TFLite (las variantes por defecto vienen
        cuantizadas a Float16), por lo que la exportación se reduce a copiar el
        ``.tflite`` cacheado al destino.

        Sobre ``quantize=True``: un ``.tflite`` ya compilado **no puede
        re-cuantizarse** con ``TFLiteConverter`` (no existe ``from_tflite_file``;
        el converter necesita el SavedModel original, que MoveNet no distribuye).
        Para INT8 full-integer real hay que descargar directamente la variante
        ``int8`` de MoveNet (sobreescribiendo ``config["model_url"]``). Aquí, con
        ``quantize=True`` se registra una advertencia y se copia el ``.tflite``
        cacheado (ya optimizado) tal cual.

        Args:
            output_dir: carpeta destino del artefacto.
            quantize: si ``True`` advierte y copia el cacheado (no re-cuantiza);
                si ``False`` copia el TFLite tal cual.

        Returns:
            :class:`~pathlib.Path` al ``.tflite`` generado en ``output_dir``.
        """
        self.load()
        output_dir.mkdir(parents=True, exist_ok=True)

        cache_path = self.cache_path
        dest = output_dir / f"movenet_{self.variant}.tflite"

        if quantize:
            logger.warning(
                "MoveNet se distribuye como .tflite ya compilado: no es posible "
                "re-cuantizar a INT8 con TFLiteConverter. Se copia el cacheado "
                "(Float16). Para INT8 real, configurá model_url a la variante int8."
            )

        shutil.copy2(cache_path, dest)
        return dest
