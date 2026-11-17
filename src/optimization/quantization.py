"""Cuantización post-training FP32 → INT8 (Fase 3).

Expone dos niveles de API:

``quantize_tflite``
    Función orientada a archivos: cuantiza desde un **SavedModel** de TensorFlow.
    Soporta dos modos:

    * **Dynamic range** (``representative_dataset_fn=None``): cuantiza pesos a
      INT8 sin dataset de calibración. Activaciones en float en runtime. Modo de
      menor precisión pero el más simple.
    * **Full integer** (``representative_dataset_fn`` provisto): cuantiza pesos Y
      activaciones a INT8 con un generador de muestras representativas. Es el modo
      recomendado para deployment Android (NNAPI / Hexagon delegate).

``prepare_calibration_dataset`` / ``quantize_to_int8``
    Wrappers orientados a instancias de :class:`~src.models.base.PoseModel`. Los
    runners de Fase 3 y 5 los llaman sin cambios. ``quantize_to_int8`` delega en
    ``model.export_tflite(quantize=True)``, que es **agnóstico al modelo**: cada
    familia conoce su propio camino INT8 (YOLO → Ultralytics ``export(int8=True,
    data=...)``, que calibra con el dataset internamente; MoveNet → dynamic range).

Entorno requerido: **venv-torch** o **venv-tf** (TF 2.15 está en ambos).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from src.models.base import PoseModel
from src.utils import strict

logger = logging.getLogger(__name__)

# Mínimo recomendado de imágenes de calibración para full-integer INT8: por debajo
# la distribución de activaciones queda poco representada y el INT8 pierde precisión.
_MIN_CALIBRATION_IMAGES = 100

# Tope de imágenes de calibración INT8. El split ``val`` de un fold UCO tiene ~16k
# frames y Ultralytics los materializa de una vez al calibrar → OOM. ~300 imágenes
# representativas bastan para estimar el rango de activaciones.
CALIB_MAX_IMAGES = 300

# Extensiones de imagen reconocidas al contar/iterar el set de calibración.
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def quantize_tflite(
    saved_model_dir: str | Path,
    output_path: str | Path,
    representative_dataset_fn: Callable | None = None,
) -> Path:
    """Cuantiza un **SavedModel** de TensorFlow a TFLite INT8.

    Importante: el ``TFLiteConverter`` **no puede reconstruir** un modelo desde un
    flatbuffer ``.tflite`` ya compilado (no existe ``from_tflite_file``); requiere
    el SavedModel (o un grafo de funciones concretas). Por eso esta función toma un
    **directorio SavedModel** como origen. Para modelos distribuidos sólo como
    ``.tflite`` (p. ej. MoveNet) la única re-cuantización viable es dynamic-range
    aplicada en la exportación original, no aquí.

    Args:
        saved_model_dir: directorio del SavedModel de origen (FP32).
        output_path: path donde se escribe el ``.tflite`` cuantizado.
        representative_dataset_fn: callable sin argumentos que devuelve un generador
            de listas ``[np.ndarray]`` (una muestra por iteración). Si es ``None`` se
            aplica dynamic range quantization; si se provee, full integer INT8.

    Returns:
        :class:`~pathlib.Path` al ``.tflite`` generado.

    Raises:
        FileNotFoundError: si ``saved_model_dir`` no existe o no es un directorio.
    """
    import tensorflow as tf  # noqa: PLC0415

    saved_model_dir = Path(saved_model_dir)
    output_path = Path(output_path)

    if not saved_model_dir.is_dir():
        raise FileNotFoundError(
            f"quantize_tflite requiere un directorio SavedModel; no existe: {saved_model_dir}. "
            "Un .tflite ya compilado no se puede re-cuantizar con TFLiteConverter."
        )

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if representative_dataset_fn is not None:
        # Full integer quantization: activaciones también a INT8.
        converter.representative_dataset = representative_dataset_fn
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    # Sólo DEFAULT en optimizations ⇒ dynamic range quantization.

    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    return output_path


def build_representative_dataset_fn(
    images_dir: str | Path,
    imgsz: int = 640,
    num_samples: int = 200,
) -> Callable[[], Any]:
    """Construye un ``representative_dataset_fn`` desde un directorio de imágenes.

    Genera muestras ``(1, imgsz, imgsz, 3)`` float32 normalizadas a 0–1 (formato
    NHWC, el que espera ``TFLiteConverter``). Reutilizable por cualquier camino TF
    de full-integer quantization. Sólo depende de ``cv2`` + ``numpy``
    (requirements-base), así que importa en ambos venvs.

    Args:
        images_dir: directorio con imágenes de calibración (búsqueda recursiva).
        imgsz: lado del cuadrado al que se redimensiona cada imagen.
        num_samples: tope de imágenes a iterar.

    Returns:
        Callable sin argumentos apto para ``converter.representative_dataset``.

    Raises:
        FileNotFoundError: si no se encuentran imágenes en ``images_dir``.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    images_dir = Path(images_dir)
    paths = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)[
        :num_samples
    ]
    if not paths:
        raise FileNotFoundError(f"Sin imágenes de calibración en {images_dir}.")

    def _generator() -> Any:  # noqa: ANN401
        for path in paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (imgsz, imgsz))
            sample = (img.astype(np.float32) / 255.0)[None, ...]  # (1, H, W, 3)
            yield [sample]

    return _generator


def _count_calibration_images(data_path: Path) -> int | None:
    """Cuenta (best-effort) las imágenes del set de calibración.

    Acepta un directorio de imágenes o un YAML de dataset Ultralytics (resuelve
    ``path`` + ``train``). El ``train`` puede ser un directorio **o** un manifest
    ``.txt`` de rutas, una por línea — que es la forma que usan los folds de este
    proyecto (pool plano compartido, ver :mod:`src.data.uco_to_yolo`); sin
    contemplarlo, el conteo devolvía ``None`` y el umbral mínimo de calibración
    quedaba inerte justo en el caso real.

    Returns:
        Cantidad de imágenes, o ``None`` si no se puede determinar.
    """
    if data_path.is_dir():
        return sum(1 for p in data_path.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)

    if data_path.suffix.lower() in (".yaml", ".yml") and data_path.exists():
        try:
            import yaml  # noqa: PLC0415

            spec = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return None
        root = Path(spec.get("path", data_path.parent))
        if not root.is_absolute():
            root = (data_path.parent / root).resolve()
        train = Path(spec.get("train", "images/train"))
        if not train.is_absolute():
            train = root / train
        if train.is_dir():
            return sum(1 for p in train.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
        if train.suffix.lower() == ".txt" and train.is_file():
            return sum(
                1
                for ln in train.read_text(encoding="utf-8").splitlines()
                if ln.strip() and Path(ln.strip()).suffix.lower() in _IMAGE_EXTS
            )
    return None


def prepare_calibration_dataset(config: dict[str, Any]) -> Path | None:
    """Devuelve la ruta al dataset de calibración para cuantización INT8.

    Valida que el corpus tenga al menos :data:`_MIN_CALIBRATION_IMAGES` imágenes;
    por debajo de ese umbral la distribución de activaciones no es representativa
    y el INT8 resultante no es comparable con las otras fases, así que **aborta**
    (salvo smoke test declarado).

    Args:
        config: configuración de la fase.

    Returns:
        Ruta al dataset YAML / directorio de calibración, o ``None`` en smoke test.
    """
    data: str | None = config.get("data")
    if not data:
        strict.fail(
            "Cuantización INT8 sin 'data': sin set de calibración las activaciones no se "
            "escalan con datos reales y el modelo INT8 no es comparable."
        )
        return None

    data_path = Path(data)
    count = _count_calibration_images(data_path)
    if count is not None and count < _MIN_CALIBRATION_IMAGES:
        strict.fail(
            f"Set de calibración con {count} imágenes (< {_MIN_CALIBRATION_IMAGES} "
            "recomendadas): el INT8 full-integer pierde precisión por baja "
            "representatividad."
        )
    return data_path


def cap_calibration_fraction(config: dict[str, Any], n_val: int) -> float | None:
    """Acota, si hace falta, la fracción del split ``val`` usada para calibrar INT8.

    Escribe ``config["calib_fraction"]`` (que
    :meth:`~src.models.yolo.YoloPoseModel.export_tflite` reenvía como ``fraction``
    a la exportación de Ultralytics) para que la calibración use a lo sumo
    :data:`CALIB_MAX_IMAGES` imágenes. Sin este tope, calibrar con los ~16k frames
    del ``val`` de un fold agota la memoria.

    Args:
        config: configuración de la fase; se modifica in-place.
        n_val: cantidad de imágenes del split de validación.

    Returns:
        La fracción aplicada, o ``None`` si el split ya estaba por debajo del tope.
    """
    if n_val <= CALIB_MAX_IMAGES:
        return None
    fraction = CALIB_MAX_IMAGES / n_val
    config["calib_fraction"] = fraction
    logger.info(
        "Calibración INT8 acotada a %d/%d imágenes (fraction=%.4f).",
        CALIB_MAX_IMAGES,
        n_val,
        fraction,
    )
    return fraction


def quantize_to_int8(model: PoseModel, config: dict[str, Any], output_dir: Path) -> Path:
    """Cuantiza el modelo a INT8 y devuelve la ruta al artefacto TFLite.

    Delega en :meth:`~src.models.base.PoseModel.export_tflite` de la familia
    concreta (camino agnóstico al modelo). Para YOLO esto invoca
    ``export(int8=True, data=...)`` de Ultralytics, que **calibra las activaciones
    con el dataset representativo internamente** — el camino correcto para
    full-integer INT8. La verificación de saturación de activaciones queda a cargo
    de esa calibración interna.

    Args:
        model: instancia de :class:`~src.models.base.PoseModel` ya cargada.
        config: configuración de la fase.
        output_dir: carpeta de destino del artefacto.

    Returns:
        Ruta al archivo ``.tflite`` INT8 exportado.
    """
    return model.export_tflite(output_dir=output_dir, quantize=True)
