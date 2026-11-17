"""Métricas de evaluación de pose estimation.

Centraliza el cómputo de las métricas comparables del pipeline. Todas las
funciones reciben arrays numpy ya computados; no tienen dependencias de dataset
ni de runners, por lo que son unitariamente testeables sin datos reales.

Dependencias:
    numpy: disponible en ambos venvs (requirements-base.txt).
    tensorflow (sólo ``measure_latency``): import diferido para mantener el
        módulo importable en venvs sin TF instalado.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

# Sigmas oficiales del COCO Keypoint Challenge para los 17 keypoints:
# nose, left/right eye, left/right ear, left/right shoulder,
# left/right elbow, left/right wrist, left/right hip,
# left/right knee, left/right ankle.
_COCO_SIGMAS = np.array(
    [
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ],
    dtype=np.float64,
)


def compute_oks(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    gt_bboxes: np.ndarray,
    sigmas: np.ndarray | None = None,
) -> float:
    """Object Keypoint Similarity promedio sobre N instancias (fórmula COCO).

    La métrica mide cuán bien las predicciones se alinean con los keypoints
    ground-truth normalizando por el área del bounding box y los sigmas por
    keypoint, que codifican la variabilidad de anotación esperada::

        OKS_i = Σ_k exp(−dₖ² / (2·sᵢ²·(2·σₖ)²)) · δ(vₖ > 0)
                ─────────────────────────────────────────────────
                              Σ_k δ(vₖ > 0)

    donde ``dₖ`` es la distancia euclidiana al keypoint k, ``sᵢ²`` el área
    del bbox de la instancia i, y ``σₖ`` el sigma COCO del keypoint k.

    Args:
        pred_keypoints: ``(N, K, 3)`` floats — x, y, confidence de cada
            keypoint predicho. Solo se usan las columnas x, y.
        gt_keypoints: ``(N, K, 3)`` floats — x, y, visibility. Keypoints con
            visibility == 0 se excluyen del cómputo.
        gt_bboxes: ``(N, 4)`` floats — x, y, w, h en píxeles. El área es
            ``w * h``; se añade un epsilon para evitar divisiones por cero.
        sigmas: ``(K,)`` floats — sigmas por keypoint. Si es None se usan los
            17 sigmas oficiales de COCO; para K ≠ 17 es obligatorio pasarlos.

    Returns:
        OKS promedio sobre las N instancias (escalar entre 0.0 y 1.0).

    Raises:
        ValueError: si las formas de los arrays son incompatibles.
    """
    pred_keypoints = np.asarray(pred_keypoints, dtype=np.float64)
    gt_keypoints = np.asarray(gt_keypoints, dtype=np.float64)
    gt_bboxes = np.asarray(gt_bboxes, dtype=np.float64)

    n, k, _ = pred_keypoints.shape
    if gt_keypoints.shape != (n, k, 3):
        raise ValueError(
            f"Forma de gt_keypoints {gt_keypoints.shape} incompatible con "
            f"pred_keypoints {pred_keypoints.shape}."
        )
    if gt_bboxes.shape != (n, 4):
        raise ValueError(f"gt_bboxes debe ser (N, 4), recibido {gt_bboxes.shape}.")

    if sigmas is None:
        if k != 17:
            raise ValueError(
                f"sigmas=None supone K=17 (COCO), pero K={k}. "
                "Pasá sigmas explícitamente para otros esqueletos."
            )
        sigmas = _COCO_SIGMAS
    else:
        sigmas = np.asarray(sigmas, dtype=np.float64)
        if sigmas.shape != (k,):
            raise ValueError(f"sigmas debe tener forma ({k},), recibido {sigmas.shape}.")

    # Áreas de bboxes: w * h, con epsilon para evitar /0.
    areas = gt_bboxes[:, 2] * gt_bboxes[:, 3] + np.finfo(np.float64).eps  # (N,)

    # Visibilidad: gt_keypoints[:, :, 2] > 0 → keypoint anotado.
    visibility = gt_keypoints[:, :, 2] > 0  # (N, K) bool

    # Distancias euclidianas al cuadrado.
    dx = pred_keypoints[:, :, 0] - gt_keypoints[:, :, 0]  # (N, K)
    dy = pred_keypoints[:, :, 1] - gt_keypoints[:, :, 1]  # (N, K)
    d2 = dx**2 + dy**2  # (N, K)

    # k_i = 2 * sigma_k (ancho del kernel gaussiano).
    k2 = (2.0 * sigmas) ** 2  # (K,)

    # Exponencial: exp(-d² / (2 * s² * k²)); s² = area por instancia.
    exp_val = np.exp(-d2 / (2.0 * areas[:, None] * k2[None, :]))  # (N, K)

    # Enmascarar keypoints no visibles.
    exp_val_masked = np.where(visibility, exp_val, 0.0)

    # OKS por instancia: suma de exponenciales visibles / nº keypoints visibles.
    n_visible = visibility.sum(axis=1).clip(min=1)  # (N,) evitar /0
    oks_per_instance = exp_val_masked.sum(axis=1) / n_visible  # (N,)

    return float(oks_per_instance.mean())


def compute_pck(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    threshold: float = 0.2,
    *,
    reference_size: np.ndarray,
) -> float:
    """Percentage of Correct Keypoints (PCK@threshold).

    Un keypoint se considera correcto si su distancia euclidiana al ground-truth
    es menor o igual a ``threshold * reference_size[i]`` para la instancia i.
    Los keypoints con visibilidad 0 en gt se excluyen del conteo.

    Args:
        pred_keypoints: ``(N, K, 2)`` floats — x, y de cada keypoint predicho.
        gt_keypoints: ``(N, K, 3)`` floats — x, y, visibility. Solo se usan
            x, y para la distancia; visibility filtra los keypoints a evaluar.
        threshold: fracción del ``reference_size`` que define el radio de
            aceptación (default 0.2, equivalente a PCK@0.2).
        reference_size: ``(N,)`` floats — lado de referencia por instancia
            (típicamente el tamaño del torso o diagonal del bbox).

    Returns:
        PCK promedio sobre todos los keypoints visibles de todas las instancias
        (escalar entre 0.0 y 1.0).

    Raises:
        ValueError: si las formas de los arrays son incompatibles.
    """
    pred_keypoints = np.asarray(pred_keypoints, dtype=np.float64)
    gt_keypoints = np.asarray(gt_keypoints, dtype=np.float64)
    reference_size = np.asarray(reference_size, dtype=np.float64)

    n, k, pred_dim = pred_keypoints.shape
    if pred_dim < 2:
        raise ValueError(f"pred_keypoints debe tener al menos 2 columnas (x,y), tiene {pred_dim}.")
    if gt_keypoints.shape[:2] != (n, k):
        raise ValueError(
            f"Forma de gt_keypoints {gt_keypoints.shape} incompatible con "
            f"pred_keypoints {pred_keypoints.shape}."
        )
    if reference_size.shape != (n,):
        raise ValueError(
            f"reference_size debe tener forma ({n},), recibido {reference_size.shape}."
        )

    visibility = gt_keypoints[:, :, 2] > 0  # (N, K)

    dx = pred_keypoints[:, :, 0] - gt_keypoints[:, :, 0]
    dy = pred_keypoints[:, :, 1] - gt_keypoints[:, :, 1]
    dist = np.sqrt(dx**2 + dy**2)  # (N, K)

    radius = threshold * reference_size[:, None]  # (N, 1) broadcast → (N, K)
    correct = (dist <= radius) & visibility  # (N, K)

    n_visible = visibility.sum()
    if n_visible == 0:
        return 0.0

    return float(correct.sum() / n_visible)


def measure_latency(
    model_path: str | Path,
    input_shape: tuple[int, ...],
    n_warmup: int = 10,
    n_runs: int = 50,
) -> float:
    """Latencia media de inferencia de un modelo TFLite, en milisegundos.

    Carga el intérprete TFLite, ejecuta ``n_warmup`` pasadas sin medir para
    estabilizar caché/JIT y luego ``n_runs`` pasadas cronometradas. Se usa
    ``time.perf_counter`` para resolución sub-milisegundo.

    Args:
        model_path: ruta al archivo ``.tflite``. Si la extensión no es
            ``.tflite`` se lanza ``ValueError``.
        input_shape: forma del tensor de entrada, p. ej. ``(1, 640, 640, 3)``.
        n_warmup: pasadas de calentamiento (sin medir, default 10).
        n_runs: pasadas cronometradas (default 50).

    Returns:
        Latencia media en milisegundos.

    Raises:
        ValueError: si ``model_path`` no tiene extensión ``.tflite``.
        FileNotFoundError: si el archivo no existe.
    """
    model_path = Path(model_path)

    if model_path.suffix.lower() != ".tflite":
        raise ValueError(
            f"measure_latency sólo admite modelos .tflite; "
            f"recibido: '{model_path.suffix}'. "
            "Para .pt usa Ultralytics benchmark; para .onnx usa onnxruntime."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo TFLite no encontrado: {model_path}")

    # Import diferido: TF está en ambos venvs pero seguimos el patrón lazy.
    import tensorflow as tf  # noqa: PLC0415

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_index = input_details[0]["index"]
    input_dtype = input_details[0]["dtype"]

    # El dummy debe respetar el dtype real del tensor de entrada: un .tflite INT8
    # full-integer espera int8/uint8, no float32 (usar float32 rompería el invoke).
    # Se prioriza el shape declarado por el intérprete si está disponible.
    interp_shape = input_details[0].get("shape")
    shape = tuple(int(d) for d in interp_shape) if interp_shape is not None else input_shape
    if np.issubdtype(input_dtype, np.integer):
        info = np.iinfo(input_dtype)
        dummy_input = np.random.randint(info.min, info.max + 1, size=shape, dtype=input_dtype)
    else:
        dummy_input = np.random.rand(*shape).astype(input_dtype)

    for _ in range(n_warmup):
        interpreter.set_tensor(input_index, dummy_input)
        interpreter.invoke()

    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_index, dummy_input)
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1_000.0)  # segundos → ms

    return float(np.mean(times))


def model_size_mb(model_path: str | Path) -> float:
    """Tamaño de un artefacto de modelo en megabytes.

    Args:
        model_path: ruta al archivo del modelo (``.pt``, ``.tflite``, etc.).

    Returns:
        Tamaño en megabytes (``os.path.getsize / 1e6``).

    Raises:
        FileNotFoundError: si el archivo no existe en disco.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Archivo de modelo no encontrado: {model_path}")
    return os.path.getsize(model_path) / 1e6
