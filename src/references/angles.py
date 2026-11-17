"""Matemática pura de ángulos articulares en coordenadas de imagen.

Replica **bit a bit** la convención del extractor de la app (invariante crítica:
referencia y captura en vivo deben vivir en el mismo espacio de features):

* Keypoints COCO-17 en píxeles (no normalizados: una imagen no cuadrada
  deformaría el ángulo), eje **y hacia abajo**.
* Ángulo SIGNADO en el vértice entre ``v1 = A − vertex`` y ``v2 = C − vertex``::

      angle = atan2(v1.x*v2.y − v1.y*v2.x, v1·v2)   ∈ (−π, π], radianes

* Keypoint con confianza < umbral ⇒ la dimensión queda **enmascarada** en ese
  cuadro (NaN + máscara), nunca un valor mágico.

Sólo numpy + stdlib: importa y testea en cualquier venv.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.inference.keypoint_schema import COCO_KEYPOINT_NAMES

_NAME_TO_INDEX = {name.upper(): i for i, name in enumerate(COCO_KEYPOINT_NAMES)}


def keypoint_index(name: str) -> int:
    """Índice COCO-17 de un keypoint por nombre (case-insensitive, p.ej. ``LEFT_HIP``)."""
    try:
        return _NAME_TO_INDEX[name.strip().upper()]
    except KeyError:
        raise ValueError(
            f"Keypoint desconocido: {name!r}; válidos: {COCO_KEYPOINT_NAMES}"
        ) from None


@dataclass(frozen=True)
class AngleDef:
    """Una dimensión del vector de features: ángulo en ``vertex`` entre A y C."""

    name: str
    point_a: int
    vertex: int
    point_c: int
    weight: float = 1.0

    @classmethod
    def from_config(cls, entry: dict) -> AngleDef:
        return cls(
            name=str(entry["name"]),
            point_a=keypoint_index(entry["pointA"]),
            vertex=keypoint_index(entry["vertex"]),
            point_c=keypoint_index(entry["pointC"]),
            weight=float(entry.get("weight", 1.0)),
        )


def signed_angle(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    """Ángulo signado en ``vertex`` (rad, (−π, π]), coordenadas de imagen y-abajo."""
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = float(v1 @ v2)
    return float(np.arctan2(cross, dot))


def angle_matrix(
    kpts: np.ndarray, defs: list[AngleDef], conf_thr: float
) -> tuple[np.ndarray, np.ndarray]:
    """Matriz de ángulos ``(T, D)`` en rad + máscara de validez ``(T, D)``.

    ``kpts`` es ``(T, 17, 3)`` = ``(x_px, y_px, conf)``. Una dimensión es válida
    en un cuadro sólo si sus **tres** keypoints superan ``conf_thr``; si no,
    queda ``NaN`` y ``False`` en la máscara.
    """
    kpts = np.asarray(kpts, dtype=np.float64)
    n_frames = kpts.shape[0]
    angles = np.full((n_frames, len(defs)), np.nan)
    valid = np.zeros((n_frames, len(defs)), dtype=bool)
    for d, ad in enumerate(defs):
        conf_ok = (
            (kpts[:, ad.point_a, 2] >= conf_thr)
            & (kpts[:, ad.vertex, 2] >= conf_thr)
            & (kpts[:, ad.point_c, 2] >= conf_thr)
        )
        for t in np.flatnonzero(conf_ok):
            angles[t, d] = signed_angle(
                kpts[t, ad.point_a, :2], kpts[t, ad.vertex, :2], kpts[t, ad.point_c, :2]
            )
        valid[:, d] = conf_ok
    return angles, valid


def wrap_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Diferencia angular signada con wraparound: ``atan2(sin(a−b), cos(a−b))``."""
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    out = np.arctan2(np.sin(delta), np.cos(delta))
    return float(out) if out.ndim == 0 else out


def wrap_to_pi(x: np.ndarray | float) -> np.ndarray | float:
    """Normaliza un ángulo a (−π, π]."""
    return wrap_diff(x, 0.0)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Media móvil centrada de misma longitud (bordes: ventana truncada).

    Pensada para señales SIN NaN (interpolar gaps antes); con ``window <= 1``
    devuelve la señal intacta.
    """
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or x.size == 0:
        return x.copy()
    kernel = np.ones(window)
    num = np.convolve(x, kernel, mode="same")
    den = np.convolve(np.ones_like(x), kernel, mode="same")
    return num / den


def smooth_wrapped(x: np.ndarray, window: int) -> np.ndarray:
    """Media móvil sobre ángulos: desenrolla, promedia y vuelve a (−π, π].

    Promediar los valores crudos cerca de ±π mezclaría +3.1 con −3.1 y daría ~0.
    """
    if window <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    result = wrap_to_pi(moving_average(np.unwrap(np.asarray(x, dtype=np.float64)), window))
    assert isinstance(result, np.ndarray)
    return result


def gap_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Corridas contiguas de cuadros inválidos como ``(inicio, fin)`` inclusive."""
    valid = np.asarray(valid, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, ok in enumerate(valid):
        if not ok and start is None:
            start = i
        elif ok and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(valid) - 1))
    return runs


def interpolate_gaps_wrapped(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rellena cuadros inválidos interpolando por el **arco corto** (wraparound).

    Se desenrolla la señal sobre las muestras válidas (``np.unwrap``), se
    interpola linealmente ahí y se vuelve a (−π, π] — interpolar los números
    crudos cerca de ±π cruzaría por el lado largo. Gaps en los extremos se
    rellenan sosteniendo el valor válido más cercano.
    """
    x = np.asarray(x, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    idx_valid = np.flatnonzero(valid)
    if idx_valid.size == 0:
        raise ValueError("No hay ningún cuadro válido para interpolar")
    unwrapped = np.unwrap(x[idx_valid])
    filled = np.interp(np.arange(x.size), idx_valid, unwrapped)
    result = wrap_to_pi(filled)
    assert isinstance(result, np.ndarray)
    return result
