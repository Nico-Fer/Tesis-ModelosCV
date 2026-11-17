"""Detección del ciclo de movimiento — robusta al signo del ángulo conductor.

El signo real del conductor lo define la geometría + el eje y hacia abajo, y
puede salir negativo. Por eso el ciclo se detecta por **excursión respecto del
reposo**, no por "máximo"::

    baseline    = mediana circular del conductor en los cuadros de reposo
    excursión_i = wrap_diff(conductor_i, baseline)
    extremo     = argmax(|excursión|)
    valles      = |excursión| vuelve cerca de 0 antes y después del extremo

Así el recorte es idéntico con pico en +2.6 o en −2.6. Toda distancia entre
ángulos usa wraparound (:func:`~src.references.angles.wrap_diff`), y el
suavizado de detección usa :func:`~src.references.angles.smooth_wrapped` —
**nunca** ``moving_average`` sobre los valores crudos: una señal que vive cerca
de ±π (un miembro casi recto: el ángulo articular vale ~π y sólo su *signo*
baila con el ruido sub-píxel) promediaría +3.11 con −3.12 y daría ~0, o sea
inventaría una excursión de π rad donde el miembro no se movió. Medido: en la
sentadilla frontal la rodilla se desvía 0.2–7.8 px de la recta cadera-tobillo y
el signo salta entre cuadros consecutivos; con el promedio crudo la señal de
excursión era una onda cuadrada de ±2.5 rad.

Sólo numpy + stdlib: puro y testeable en cualquier venv.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.references.angles import smooth_wrapped, wrap_diff, wrap_to_pi

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleSpan:
    """Recorte de un ciclo: índices inclusive, ``sign`` = signo de la excursión pico."""

    start: int
    end: int
    peak: int
    sign: int


def rest_baseline(driver: np.ndarray, valid: np.ndarray, rest_frames: int) -> float:
    """Mediana **circular** del conductor sobre los cuadros de reposo (inicio + fin).

    Cada muestra se expresa como diferencia envuelta contra un ancla (el primer
    cuadro **válido**, no el cuadro 0: puede estar enmascarado), se mediana esa
    diferencia y se vuelve al círculo. Una mediana común fallaría con reposo
    cerca de ±π: 3.13 y −3.14 son casi el mismo lugar físico.
    """
    driver = np.asarray(driver, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValueError("El conductor no tiene ningún cuadro válido")
    head = idx[idx < rest_frames]
    tail = idx[idx >= driver.size - rest_frames]
    rest_idx = np.concatenate([head, tail])
    if rest_idx.size == 0:
        raise ValueError(f"Sin cuadros de reposo válidos en los primeros/últimos {rest_frames}")
    anchor = driver[idx[0]]
    diffs = wrap_diff(driver[rest_idx], anchor)
    result = wrap_to_pi(anchor + float(np.median(diffs)))
    return float(result)


def excursion(driver: np.ndarray, baseline: float) -> np.ndarray:
    """Excursión signada con wraparound de cada cuadro respecto del reposo."""
    out = wrap_diff(driver, baseline)
    assert isinstance(out, np.ndarray)
    return out


def detect_single_cycle(
    driver: np.ndarray,
    valid: np.ndarray,
    *,
    rest_frames: int,
    valley_tol_rad: float,
    detect_window: int = 5,
) -> CycleSpan:
    """Recorta UN ciclo reposo→extremo→reposo (modo default, video de 1 rep limpia).

    ``driver`` debe venir SIN NaN (interpolar gaps antes); ``valid`` sólo se usa
    para el baseline. El suavizado es exclusivo de la detección.
    """
    smooth = smooth_wrapped(driver, detect_window)
    base = rest_baseline(driver, valid, rest_frames)
    exc = excursion(smooth, base)
    abs_exc = np.abs(exc)
    peak = int(np.argmax(abs_exc))
    if abs_exc[peak] <= valley_tol_rad:
        raise ValueError(
            f"El conductor no se mueve: |excursión| máxima {abs_exc[peak]:.3f} rad "
            f"<= tolerancia de valle {valley_tol_rad:.3f}"
        )
    start = _walk_to_valley(abs_exc, peak, -1, valley_tol_rad)
    end = _walk_to_valley(abs_exc, peak, +1, valley_tol_rad)
    return CycleSpan(start=start, end=end, peak=peak, sign=int(np.sign(exc[peak])))


def _walk_to_valley(abs_exc: np.ndarray, peak: int, step: int, tol: float) -> int:
    """Camina desde el pico hasta que |excursión| < tol; devuelve ese índice."""
    i = peak
    while 0 <= i + step < abs_exc.size:
        i += step
        if abs_exc[i] < tol:
            return i
    raise ValueError(
        f"|excursión| nunca vuelve por debajo de {tol:.3f} rad "
        f"{'antes' if step < 0 else 'después'} del extremo (cuadro {peak}): "
        "el video no parece cubrir reposo→extremo→reposo"
    )


def detect_reps(
    driver: np.ndarray,
    valid: np.ndarray,
    fps: float,
    *,
    prominence_rad: float,
    min_gap_s: float,
    valley_tol_rad: float,
    rest_frames: int,
    detect_window: int = 5,
) -> list[CycleSpan]:
    """Detecta varias reps (modo ``--multi``) por prominencia + separación mínima."""
    smooth = smooth_wrapped(driver, detect_window)
    base = rest_baseline(driver, valid, rest_frames)
    exc = excursion(smooth, base)
    abs_exc = np.abs(exc)
    min_gap = max(1, int(round(min_gap_s * fps)))
    peaks = _find_peaks(abs_exc, prominence=prominence_rad, min_distance=min_gap)
    spans: list[CycleSpan] = []
    for peak in peaks:
        try:
            start = _walk_to_valley(abs_exc, peak, -1, valley_tol_rad)
            end = _walk_to_valley(abs_exc, peak, +1, valley_tol_rad)
        except ValueError as exc_err:
            logger.warning("Rep con pico en el cuadro %d descartada: %s", peak, exc_err)
            continue
        spans.append(CycleSpan(start=start, end=end, peak=peak, sign=int(np.sign(exc[peak]))))
    return spans


def _find_peaks(x: np.ndarray, *, prominence: float, min_distance: int) -> list[int]:
    """Máximos locales 1-D con prominencia mínima (numpy puro, sin scipy).

    Prominencia = altura del pico − el mayor de los mínimos entre el pico y el
    primer pico más alto (o el borde) hacia cada lado.
    """
    candidates = [i for i in range(1, x.size - 1) if x[i] > x[i - 1] and x[i] >= x[i + 1]]
    prominent: list[tuple[int, float]] = []
    for i in candidates:
        left_min = x[i]
        for j in range(i - 1, -1, -1):
            left_min = min(left_min, x[j])
            if x[j] > x[i]:
                break
        right_min = x[i]
        for j in range(i + 1, x.size):
            right_min = min(right_min, x[j])
            if x[j] > x[i]:
                break
        prom = x[i] - max(left_min, right_min)
        if prom >= prominence:
            prominent.append((i, x[i]))
    # De mayor a menor altura, descartando los que caen a < min_distance de uno aceptado.
    prominent.sort(key=lambda p: -p[1])
    kept: list[int] = []
    for i, _ in prominent:
        if all(abs(i - k) >= min_distance for k in kept):
            kept.append(i)
    return sorted(kept)


def dtw_distance(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    """DTW entre trayectorias ``(Ta, D)`` y ``(Tb, D)``, normalizada por camino.

    Costo por paso = Σ_d w_d · |wrap_diff(a_d, b_d)| — el wraparound importa:
    sin él, 3.10 y −3.10 parecerían lejísimos.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    weights = np.asarray(weights, dtype=np.float64)
    ta, tb = a.shape[0], b.shape[0]
    # Matriz de costo local vectorizada: (Ta, Tb).
    diff = wrap_diff(a[:, None, :], b[None, :, :])
    local = np.abs(diff) @ weights
    acc = np.full((ta + 1, tb + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, ta + 1):
        for j in range(1, tb + 1):
            acc[i, j] = local[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[ta, tb]) / (ta + tb)


def select_medoid(
    reps: list[np.ndarray], weights: np.ndarray, *, outlier_factor: float = 2.0
) -> tuple[int, list[int]]:
    """Rep MEDIANA por medoide DTW + outliers (d al medoide > factor·mediana).

    ``outlier_factor`` es placeholder.  # TODO: calibrar
    """
    if not reps:
        raise ValueError("No hay reps para elegir medoide")
    if len(reps) == 1:
        return 0, []
    n = len(reps)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = dtw_distance(reps[i], reps[j], weights)
    medoid = int(np.argmin(dist.sum(axis=1)))
    to_medoid = np.delete(dist[medoid], medoid)
    threshold = outlier_factor * float(np.median(to_medoid))
    outliers = [
        i for i in range(n) if i != medoid and dist[medoid, i] > threshold and threshold > 0
    ]
    return medoid, outliers
