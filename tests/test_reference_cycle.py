"""Detección de ciclo de :mod:`src.references.cycle` (funciones puras).

La prueba central: la MISMA señal con conductor positivo y negativo debe
recortar el MISMO ciclo — el detector trabaja por |excursión| respecto del
reposo, nunca por "máximo". Las señales son sintéticas a propósito (esto testea
el detector; las referencias reales siempre salen del video).

Ejecutar::

    python tests/test_reference_cycle.py    # sin pytest
    pytest tests/                            # si pytest está instalado
"""

from __future__ import annotations

import math

import numpy as np

from src.references.angles import wrap_to_pi
from src.references.cycle import (
    detect_reps,
    detect_single_cycle,
    dtw_distance,
    rest_baseline,
    select_medoid,
)


def _one_rep(sign: float, rest: float = 0.3, amplitude: float = 2.0, fps: int = 30) -> np.ndarray:
    """reposo(1 s) → subida/bajada tipo campana (2 s) → reposo(1 s), sin ruido."""
    t_rest = np.full(fps, rest)
    t_move = rest + sign * amplitude * np.sin(np.linspace(0.0, math.pi, 2 * fps)) ** 2
    return np.concatenate([t_rest, t_move, t_rest])


def test_single_cycle_positive_and_negative_same_crop() -> None:
    """Robustez al signo: pico +2.3 y −2.3 recortan el mismo ciclo."""
    pos = _one_rep(sign=+1.0)
    neg = _one_rep(sign=-1.0)
    valid = np.ones(pos.size, dtype=bool)
    span_pos = detect_single_cycle(pos, valid, rest_frames=15, valley_tol_rad=0.15)
    span_neg = detect_single_cycle(neg, valid, rest_frames=15, valley_tol_rad=0.15)
    assert span_pos.start == span_neg.start and span_pos.end == span_neg.end
    assert span_pos.sign == +1 and span_neg.sign == -1
    # El recorte cubre el movimiento: el pico queda dentro y los bordes en reposo.
    assert span_pos.start < span_pos.peak < span_pos.end
    assert abs(pos[span_pos.start] - 0.3) < 0.2 and abs(pos[span_pos.end] - 0.3) < 0.2


def test_single_cycle_flat_signal_fails() -> None:
    flat = np.full(90, 0.3)
    try:
        detect_single_cycle(flat, np.ones(90, dtype=bool), rest_frames=15, valley_tol_rad=0.15)
    except ValueError:
        return
    raise AssertionError("detect_single_cycle aceptó una señal sin movimiento")


def test_rest_baseline_circular_near_pi() -> None:
    """Reposo alternando 3.13/−3.14 (mismo lugar físico): baseline ≈ ±π, no ≈ 0."""
    rest = np.tile([3.13, -3.14], 30)
    base = rest_baseline(rest, np.ones(rest.size, dtype=bool), rest_frames=15)
    assert abs(base) > 3.0, f"baseline {base:.3f}: la mediana circular falló cerca de ±π"


def test_rest_baseline_anchors_on_first_valid() -> None:
    """El cuadro 0 enmascarado no puede romper el ancla."""
    driver = np.concatenate([[99.0], np.full(59, 0.5)])  # cuadro 0 = basura enmascarada
    valid = np.ones(60, dtype=bool)
    valid[0] = False
    base = rest_baseline(driver, valid, rest_frames=10)
    assert math.isclose(base, 0.5, abs_tol=1e-9), base


def test_detect_single_cycle_near_pi_is_not_fooled_by_sign_flips() -> None:
    """Un miembro casi recto vive en ±π y su SIGNO baila: no es movimiento.

    Caso medido en la sentadilla frontal (rodilla a 0.2–7.8 px de la recta
    cadera-tobillo). Con un promedio crudo el suavizado mezcla +3.11 y −3.12,
    da ~0 e inventa una excursión de ~π rad; con ``smooth_wrapped`` la señal se
    reconoce como plana y el detector falla, que es lo correcto.
    """
    flat_near_pi = np.array([math.pi - 0.02 if i % 2 else -math.pi + 0.03 for i in range(120)])
    try:
        detect_single_cycle(
            flat_near_pi, np.ones(120, dtype=bool), rest_frames=15, valley_tol_rad=0.15
        )
    except ValueError:
        return
    raise AssertionError("el ruido de signo cerca de ±π se tomó como un ciclo real")


def test_detect_reps_near_pi_counts_real_reps_only() -> None:
    """Reps reales montadas sobre un reposo en ±π: se detectan esas y nada más."""
    fps = 30
    rep = _one_rep(sign=+1.0, rest=math.pi - 0.05, amplitude=1.0, fps=fps)
    signal = np.asarray(wrap_to_pi(np.concatenate([rep, rep])))
    # El reposo cruza a −π y vuelve, igual que un miembro recto con ruido.
    assert np.any(signal < -3.0) and np.any(signal > 3.0), "el caso de prueba no cruza ±π"
    spans = detect_reps(
        signal,
        np.ones(signal.size, dtype=bool),
        fps,
        prominence_rad=0.35,
        min_gap_s=0.8,
        valley_tol_rad=0.15,
        rest_frames=15,
    )
    assert len(spans) == 2, [s.peak for s in spans]


def test_detect_reps_finds_three() -> None:
    fps = 30
    rep = _one_rep(sign=-1.0, fps=fps)  # también en negativo acá
    signal = np.concatenate([rep, rep, rep])
    valid = np.ones(signal.size, dtype=bool)
    spans = detect_reps(
        signal,
        valid,
        fps,
        prominence_rad=0.35,
        min_gap_s=0.8,
        valley_tol_rad=0.15,
        rest_frames=15,
    )
    assert len(spans) == 3, [s.peak for s in spans]
    assert all(s.sign == -1 for s in spans)


def test_dtw_wraparound_and_identity() -> None:
    a = np.array([[3.10], [3.12], [-3.14]])
    b = np.array([[-3.10], [-3.12], [3.13]])  # mismos lugares físicos, signo del wrap opuesto
    w = np.array([1.0])
    assert dtw_distance(a, a, w) == 0.0
    assert dtw_distance(a, b, w) < 0.1, "DTW sin wraparound: 3.10 vs −3.10 parecería ~2π"


def test_select_medoid_rejects_outlier() -> None:
    fps = 30
    base = _one_rep(sign=+1.0, fps=fps)[:, None]
    similar1 = base + 0.02
    similar2 = base - 0.02
    outlier = _one_rep(sign=+1.0, amplitude=0.6, fps=fps)[:, None]  # rep a medio hacer
    reps = [base, similar1, similar2, outlier]
    medoid, outliers = select_medoid(reps, np.array([1.0]))
    assert medoid in (0, 1, 2)
    assert outliers == [3], outliers


# --------------------------------------------------------------------------- #
# Runner sin pytest
# --------------------------------------------------------------------------- #


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
