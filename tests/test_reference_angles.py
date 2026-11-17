"""Matemática de ángulos de :mod:`src.references.angles` (funciones puras).

Cubre la convención del extractor de la app: ángulo signado con eje y hacia
abajo, wraparound en ±π, enmascarado por confianza e interpolación por arco
corto. Sólo numpy: corre en cualquier venv.

Ejecutar::

    python tests/test_reference_angles.py   # sin pytest
    pytest tests/                            # si pytest está instalado
"""

from __future__ import annotations

import math

import numpy as np

from src.references.angles import (
    AngleDef,
    angle_matrix,
    gap_runs,
    interpolate_gaps_wrapped,
    keypoint_index,
    moving_average,
    signed_angle,
    wrap_diff,
)


def test_signed_angle_right_angle() -> None:
    """Configuración en L: |ángulo| = π/2, y el espejo en x invierte el signo."""
    vertex = np.array([100.0, 100.0])
    a = np.array([100.0, 200.0])  # debajo del vértice (y hacia abajo)
    c = np.array([200.0, 100.0])  # a la derecha
    ang = signed_angle(a, vertex, c)
    assert math.isclose(abs(ang), math.pi / 2, abs_tol=1e-9), ang
    c_mirror = np.array([0.0, 100.0])  # a la izquierda: espejo en x
    ang_m = signed_angle(a, vertex, c_mirror)
    assert math.isclose(abs(ang_m), math.pi / 2, abs_tol=1e-9), ang_m
    assert ang == -ang_m, (ang, ang_m)


def test_signed_angle_collinear_opposite_is_pi() -> None:
    """Brazo recto (segmentos colineales opuestos): |ángulo| = π."""
    vertex = np.array([50.0, 50.0])
    a = np.array([50.0, 0.0])
    c = np.array([50.0, 100.0])
    ang = signed_angle(a, vertex, c)
    assert math.isclose(abs(ang), math.pi, abs_tol=1e-9), ang


def test_signed_angle_mirror_flips_sign() -> None:
    """Una configuración y su espejo en x dan ángulos de signo opuesto."""
    vertex = np.array([100.0, 100.0])
    a = np.array([130.0, 180.0])
    c = np.array([170.0, 60.0])

    def mirror(p: np.ndarray) -> np.ndarray:
        return np.array([200.0 - p[0], p[1]])

    ang = signed_angle(a, vertex, c)
    ang_m = signed_angle(mirror(a), mirror(vertex), mirror(c))
    assert math.isclose(ang, -ang_m, abs_tol=1e-9), (ang, ang_m)
    assert ang != 0.0


def test_wrap_diff_across_discontinuity() -> None:
    """d(3.10, −3.10) es chica (cruza ±π por el arco corto), no ~2π."""
    d = wrap_diff(3.10, -3.10)
    expected = 6.20 - 2 * math.pi  # ≈ −0.0832: el arco corto es negativo
    assert math.isclose(float(d), expected, abs_tol=1e-9), d
    assert abs(float(d)) < 0.1


def test_keypoint_index_case_insensitive() -> None:
    assert keypoint_index("LEFT_HIP") == 11
    assert keypoint_index("left_shoulder") == 5
    try:
        keypoint_index("LEFT_TOE")
    except ValueError:
        return
    raise AssertionError("keypoint_index aceptó un nombre inexistente")


def test_angle_matrix_masks_low_confidence() -> None:
    """Confianza < umbral en cualquiera de los 3 puntos invalida la dimensión."""
    defs = [AngleDef(name="hombro_izq", point_a=11, vertex=5, point_c=7)]
    kpts = np.zeros((2, 17, 3))
    # Cuadro 0: L en el hombro izquierdo, todo con confianza alta.
    kpts[0, 11] = [100.0, 200.0, 0.9]  # left_hip
    kpts[0, 5] = [100.0, 100.0, 0.9]  # left_shoulder (vértice)
    kpts[0, 7] = [200.0, 100.0, 0.9]  # left_elbow
    # Cuadro 1: misma geometría pero el codo cae debajo del umbral.
    kpts[1] = kpts[0]
    kpts[1, 7, 2] = 0.3
    angles, valid = angle_matrix(kpts, defs, conf_thr=0.5)
    assert valid[0, 0] and not valid[1, 0]
    assert math.isclose(abs(angles[0, 0]), math.pi / 2, abs_tol=1e-9)
    assert np.isnan(angles[1, 0])


def test_interpolate_gaps_wrapped_short_arc() -> None:
    """Un gap sobre ±π se rellena por el arco corto, nunca cruzando por 0."""
    x = np.array([3.10, np.nan, np.nan, -3.10])
    valid = np.array([True, False, False, True])
    filled = interpolate_gaps_wrapped(x, valid)
    assert np.all(np.isfinite(filled))
    # Los interpolados quedan pegados a ±π (|x| > 3), no cerca de 0.
    assert np.all(np.abs(filled[1:3]) > 3.0), filled
    # Los válidos no se tocan.
    assert math.isclose(filled[0], 3.10, abs_tol=1e-9)
    assert math.isclose(filled[3], -3.10, abs_tol=1e-9)


def test_interpolate_gaps_holds_edges() -> None:
    x = np.array([np.nan, 1.0, 2.0, np.nan])
    valid = np.array([False, True, True, False])
    filled = interpolate_gaps_wrapped(x, valid)
    assert math.isclose(filled[0], 1.0, abs_tol=1e-9)
    assert math.isclose(filled[3], 2.0, abs_tol=1e-9)


def test_gap_runs_finds_contiguous_gaps() -> None:
    valid = np.array([True, False, False, True, False, True])
    assert gap_runs(valid) == [(1, 2), (4, 4)]
    assert gap_runs(np.array([True, True])) == []
    assert gap_runs(np.array([False, False])) == [(0, 1)]


def test_moving_average_preserves_length_and_constants() -> None:
    x = np.full(10, 0.7)
    sm = moving_average(x, 5)
    assert sm.shape == x.shape
    assert np.allclose(sm, 0.7)
    assert np.allclose(moving_average(x, 1), x)


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
