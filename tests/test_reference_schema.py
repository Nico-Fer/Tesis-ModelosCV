"""Schema v1 de :mod:`src.references.schema` (emisión y validación puras).

Ejecutar::

    python tests/test_reference_schema.py   # sin pytest
    pytest tests/                            # si pytest está instalado
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from src.references.schema import build_reference, validate_reference, write_reference

_DEFS = [
    {
        "name": "hombro_izq",
        "pointA": "LEFT_HIP",
        "vertex": "LEFT_SHOULDER",
        "pointC": "LEFT_ELBOW",
        "weight": 1.0,
    },
    {
        "name": "hombro_der",
        "pointA": "RIGHT_HIP",
        "vertex": "RIGHT_SHOULDER",
        "pointC": "RIGHT_ELBOW",
        "weight": 1.0,
    },
]
_SCORING = {"similarityScale": 0.5, "goodThreshold": 0.7}


def _trajectory(t: int = 40, d: int = 2) -> np.ndarray:
    """Ciclo suave que empieza y termina en el mismo reposo (0.3 rad)."""
    curve = 0.3 + 1.8 * np.sin(np.linspace(0.0, math.pi, t)) ** 2
    return np.stack([curve] * d, axis=1)


def _build(traj: np.ndarray | None = None) -> dict:
    return build_reference(
        exercise_id="C1111111-2222-3333-4444-555555555555",
        angle_definitions=_DEFS,
        driver_angle_index=0,
        trajectory=_trajectory() if traj is None else traj,
        fps=29.97,
        video_name="videos/levantamiento_brazos.mp4",
        scoring=_SCORING,
    )


def test_exact_key_sets() -> None:
    doc = _build()
    assert set(doc) == {
        "schemaVersion",
        "exerciseId",
        "angleDefinitions",
        "driverAngleIndex",
        "referenceTrajectory",
        "metadata",
        "scoring",
    }
    assert set(doc["metadata"]) == {"fps", "version", "source"}
    assert set(doc["scoring"]) == {"similarityScale", "goodThreshold"}
    for d in doc["angleDefinitions"]:
        assert set(d) == {"name", "pointA", "vertex", "pointC", "weight"}
    assert doc["schemaVersion"] == 1
    assert doc["metadata"]["source"] == "video:levantamiento_brazos.mp4"
    assert doc["metadata"]["fps"] == 29.97


def test_trajectory_shape_rounding_and_finiteness() -> None:
    doc = _build()
    traj = doc["referenceTrajectory"]
    assert len(traj) == 40
    assert all(len(row) == 2 for row in traj)
    for row in traj:
        for v in row:
            assert isinstance(v, float) and math.isfinite(v)
            assert v == round(v, 4), f"{v} no está redondeado a 4 decimales"


def test_first_and_last_rows_close() -> None:
    doc = _build()
    assert validate_reference(doc) == []
    first, last = doc["referenceTrajectory"][0], doc["referenceTrajectory"][-1]
    assert all(abs(a - b) < 0.35 for a, b in zip(first, last))


def test_nonfinite_trajectory_rejected_at_build() -> None:
    traj = _trajectory()
    traj[3, 1] = np.nan
    try:
        _build(traj)
    except ValueError as exc:
        assert "no finitos" in str(exc)
        return
    raise AssertionError("build_reference aceptó una trayectoria con NaN")


def test_validate_flags_open_cycle_and_bad_keys() -> None:
    open_cycle = _trajectory()
    open_cycle[-1, :] = open_cycle[-1, :] + 1.5  # termina lejos del reposo
    errors = validate_reference(_build(open_cycle))
    assert any("primera≈última" in e for e in errors), errors

    doc = _build()
    doc["extra"] = 1
    assert any("top-level" in e for e in validate_reference(doc))


def test_write_reference_roundtrip() -> None:
    doc = _build()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_reference(doc, Path(tmp))
        # filename lowercase (contrato de assets); exerciseId intacto adentro
        assert path.name == f"{doc['exerciseId'].lower()}.json"
        assert path.name != f"{doc['exerciseId']}.json"
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == doc
        assert loaded["exerciseId"] == doc["exerciseId"]
        assert validate_reference(loaded) == []


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
