"""Mapa frame → job de CVAT y señal de auditado (:mod:`src.audit.cvat_frames`).

"Auditado" sale del estado del job, nunca de la presencia en el export (CVAT
exporta todos los frames pre-anotados, revisados o no). Estos tests fijan esa
regla y la construcción del mapa a partir de lo que devuelve la API.
Framework-free: corre en cualquier venv.

Ejecutar::

    python tests/test_audit_cvat_frames.py  # sin pytest
    pytest tests/                            # si pytest está instalado
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.audit.cvat_frames import (
    build_frame_map,
    read_frame_map,
    stems_auditados,
    write_frame_map,
)

_NOMBRES = ["b_01_cam0_000010.jpg", "a_01_cam0_000000.jpg", "c_01_cam0_000020.jpg"]
_JOBS = [
    {"id": 7, "start_frame": 0, "stop_frame": 1, "type": "annotation"},
    {"id": 8, "start_frame": 2, "stop_frame": 2, "type": "annotation"},
    # Un job de ground truth solapa frames pero no es de anotación: se ignora.
    {"id": 99, "start_frame": 0, "stop_frame": 2, "type": "ground_truth"},
]


def _raises(fn, exc: type[Exception]) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def test_build_frame_map_asigna_por_rango_y_respeta_orden() -> None:
    filas = build_frame_map(3, _NOMBRES, _JOBS)
    assert [f["stem"] for f in filas] == [
        "b_01_cam0_000010",
        "a_01_cam0_000000",
        "c_01_cam0_000020",
    ]
    assert [f["job_id"] for f in filas] == ["7", "7", "8"]
    assert [f["frame"] for f in filas] == ["0", "1", "2"]
    assert {f["task_id"] for f in filas} == {"3"}


def test_build_frame_map_rechaza_huecos_solapes_y_duplicados() -> None:
    sin_job = [{"id": 7, "start_frame": 0, "stop_frame": 1, "type": "annotation"}]
    assert _raises(lambda: build_frame_map(3, _NOMBRES, sin_job), ValueError)
    solape = _JOBS[:2] + [{"id": 9, "start_frame": 1, "stop_frame": 1, "type": "annotation"}]
    assert _raises(lambda: build_frame_map(3, _NOMBRES, solape), ValueError)
    dup = ["a_01_cam0_000000.jpg", "sub/a_01_cam0_000000.jpg", "c_01_cam0_000020.jpg"]
    assert _raises(lambda: build_frame_map(3, dup, _JOBS), ValueError)


def test_auditados_solo_jobs_completados() -> None:
    filas = build_frame_map(3, _NOMBRES, _JOBS)
    estados = {7: "completed", 8: "in progress"}
    assert stems_auditados(filas, estados, "completed") == ["a_01_cam0_000000", "b_01_cam0_000010"]
    assert stems_auditados(filas, {7: "new", 8: "new"}, "completed") == []


def test_auditados_excluye_frames_borrados() -> None:
    filas = build_frame_map(3, _NOMBRES, _JOBS)
    estados = {7: "completed", 8: "completed"}
    auditados = stems_auditados(filas, estados, "completed", frames_borrados={3: {1}})
    assert "a_01_cam0_000000" not in auditados and len(auditados) == 2


def test_auditados_job_inexistente_aborta() -> None:
    filas = build_frame_map(3, _NOMBRES, _JOBS)
    assert _raises(lambda: stems_auditados(filas, {7: "completed"}, "completed"), ValueError)


def test_frame_map_round_trip_por_disco() -> None:
    filas = build_frame_map(3, _NOMBRES, _JOBS)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "frames.csv"
        write_frame_map(filas, path)
        assert read_frame_map(path) == filas


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
