"""Selección del subconjunto de auditoría (:mod:`src.audit.select_audit_subset`).

Cubre la clasificación por motivo (la precedencia importa: un frame sin
detección también tiene score 0, y la discrepancia con el GT va antes que las
sospechas de confianza), la comparación pseudo vs GT de UCO, la exclusión de
``multi_persona`` sin discrepancia, el chequeo de consistencia contra los
umbrales de la corrida, el mapa ejercicio → postura y la selección completa
sobre un mini-dataset sintético. Framework-free: corre en cualquier venv.

Ejecutar::

    python tests/test_audit_subset.py      # sin pytest
    pytest tests/                           # si pytest está instalado
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from src.audit.common import (
    AuditConfig,
    PoseLabel,
    build_config,
    discrepancia_gt,
    read_selection,
)
from src.audit.select_audit_subset import motivo_de, select, write_selection
from src.inference.keypoint_schema import NUM_COCO_KEYPOINTS

_MANIFEST_FIELDS = [
    "image",
    "subject",
    "exercise",
    "cam",
    "frame",
    "n_personas_detectadas",
    "n_personas_escritas",
    "det_score",
    "mean_kpt_score",
    "n_v2",
    "n_v1",
    "n_v0",
    "baja_confianza",
]
_W, _H = 1280, 720
_MEAN_THR, _MIN_VIS = 0.5, 10
# Articulaciones que UCO anota en un ejercicio de pierna izquierda (COCO 11/13/15).
_GT_JOINTS = (11, 13, 15)


def _row(
    stem: str, detectadas: int, escritas: int, mean: float = 0.85, n_v2: int = 15
) -> dict[str, str]:
    """Fila de manifest con ``baja_confianza`` calculado con las reglas reales."""
    from src.data.uco_to_yolo import parse_stem

    parts = parse_stem(stem)
    baja = detectadas != 1 or escritas == 0 or mean < _MEAN_THR or n_v2 < _MIN_VIS
    return {
        "image": f"{stem}.jpg",
        "subject": parts.subject,
        "exercise": parts.exercise,
        "cam": parts.cam,
        "frame": str(parts.frame),
        "n_personas_detectadas": str(detectadas),
        "n_personas_escritas": str(escritas),
        "det_score": "0.9",
        "mean_kpt_score": f"{mean:.6f}",
        "n_v2": str(n_v2),
        "n_v1": "1",
        "n_v0": "1",
        "baja_confianza": str(int(baja)),
    }


def _line(joints: dict[int, tuple[float, float]], resto_visible: bool) -> str:
    """Línea YOLO-pose: ``joints`` en px; el resto visible en (0.5, 0.5) o en v=0."""
    toks = ["0", "0.5", "0.5", "0.2", "0.4"]
    for k in range(NUM_COCO_KEYPOINTS):
        if k in joints:
            x, y = joints[k]
            toks += [f"{x / _W:.6f}", f"{y / _H:.6f}", "2"]
        elif resto_visible:
            toks += ["0.5", "0.5", "2"]
        else:
            toks += ["0", "0", "0"]
    return " ".join(toks)


_GT_PX = {11: (600.0, 300.0), 13: (620.0, 450.0), 15: (640.0, 600.0)}


def _dataset(
    tmp: Path, filas: list[tuple[dict[str, str], str]], **overrides: object
) -> AuditConfig:
    """Mini-dataset: manifest + labels pseudo/GT + 1 imagen + run_metadata.

    ``filas``: ``(row, pseudo)`` donde ``pseudo`` es ``"ok"`` (coincide con el GT),
    ``"lejos"`` (una articulación a 150 px), ``"perdido"`` (una articulación en
    v=0) o ``"vacio"`` (sin persona).
    """
    import cv2
    import numpy as np

    img_dir, lbl_dir, gt_dir = tmp / "images", tmp / "labels", tmp / "gt"
    for d in (img_dir, lbl_dir, gt_dir):
        d.mkdir()
    manifest = tmp / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        for row, _ in filas:
            writer.writerow(row)
    for row, pseudo in filas:
        stem = Path(row["image"]).stem
        (gt_dir / f"{stem}.txt").write_text(_line(_GT_PX, False) + "\n", encoding="utf-8")
        if pseudo == "vacio":
            texto = ""
        else:
            joints = dict(_GT_PX)
            if pseudo == "lejos":
                joints[13] = (620.0 + 150.0, 450.0)
            texto = _line(joints, True)
            if pseudo == "perdido":
                toks = texto.split()
                toks[5 + 3 * 15 : 8 + 3 * 15] = ["0", "0", "0"]
                texto = " ".join(toks)
            texto += "\n"
        (lbl_dir / f"{stem}.txt").write_text(texto, encoding="utf-8")
    primera = Path(filas[0][0]["image"]).stem
    cv2.imwrite(str(img_dir / f"{primera}.jpg"), np.zeros((_H, _W, 3), dtype=np.uint8))
    run_meta = tmp / "run_metadata.json"
    run_meta.write_text(
        json.dumps(
            {"config": {"review": {"mean_score_thr": _MEAN_THR, "min_visible_kpts": _MIN_VIS}}}
        ),
        encoding="utf-8",
    )
    raw = {
        "pseudo": {
            "manifest": str(manifest),
            "images_dir": str(img_dir),
            "labels_dir": str(lbl_dir),
            "run_metadata": str(run_meta),
        },
        "gt": {
            "labels_dir": str(gt_dir),
            "img_size": list(overrides.pop("img_size", [_W, _H])),  # type: ignore[call-overload]
            "discrepancia_thr_px": 100,
        },
        "seleccion": {
            "control_size": overrides.pop("control_size", 2),
            "out_csv": str(tmp / "selection.csv"),
            "muestra_multi_persona": overrides.pop("muestra_multi", 1),
            "excluir_multi_sin_discrepancia": overrides.pop("excluir_multi", True),
        },
        "estratos": {
            "postura": {f"{k:02d}": "sedente" for k in range(1, 9)} | {"09": "brazo", "13": "brazo"}
        },
    }
    return build_config(raw)


# --------------------------------------------------------------------------- #
# Lógica pura
# --------------------------------------------------------------------------- #


def test_motivo_precedencia() -> None:
    def m(det: int, esc: int, disc: bool, mean: float = 0.85, n_v2: int = 15) -> str | None:
        return motivo_de(_row("0_01_cam0_000001", det, esc, mean, n_v2), disc, _MEAN_THR, _MIN_VIS)

    assert m(0, 0, True) == "sin_deteccion"
    assert m(1, 0, True) == "sin_kpts_validos"
    assert m(2, 1, True) == "discrepancia_gt"
    assert m(1, 1, True, mean=0.3) == "discrepancia_gt"
    assert m(1, 1, False, mean=0.3) == "score_bajo"
    assert m(1, 1, False, n_v2=5) == "score_bajo"
    assert m(2, 1, False, mean=0.3) == "score_bajo", "score_bajo precede a multi_persona"
    assert m(2, 1, False) == "multi_persona"
    assert m(1, 1, False) is None


def test_discrepancia_gt_error_y_perdido() -> None:
    def persona(joints: dict[int, tuple[float, float, int]]) -> PoseLabel:
        kpts = tuple(joints.get(k, (0.0, 0.0, 0)) for k in range(NUM_COCO_KEYPOINTS))
        return PoseLabel(bbox=(0.5, 0.5, 0.1, 0.1), kpts=kpts)

    gt = persona({11: (0.5, 0.5, 2), 13: (0.5, 0.6, 2)})
    # kpt 11 corrido 0.1 en x = 128 px; kpt 13 exacto; kpt 0 (sin GT) no cuenta.
    pseudo = persona({11: (0.6, 0.5, 2), 13: (0.5, 0.6, 1), 0: (0.9, 0.9, 2)})
    d = discrepancia_gt([gt], [pseudo], _W, _H)
    assert d.err_px is not None and abs(d.err_px - 128.0) < 1e-6
    assert not d.kpt_perdido
    assert d.es_discrepante(100.0) and not d.es_discrepante(200.0)

    perdido = persona({11: (0.5, 0.5, 2), 13: (0.0, 0.0, 0)})
    d = discrepancia_gt([gt], [perdido], _W, _H)
    assert d.kpt_perdido and d.err_px == 0.0 and d.es_discrepante(1e9)

    assert discrepancia_gt([gt], [], _W, _H).err_px is None
    assert not discrepancia_gt([], [pseudo], _W, _H).es_discrepante(0.0)


# --------------------------------------------------------------------------- #
# Selección completa
# --------------------------------------------------------------------------- #


def test_select_motivos_exclusion_multi_y_control() -> None:
    filas = [
        (_row("0_01_cam0_000001", 0, 0, mean=0.0, n_v2=0), "vacio"),  # sin_deteccion
        (_row("1_05_cam1_000002", 1, 1), "lejos"),  # discrepancia (no marcado por el teacher)
        (_row("1_05_cam1_000003", 1, 1), "perdido"),  # discrepancia por kpt perdido
        (_row("2_09_cam2_000004", 1, 1, mean=0.3), "ok"),  # score_bajo
        (_row("3_13_cam0_000005", 2, 1), "ok"),  # multi sin discrepancia
        (_row("3_13_cam0_000006", 2, 1), "ok"),  # multi sin discrepancia
        (_row("4_02_cam3_000007", 2, 1), "lejos"),  # multi CON discrepancia
    ] + [(_row(f"{s}_03_cam0_{n:06d}", 1, 1), "ok") for s, n in ((5, 8), (6, 9), (7, 10))]
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _dataset(Path(tmpdir), filas, control_size=2, muestra_multi=1)
        sel = select(cfg, workers=2)
        motivo = {f["stem"]: f["motivo"] for f in sel}

        assert motivo["0_01_cam0_000001"] == "sin_deteccion"
        assert motivo["1_05_cam1_000002"] == "discrepancia_gt"
        assert motivo["1_05_cam1_000003"] == "discrepancia_gt"
        assert motivo["2_09_cam2_000004"] == "score_bajo"
        assert motivo["4_02_cam3_000007"] == "discrepancia_gt"
        multi = [s for s in ("3_13_cam0_000005", "3_13_cam0_000006") if s in motivo]
        assert len(multi) == 1 and motivo[multi[0]] == "multi_persona_muestra"
        assert sum(1 for m in motivo.values() if m == "control") == 2

        por_stem = {f["stem"]: f for f in sel}
        assert por_stem["1_05_cam1_000002"]["err_gt_px"] == "150.0"
        assert por_stem["1_05_cam1_000003"]["kpt_perdido"] == "1"
        assert por_stem["0_01_cam0_000001"]["err_gt_px"] == ""
        assert [f["stem"] for f in sel] == sorted(f["stem"] for f in sel)

        write_selection(sel, cfg.selection_csv)
        assert read_selection(cfg.selection_csv) == sel


def test_multi_incluidos_si_no_se_excluyen() -> None:
    filas = [(_row(f"{s}_01_cam0_{s:06d}", 2, 1), "ok") for s in range(3)]
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _dataset(Path(tmpdir), filas, control_size=0, excluir_multi=False)
        assert [f["motivo"] for f in select(cfg, workers=2)] == ["multi_persona"] * 3


def test_flag_inconsistente_aborta() -> None:
    row = _row("0_01_cam0_000001", 1, 1, mean=0.3)
    row["baja_confianza"] = "0"  # el manifest dice sano, las reglas dicen score bajo
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _dataset(Path(tmpdir), [(row, "ok")])
        try:
            select(cfg, workers=1)
        except ValueError as exc:
            assert "baja_confianza" in str(exc)
            return
    raise AssertionError("Un flag que no cuadra con los umbrales debe abortar.")


def test_img_size_distinto_aborta() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _dataset(Path(tmpdir), [(_row("0_01_cam0_000001", 1, 1), "ok")], img_size=[640, 480])
        try:
            select(cfg, workers=1)
        except ValueError as exc:
            assert "img_size" in str(exc)
            return
    raise AssertionError("gt.img_size que no coincide con la imagen debe abortar.")


def test_postura_desconocida_aborta() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _dataset(Path(tmpdir), [(_row("0_16_cam0_000001", 1, 1, mean=0.3), "ok")])
        try:
            select(cfg, workers=1)  # ejercicio 16 no está en el mapa
        except ValueError:
            return
    raise AssertionError("Un ejercicio sin postura mapeada debe abortar, no inventar estrato.")


def test_control_determinista_con_seed() -> None:
    filas = [(_row(f"{s}_01_cam0_{n:06d}", 1, 1), "ok") for s, n in ((0, 1), (1, 2), (2, 3))]
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _dataset(Path(tmpdir), filas, control_size=2)
        assert select(cfg, workers=2) == select(cfg, workers=2)


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
