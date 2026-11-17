"""Reconciliación y round-trip (:mod:`src.audit.reconcile_corrections` y
:mod:`src.audit.verify_roundtrip`).

Ejercita la lógica pura sobre labels sintéticos: matching de personas,
conteo de correcciones según las definiciones de la métrica, reconstrucción
de las líneas gold (bbox hull + 5 %, byte-compatible con el GT), el
comparador del gate y ``reconcile()`` de punta a punta sobre un export
sintético (auditado = ``auditados.txt``, no presencia en el export).
Framework-free: corre en cualquier venv.

Ejecutar::

    python tests/test_audit_reconcile.py    # sin pytest
    pytest tests/                            # si pytest está instalado
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from src.audit.common import PoseLabel, match_persons, parse_label_line
from src.audit.reconcile_corrections import gold_label_lines, reconcile_frame
from src.audit.verify_roundtrip import compare_frame
from src.inference.keypoint_schema import NUM_COCO_KEYPOINTS
from tests.test_pseudo_label_format import check_label_line

_W, _H = 1280, 720


def _persona(base_x: float, base_y: float, vis: int = 2) -> PoseLabel:
    """Persona sintética: 17 kpts en línea diagonal desde ``(base_x, base_y)``."""
    kpts = tuple((base_x + 0.01 * k, base_y + 0.01 * k, vis) for k in range(NUM_COCO_KEYPOINTS))
    bbox = (base_x + 0.08, base_y + 0.08, 0.2, 0.2)
    return PoseLabel(bbox=bbox, kpts=kpts)


def _mover_kpt(p: PoseLabel, idx: int, dx: float = 0.0, dy: float = 0.0, v: int | None = None):
    kpts = list(p.kpts)
    x, y, vis = kpts[idx]
    kpts[idx] = (x + dx, y + dy, vis if v is None else v)
    return PoseLabel(bbox=p.bbox, kpts=tuple(kpts))


def test_match_persons_sobrevive_reordenamiento() -> None:
    a1, a2 = _persona(0.1, 0.1), _persona(0.6, 0.6)
    pares, sin_a, sin_b = match_persons([a1, a2], [a2, a1])
    assert sorted(pares) == [(0, 1), (1, 0)]
    assert sin_a == [] and sin_b == []


def test_match_persons_detecta_altas_y_bajas() -> None:
    pares, sin_a, sin_b = match_persons(
        [_persona(0.1, 0.1), _persona(0.6, 0.6)], [_persona(0.1, 0.1)]
    )
    assert len(pares) == 1 and sin_a == [1] and sin_b == []
    pares, sin_a, sin_b = match_persons([], [_persona(0.1, 0.1)])
    assert pares == [] and sin_a == [] and sin_b == [0]


def test_reconcile_cuenta_solo_movimientos_sobre_umbral() -> None:
    orig = _persona(0.1, 0.1)
    # kpt 0 movido 12.8 px (> thr 3), kpt 1 movido ~1.3 px (< thr): solo 1 corregido.
    corr = _mover_kpt(_mover_kpt(orig, 0, dx=0.01), 1, dx=0.001)
    res = reconcile_frame([orig], [corr], _W, _H, thr_px=3.0)
    assert res.kpts_auditados == NUM_COCO_KEYPOINTS
    assert res.kpts_movidos == 1 and res.kpts_corregidos == 1
    assert len(res.despl_px) == 1 and abs(res.despl_px[0] - 12.8) < 0.1


def test_reconcile_agregados_eliminados_y_flips() -> None:
    orig = _mover_kpt(_persona(0.1, 0.1), 0, v=0)  # kpt 0 no anotado en el pseudo
    corr = _mover_kpt(_mover_kpt(_mover_kpt(_persona(0.1, 0.1), 0, v=2), 1, v=0), 2, v=1)
    res = reconcile_frame([orig], [corr], _W, _H, thr_px=3.0)
    assert res.kpts_agregados == 1  # kpt 0: v 0→2
    assert res.kpts_eliminados == 1  # kpt 1: v 2→0
    assert res.cambios_v1v2 == 1  # kpt 2: v 2→1, NO cuenta como corregido
    assert res.kpts_corregidos == 2


def test_reconcile_personas_agregadas_de_cero() -> None:
    res = reconcile_frame([], [_persona(0.3, 0.3)], _W, _H, thr_px=3.0)
    assert res.personas_agregadas == 1
    assert res.kpts_anotados_de_cero == NUM_COCO_KEYPOINTS
    assert res.kpts_auditados == 0  # sin par: no hay pseudo contra el cual medir


def test_gold_lines_cumplen_invariantes_y_zeroan_v0() -> None:
    persona = _mover_kpt(_persona(0.3, 0.3), 5, dx=0.05, v=0)
    lines = gold_label_lines([persona], _W, _H)
    assert len(lines) == 1
    check_label_line(lines[0])  # 56 tokens, coords ∈ [0,1], v=0 ⇒ coords 0
    kpt5 = parse_label_line(lines[0]).kpts[5]
    assert kpt5 == (0.0, 0.0, 0), "v=0 debe zerear coordenadas aunque CVAT las exporte"


def test_gold_descarta_persona_sin_kpts_visibles() -> None:
    invisible = PoseLabel(bbox=(0.5, 0.5, 0.1, 0.1), kpts=((0.0, 0.0, 0),) * NUM_COCO_KEYPOINTS)
    assert gold_label_lines([invisible], _W, _H) == []


def test_compare_frame_acepta_identicos_y_rechaza_flip() -> None:
    p = _persona(0.2, 0.2)
    errores, delta = compare_frame("x", [p], [p], _W, _H, tol_px=1.0)
    assert errores == [] and delta == 0.0

    flip = _mover_kpt(p, 3, v=1)
    errores, _ = compare_frame("x", [p], [flip], _W, _H, tol_px=1.0)
    assert any("visibilidad" in e for e in errores)


def test_compare_frame_mide_noise_floor() -> None:
    p = _persona(0.2, 0.2)
    ruido = _mover_kpt(p, 0, dx=0.0005)  # 0.64 px < tol 1.0: pasa pero se mide
    errores, delta = compare_frame("x", [p], [ruido], _W, _H, tol_px=1.0)
    assert errores == []
    assert 0.5 < delta < 0.8, delta


# --------------------------------------------------------------------------- #
# reconcile() de punta a punta: auditado = auditados.txt, no presencia en el export
# --------------------------------------------------------------------------- #

_STEMS = ("0_01_cam0_000000", "1_02_cam1_000030", "2_09_cam2_000060")


def _linea(p: PoseLabel) -> str:
    kpts = " ".join(f"{x:.6f} {y:.6f} {v}" for x, y, v in p.kpts)
    return f"0 {p.bbox[0]:.6f} {p.bbox[1]:.6f} {p.bbox[2]:.6f} {p.bbox[3]:.6f} {kpts}"


def _escenario(tmp: Path, auditados: list[str], exportados: dict[str, PoseLabel]):
    """Pool pseudo de 3 frames + selección + export sintético."""
    import cv2
    import numpy as np

    from src.audit.common import SELECTION_FIELDS, build_config

    img_dir, lbl_dir = tmp / "images", tmp / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    for stem in _STEMS:
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), np.zeros((_H, _W, 3), dtype=np.uint8))
        (lbl_dir / f"{stem}.txt").write_text(_linea(_persona(0.1, 0.1)) + "\n", encoding="utf-8")

    sel = tmp / "selection.csv"
    with sel.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        for stem in _STEMS:
            subject, exercise, cam, frame = stem.split("_")
            writer.writerow(
                {k: "0" for k in SELECTION_FIELDS}
                | {
                    "stem": stem,
                    "motivo": "score_bajo",
                    "postura": "sedente",
                    "subject": subject,
                    "exercise": exercise,
                    "cam": cam[3:],
                    "frame": frame,
                    "err_gt_px": "",
                }
            )

    export = tmp / "export"
    (export / "labels").mkdir(parents=True)
    (export / "auditados.txt").write_text("".join(f"{s}\n" for s in auditados), encoding="utf-8")
    for stem, persona in exportados.items():
        (export / "labels" / f"{stem}.txt").write_text(_linea(persona) + "\n", encoding="utf-8")

    cfg = build_config(
        {
            "pseudo": {"images_dir": str(img_dir), "labels_dir": str(lbl_dir)},
            "seleccion": {"out_csv": str(sel)},
            "estratos": {"postura": {"01": "sedente"}},
            "pack": {"link_mode": "copy"},
            "reconcile": {"export_dir": str(export), "gold_root": str(tmp / "gold")},
        }
    )
    return cfg, export


def test_reconcile_usa_auditados_y_no_la_presencia_en_el_export() -> None:
    from src.audit.reconcile_corrections import reconcile

    s0, s1, s2 = _STEMS
    corregida = _mover_kpt(_persona(0.1, 0.1), 0, dx=0.05)  # 64 px
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # s0: auditado y corregido; s1: en el export pero su job NO se completó;
        # s2: auditado y el anotador borró todo (no aparece en el export).
        cfg, export = _escenario(
            tmp, auditados=[s0, s2], exportados={s0: corregida, s1: _persona(0.1, 0.1)}
        )
        report = reconcile(cfg, export)

        assert report["n_frames_auditados"] == 2
        assert report["n_frames_sin_auditar"] == 1
        assert report["n_frames_auditados_sin_personas"] == 1
        assert report["global"]["kpts_movidos"] == 1
        assert report["global"]["personas_eliminadas"] == 1

        gold_lbl = cfg.gold_root / "labels" / "all"
        assert sorted(p.stem for p in gold_lbl.glob("*.txt")) == [s0, s2]
        assert (gold_lbl / f"{s2}.txt").read_text(encoding="utf-8") == ""
        stems_gold = (cfg.gold_root / "gold_stems.txt").read_text(encoding="utf-8").split()
        assert stems_gold == [s0, s2]


def test_reconcile_reconstruye_el_gold_de_cero() -> None:
    from src.audit.reconcile_corrections import reconcile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg, export = _escenario(tmp, auditados=[_STEMS[0]], exportados={})
        viejo = cfg.gold_root / "labels" / "all" / "9_01_cam0_000000.txt"
        viejo.parent.mkdir(parents=True)
        viejo.write_text("", encoding="utf-8")
        reconcile(cfg, export)
        assert not viejo.exists(), "un frame de una corrida anterior no puede quedar en el gold"


def test_reconcile_sin_auditados_txt_aborta() -> None:
    from src.audit.reconcile_corrections import reconcile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg, export = _escenario(Path(tmpdir), auditados=[], exportados={})
        (export / "auditados.txt").unlink()
        try:
            reconcile(cfg, export)
        except FileNotFoundError:
            return
    raise AssertionError("Sin auditados.txt no se sabe qué se revisó: debe abortar.")


def test_reconcile_stem_ajeno_aborta() -> None:
    from src.audit.reconcile_corrections import reconcile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg, export = _escenario(Path(tmpdir), auditados=["9_01_cam0_000000"], exportados={})
        try:
            reconcile(cfg, export)
        except ValueError:
            return
    raise AssertionError("Un stem auditado fuera de la selección indica otro proyecto.")


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
