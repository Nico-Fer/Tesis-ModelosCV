"""Invariantes del formato de labels YOLO-pose generados por el pseudo-labeling.

Sólo ejercita las partes **sin dependencias de framework** de
:mod:`src.data.pseudo_label_vitpose` y :mod:`src.data.uco_to_yolo`, así que corre
en cualquiera de los venvs y **sin OpenMMLab instalado**.

Ojo con la cadena de imports: ``uco_to_yolo`` importa de
``src.inference.keypoint_schema``, lo que ejecuta ``src/inference/__init__.py`` y
arrastra ``numpy`` y ``cv2`` (vía ``normalizer``/``video_pipeline``). Ambos están
en los tres requirements, así que no hace falta nada extra — pero un venv recién
creado y vacío falla con ``ModuleNotFoundError: numpy`` hasta instalar las deps.
MediaPipe **no** se arrastra: ``src/models/blazepose.py`` lo importa en diferido.

Ejecutar::

    python tests/test_pseudo_label_format.py      # sin pytest (no hay suite aún)
    pytest tests/                                  # si pytest está instalado
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.data.pseudo_label_vitpose import (
    derive_visibility,
    pseudo_label_line,
    stratified_sample,
    verify_folds_against,
)
from src.data.uco_to_yolo import make_stem, parse_stem
from src.inference.keypoint_schema import NUM_COCO_KEYPOINTS

# 1 (clase) + 4 (bbox) + 17 × 3 (keypoints) = 56.
EXPECTED_TOKENS = 1 + 4 + NUM_COCO_KEYPOINTS * 3


def check_label_line(line: str) -> None:
    """Verifica todos los invariantes de una línea de label YOLO-pose.

    Es el chequeo que también sirve para auditar un dataset ya generado (ver
    :func:`check_label_dir`).

    Raises:
        AssertionError: si la línea viola algún invariante.
    """
    tokens = line.split()
    assert len(tokens) == EXPECTED_TOKENS, f"{len(tokens)} tokens, se esperaban {EXPECTED_TOKENS}"
    assert tokens[0] == "0", f"clase {tokens[0]!r}, se esperaba '0' (person)"

    for name, raw in zip(("cx", "cy", "w", "h"), tokens[1:5]):
        value = float(raw)
        assert 0.0 <= value <= 1.0, f"bbox.{name}={value} fuera de [0,1]"

    for idx in range(NUM_COCO_KEYPOINTS):
        px, py, vis = tokens[5 + idx * 3 : 8 + idx * 3]
        v = int(vis)
        assert v in (0, 1, 2), f"keypoint {idx}: v={v} fuera de {{0,1,2}}"
        x, y = float(px), float(py)
        assert 0.0 <= x <= 1.0, f"keypoint {idx}: px={x} fuera de [0,1]"
        assert 0.0 <= y <= 1.0, f"keypoint {idx}: py={y} fuera de [0,1]"
        if v == 0:
            assert x == 0.0 and y == 0.0, f"keypoint {idx}: v=0 pero coords ({x},{y}) != 0"


def check_label_dir(labels_dir: Path) -> tuple[int, int]:
    """Aplica :func:`check_label_line` a todos los ``.txt`` de un directorio.

    Returns:
        ``(archivos, lineas)`` verificados. Los ``.txt`` vacíos (imágenes sin
        persona detectada) son válidos y se cuentan sin líneas.
    """
    files = lines = 0
    for path in sorted(labels_dir.glob("*.txt")):
        files += 1
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                check_label_line(line)
                lines += 1
    return files, lines


def _kpts(n: int = NUM_COCO_KEYPOINTS) -> list[tuple[float, float]]:
    return [(10.0 + i, 20.0 + i) for i in range(n)]


# --------------------------------------------------------------------------- #
# Stems
# --------------------------------------------------------------------------- #


def test_make_stem_parse_stem_roundtrip() -> None:
    for subject, exercise, cam, frame in (("0", "01", "0", 0), ("26", "13", "4", 123456)):
        stem = make_stem(subject, exercise, cam, frame)
        parts = parse_stem(stem)
        assert (parts.subject, parts.exercise, parts.cam, parts.frame) == (
            subject,
            exercise,
            cam,
            frame,
        )


def test_make_stem_keeps_legacy_format() -> None:
    # El pool ya generado depende de este formato exacto; no puede cambiar.
    assert make_stem("11", "01", "0", 0) == "11_01_cam0_000000"


def test_parse_stem_rejects_invalid() -> None:
    for bad in ("basura", "11_01_0_000000", "11_01_cam0_abc", "", "11_01_cam0"):
        try:
            parse_stem(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_stem no lanzó para {bad!r}")


# --------------------------------------------------------------------------- #
# Visibilidad
# --------------------------------------------------------------------------- #


def test_derive_visibility_thresholds() -> None:
    scores = [0.0, 0.29, 0.3, 0.45, 0.59, 0.6, 0.99]
    assert derive_visibility(scores, 0.3, 0.6) == [0, 0, 1, 1, 1, 2, 2]


def test_derive_visibility_collapses_when_thresholds_equal() -> None:
    # Con vis_thr_low == vis_thr_high no se emite ningún v=1.
    assert derive_visibility([0.2, 0.5, 0.8], 0.5, 0.5) == [0, 2, 2]


# --------------------------------------------------------------------------- #
# Serialización
# --------------------------------------------------------------------------- #


def test_pseudo_label_line_token_count() -> None:
    line = pseudo_label_line((10, 20, 110, 220), _kpts(), [2] * 17, 640, 480)
    check_label_line(line)


def test_pseudo_label_line_uses_detector_bbox() -> None:
    # El bbox sale del detector, NO de la envolvente de los keypoints (que acá
    # están todos entre x=10..26, y=20..36).
    line = pseudo_label_line((100, 200, 300, 400), _kpts(), [2] * 17, 1000, 1000)
    cx, cy, w, h = (float(t) for t in line.split()[1:5])
    assert (cx, cy, w, h) == (0.2, 0.3, 0.2, 0.2)


def test_pseudo_label_line_clamps_out_of_frame() -> None:
    # ViTPose puede extrapolar fuera del recorte: coords negativas o > ancho.
    kpts = [(-50.0, -50.0)] * 8 + [(9999.0, 9999.0)] * 9
    line = pseudo_label_line((-10, -10, 700, 500), kpts, [2] * 17, 640, 480)
    check_label_line(line)  # el clamp a [0,1] es parte del invariante


def test_pseudo_label_line_zeroes_invisible_coords() -> None:
    visibility = [0] * 17
    visibility[5] = 2
    line = pseudo_label_line((0, 0, 100, 100), _kpts(), visibility, 640, 480)
    tokens = line.split()
    check_label_line(line)
    # El slot visible conserva su coordenada; el resto se anula.
    assert float(tokens[5 + 5 * 3]) > 0.0
    assert tokens[5 : 5 + 3] == ["0.000000", "0.000000", "0"]


def test_pseudo_label_line_matches_ground_truth_shape() -> None:
    """El label pseudo debe ser intercambiable con el del converter existente."""
    gt_dir = Path("data/processed/uco_yolo/labels/all")
    if not gt_dir.exists():
        print("  (omitido: no hay dataset GT local)")
        return
    sample = next(iter(sorted(gt_dir.glob("*.txt"))), None)
    if sample is None:
        print("  (omitido: no hay labels GT)")
        return
    gt_line = sample.read_text(encoding="utf-8").splitlines()[0]
    pseudo = pseudo_label_line((10, 20, 110, 220), _kpts(), [2] * 17, 1280, 720)
    assert len(gt_line.split()) == len(pseudo.split()) == EXPECTED_TOKENS


# --------------------------------------------------------------------------- #
# Muestreo estratificado
# --------------------------------------------------------------------------- #


def test_stratified_sample_is_deterministic_and_balanced() -> None:
    stems = [
        make_stem(subject, exercise, "0", frame)
        for subject in ("0", "1", "2")
        for exercise in ("01", "05")
        for frame in range(20)
    ]
    first = stratified_sample(stems, ("exercise", "subject"), 30, seed=42)
    second = stratified_sample(stems, ("exercise", "subject"), 30, seed=42)
    assert first == second, "el muestreo debe ser reproducible con el mismo seed"
    assert len(first) == 30
    # 6 grupos (3 sujetos × 2 ejercicios) repartidos round-robin ⇒ 5 por grupo.
    per_group: dict[tuple[str, str], int] = {}
    for stem in first:
        parts = parse_stem(stem)
        key = (parts.exercise, parts.subject)
        per_group[key] = per_group.get(key, 0) + 1
    assert set(per_group.values()) == {5}, per_group


def test_stratified_sample_handles_size_over_population() -> None:
    stems = [make_stem("0", "01", "0", f) for f in range(5)]
    assert len(stratified_sample(stems, ("subject",), 100, seed=1)) == 5


# --------------------------------------------------------------------------- #
# Verificación de folds
# --------------------------------------------------------------------------- #


def _write_val_manifests(root: Path, subjects_by_fold: dict[int, list[str]]) -> None:
    (root / "folds").mkdir(parents=True, exist_ok=True)
    for k, subjects in subjects_by_fold.items():
        lines = [
            str(root / "images" / "all" / f"{make_stem(s, '01', '0', 0)}.jpg") for s in subjects
        ]
        (root / "folds" / f"fold{k}_val.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_verify_folds_accepts_identical_split() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gt, pseudo = Path(tmp) / "gt", Path(tmp) / "pseudo"
        split = {0: ["0", "1"], 1: ["2", "3"]}
        _write_val_manifests(gt, split)
        _write_val_manifests(pseudo, split)
        verify_folds_against(pseudo, gt, n_folds=2)  # no debe lanzar


def test_verify_folds_rejects_divergent_split() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gt, pseudo = Path(tmp) / "gt", Path(tmp) / "pseudo"
        _write_val_manifests(gt, {0: ["0", "1"], 1: ["2", "3"]})
        _write_val_manifests(pseudo, {0: ["0", "2"], 1: ["1", "3"]})
        try:
            verify_folds_against(pseudo, gt, n_folds=2)
        except RuntimeError as exc:
            assert "fold0" in str(exc)
            return
        raise AssertionError("verify_folds_against no detectó el reparto divergente")


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
