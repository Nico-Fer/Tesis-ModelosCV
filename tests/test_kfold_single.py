"""Selección de fold único (``cross_validation: false``) en :mod:`src.training._kfold`.

Ejercita sólo :func:`~src.training._kfold._select_fold`, que no entrena ni abre
runs MLflow: opera sobre rutas. El módulo bajo test importa ``mlflow`` (está en
``requirements-base``), pero **ninguna** dependencia de framework ML (torch,
tensorflow), así que corre en cualquiera de los venvs.

Lo que justifica el test: ``_select_fold`` resuelve el fold **por nombre** en vez
de indexar la lista ordenada de :func:`~src.training._kfold.discover_fold_yamls`.
Con ≥10 folds el orden lexicográfico pone ``fold10`` antes que ``fold2``, y un
índice desalineado entrenaría sobre otro split sin ningún síntoma visible.

Ejecutar::

    python tests/test_kfold_single.py      # sin pytest (no hay suite aún)
    pytest tests/                           # si pytest está instalado
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.training._kfold import _select_fold, discover_fold_yamls
from src.utils import strict


def _make_dataset(tmp: Path, n_folds: int, stem: str = "pose_dataset_pseudo") -> Path:
    """Crea ``{stem}_fold{k}.yaml`` vacíos y devuelve la ruta base (inexistente)."""
    for k in range(n_folds):
        (tmp / f"{stem}_fold{k}.yaml").write_text("", encoding="utf-8")
    return tmp / f"{stem}.yaml"


def test_select_fold_returns_requested_fold() -> None:
    strict.set_smoke_mode(False)
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _make_dataset(Path(tmpdir), 5)
        folds = discover_fold_yamls(data)
        assert len(folds) == 5, folds
        for k in range(5):
            assert _select_fold(folds, data, k).name == f"pose_dataset_pseudo_fold{k}.yaml"


def test_select_fold_survives_lexicographic_order() -> None:
    """Con 11 folds, ``sorted()`` pone fold10 en la posición 2: el índice mentiría."""
    strict.set_smoke_mode(False)
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _make_dataset(Path(tmpdir), 11)
        folds = discover_fold_yamls(data)
        assert folds[2].name == "pose_dataset_pseudo_fold10.yaml", "premisa del test rota"
        assert _select_fold(folds, data, 2).name == "pose_dataset_pseudo_fold2.yaml"
        assert _select_fold(folds, data, 10).name == "pose_dataset_pseudo_fold10.yaml"


def test_select_fold_rejects_out_of_range() -> None:
    strict.set_smoke_mode(False)
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _make_dataset(Path(tmpdir), 5)
        folds = discover_fold_yamls(data)
        try:
            _select_fold(folds, data, 9)
        except strict.PipelineError as exc:
            assert "fold9" in str(exc), exc
            return
        raise AssertionError("_select_fold aceptó un fold inexistente")


def test_select_fold_degrades_in_smoke_mode() -> None:
    """En smoke, ``strict.fail`` es un warning: debe seguir devolviendo un fold usable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _make_dataset(Path(tmpdir), 5)
        folds = discover_fold_yamls(data)
        strict.set_smoke_mode(True)
        try:
            assert _select_fold(folds, data, 9).name == "pose_dataset_pseudo_fold0.yaml"
        finally:
            strict.set_smoke_mode(False)


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
