"""Config de la etapa de distillation (:mod:`src.optimization.distillation`).

Cubre las dos reglas que corrigieron la divergencia de la Fase 5 y que no dependen de
entrenar nada:

* **LR de la etapa = ``lr0/10``** (o ``distillation.lr0`` explícito). Antes se reenviaba
  el ``lr0`` general y un estudiante ya entrenado se reiniciaba a AdamW 0.01. El warmup
  de los bias arranca en 0.0 (``distillation.warmup_bias_lr``), no en el 0.1 de Ultralytics.
* **``distillation.temperature`` ya no existe** y aborta: T² no aplica a attention
  transfer, y aceptarla la dejaría logueada en MLflow sin efecto. Un typo también aborta.

Framework-free (el módulo difiere torch/ultralytics a las funciones que entrenan), así
que corre en cualquier venv::

    python tests/test_distillation_config.py   # PYTHONPATH en la raíz del repo
"""

from __future__ import annotations

from src.optimization.distillation import build_distillation_loss, distillation_lr0
from src.utils import strict


def _raises_pipeline_error(config: dict) -> bool:
    strict.set_smoke_mode(False)
    try:
        build_distillation_loss(config)
    except strict.PipelineError:
        return True
    return False


def test_lr_is_tenth_of_general_lr0() -> None:
    assert distillation_lr0({"lr0": 0.01, "distillation": {}}) == 0.001


def test_explicit_distillation_lr0_wins() -> None:
    assert distillation_lr0({"lr0": 0.01, "distillation": {"lr0": 0.0005}}) == 0.0005


def test_no_lr0_leaves_it_to_ultralytics() -> None:
    assert distillation_lr0({"distillation": {"alpha": 0.5}}) is None


def test_logged_params_are_the_effective_ones() -> None:
    params = build_distillation_loss({"lr0": 0.01, "distillation": {"alpha": 0.5}})
    assert params == {
        "distillation.alpha": 0.5,
        "distillation.beta": 1000.0,
        "distillation.warmup_bias_lr": 0.0,
        "distillation.lr0_efectivo": 0.001,
    }, params


def test_warmup_bias_lr_is_overridable() -> None:
    """Default 0.0 (no el 0.1 de Ultralytics), pero declarable en el YAML."""
    params = build_distillation_loss({"distillation": {"warmup_bias_lr": 0.05}})
    assert params["distillation.warmup_bias_lr"] == 0.05, params


def test_temperature_is_rejected() -> None:
    assert _raises_pipeline_error({"distillation": {"temperature": 4.0, "alpha": 0.5}})


def test_typo_is_rejected() -> None:
    assert _raises_pipeline_error({"distillation": {"alpah": 0.5}})


def test_valid_section_passes() -> None:
    cfg = {"distillation": {"alpha": 0.5, "beta": 1000, "epochs": 20, "val": False}}
    assert not _raises_pipeline_error(cfg)


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
