"""Checkpoints reanudables de la CV (:mod:`src.utils.checkpoints`).

Cubre lo que decide si una reanudación tras un corte de luz es correcta:

* la huella de config: reanudar con otra receta debe abortar (mezclaría folds
  entrenados distinto en una misma media ± σ), salvo ``workers``, que sólo cambia la
  velocidad de carga;
* el estado por fold y por etapa sobrevive a reabrirlo desde disco;
* la tabla de avance se reescribe **sólo** entre sus marcadores (el resto del
  Markdown es texto escrito a mano);
* :func:`~src.utils.checkpoints.training_resume_point` distingue entrenamiento
  nuevo / a mitad / terminado a partir del ``last.pt`` de Ultralytics. Esos tests
  necesitan ``torch`` (venv-torch); sin él se saltean.

Ejecutar::

    python tests/test_checkpoints.py      # sin pytest (no hay suite aún)
    pytest tests/                          # si pytest está instalado
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.models.base import ModelMetrics
from src.utils import checkpoints
from src.utils.strict import PipelineError

_CONFIG = {"family": "yolo", "model": "m.pt", "workers": 2, "pruning": {"sparsity": 0.3}}


def _config(tmp: Path, **overrides: object) -> dict[str, object]:
    return {**_CONFIG, "checkpoint_dir": str(tmp / "cv"), **overrides}


def test_fingerprint_ignores_workers_only() -> None:
    base = checkpoints.config_fingerprint(_CONFIG)
    assert checkpoints.config_fingerprint({**_CONFIG, "workers": 8}) == base
    assert checkpoints.config_fingerprint({**_CONFIG, "pruning": {"sparsity": 0.5}}) != base


def test_resume_with_other_config_aborts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        checkpoints.CvCheckpoint.open(_config(tmp))
        checkpoints.CvCheckpoint.open(_config(tmp, workers=4))  # workers: permitido
        try:
            checkpoints.CvCheckpoint.open(_config(tmp, pruning={"sparsity": 0.5}))
        except PipelineError:
            return
        raise AssertionError("reanudar con otra receta no abortó")


def test_fold_state_survives_reopen() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cv = checkpoints.CvCheckpoint.open(_config(tmp))
        cv.set_parent_run("parent123")
        cv.set_fold_run(0, "fold0run", "fold0.yaml")
        cv.mark_fold_done(0, ModelMetrics(oks=0.9, size_mb=2.4))
        cv.set_fold_run(1, "fold1run", "fold1.yaml")

        again = checkpoints.CvCheckpoint.open(_config(tmp))
        assert again.parent_run_id == "parent123"
        assert again.fold_done(0) and not again.fold_done(1)
        assert again.fold_run_id(1) == "fold1run"
        assert again.fold_metrics(0) == ModelMetrics(oks=0.9, size_mb=2.4)


def test_stage_state_and_progress() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fold = checkpoints.FoldCheckpoint(Path(tmpdir) / "fold0")
        fold.progress("pruned_ft", epoch=3, epochs=10, save_dir="x")
        assert not fold.done("pruned_ft")
        fold.mark(
            "pruned_ft",
            metrics=ModelMetrics(oks=0.95),
            logged={"oks_pruned_ft": 0.95},
            pt="ft.pt",
        )
        reopened = checkpoints.FoldCheckpoint(Path(tmpdir) / "fold0")
        assert reopened.done("pruned_ft")
        assert reopened.metrics("pruned_ft").oks == 0.95
        assert reopened.artifact("pruned_ft", "pt") == "ft.pt"
        assert "progress" not in reopened.get("pruned_ft")  # el avance no queda colgado


def test_eval_ctx_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fold = checkpoints.FoldCheckpoint(Path(tmpdir) / "fold0")
        assert fold.load_ctx() is None
        fold.save_ctx({"img": [1, 2, 3]})
        assert fold.load_ctx() == {"img": [1, 2, 3]}


def test_report_rewrites_only_between_markers() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        report = tmp / "bitacora.md"
        report.write_text("# Bitácora\n\nTexto a mano.\n", encoding="utf-8")
        cv = checkpoints.CvCheckpoint.open(_config(tmp, checkpoint_report=str(report)))
        cv.set_fold_run(0, "abcdef123456", "fold0.yaml")
        fold = checkpoints.FoldCheckpoint(cv.fold_dir(0))
        fold.mark("base", metrics=ModelMetrics(oks=0.85), logged={"oks_base": 0.85})
        fold.progress("pruned_ft", epoch=2, epochs=10, save_dir="x")

        text = report.read_text(encoding="utf-8")
        assert text.startswith("# Bitácora\n\nTexto a mano.\n"), text[:80]
        assert text.count("CHECKPOINTS:BEGIN") == 1 and text.count("CHECKPOINTS:END") == 1
        assert "✅ 0.8500" in text and "⏳ ep 2/10" in text, text
        assert "0.8500 ± 0.0000 (n=1)" in text, text

        # Texto agregado a mano DESPUÉS de la tabla sobrevive a una nueva escritura.
        report.write_text(text + "\n## Hallazgos\n\nNota.\n", encoding="utf-8")
        fold.mark("pruned", metrics=ModelMetrics(oks=0.0), logged={"oks_pruned": 0.0})
        text = report.read_text(encoding="utf-8")
        assert text.endswith("## Hallazgos\n\nNota.\n"), text[-80:]
        assert text.count("CHECKPOINTS:BEGIN") == 1


def _torch():  # noqa: ANN202
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return torch


def test_training_resume_point_states() -> None:
    torch = _torch()
    if torch is None:
        print("  (sin torch: se saltea)")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "distill_student"

        # Dir a medio crear (sin epoch completo): se borra y se arranca de cero.
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "results.csv").write_text("epoch\n", encoding="utf-8")
        assert checkpoints.training_resume_point(run_dir) == ("fresh", None)
        assert not run_dir.exists()

        # last.pt con optimizer: entrenamiento a mitad -> se reanuda.
        (run_dir / "weights").mkdir(parents=True)
        last = run_dir / "weights" / "last.pt"
        torch.save({"epoch": 4, "optimizer": {"state": {}}}, str(last))
        assert checkpoints.training_resume_point(run_dir) == ("resume", last)

        # Optimizer quitado (strip_optimizer al final de train): terminado -> best.pt.
        torch.save({"epoch": -1, "optimizer": None}, str(last))
        best = run_dir / "weights" / "best.pt"
        best.write_bytes(b"x")
        assert checkpoints.training_resume_point(run_dir) == ("finished", best)

        # last.pt truncado (corte mientras se escribía): aborta sin borrar nada.
        last.write_bytes(b"\x80\x02corrupto")
        try:
            checkpoints.training_resume_point(run_dir)
        except PipelineError:
            assert last.exists()
            return
        raise AssertionError("un last.pt ilegible no abortó")


def test_fp32_optimizer_mixin_skips_fp16_conversion() -> None:
    try:
        import ultralytics.engine.trainer as trainer_module  # noqa: PLC0415
    except ImportError:
        print("  (sin ultralytics: se saltea)")
        return
    import torch  # noqa: PLC0415

    seen: dict[str, object] = {}

    class FakeTrainer:
        def save_model(self) -> None:
            # Lo mismo que hace BaseTrainer.save_model con el state_dict del optimizer.
            state = {"state": {0: {"step": 1, "exp_avg_sq": torch.full((2,), 1e-9)}}}
            seen["out"] = trainer_module.convert_optimizer_state_dict_to_fp16(state)

    class Mixed(checkpoints.Fp32OptimizerCheckpointMixin, FakeTrainer):
        pass

    original = trainer_module.convert_optimizer_state_dict_to_fp16
    # Premisa: la conversión de Ultralytics lleva 1e-9 a 0 (por eso diverge al reanudar).
    FakeTrainer().save_model()
    premise = seen["out"]["state"][0]["exp_avg_sq"]  # type: ignore[index]
    assert premise.dtype == torch.float16 and float(premise.max()) == 0.0, premise

    Mixed().save_model()
    kept = seen["out"]["state"][0]["exp_avg_sq"]  # type: ignore[index]
    assert kept.dtype == torch.float32 and float(kept.min()) > 0.0, kept
    assert trainer_module.convert_optimizer_state_dict_to_fp16 is original  # restaurada


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
