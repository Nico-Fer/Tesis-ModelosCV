"""Checkpoints reanudables para corridas largas (validación cruzada de la Fase 5).

Motivación
----------
La CV de la Fase 5 son 5 folds × (poda → fine-tune → distillation → INT8) y dura
días. Sin checkpoints, un corte de luz obliga a repetir el fold en curso (o la CV
entera: el run padre se cierra y la agregación media ± σ se pierde). Este módulo
persiste el avance en tres niveles, de grueso a fino:

1. **Fold** (:class:`CvCheckpoint`): qué folds terminaron, con sus métricas y el
   ``run_id`` MLflow (padre y anidado) para reabrir los mismos runs al reanudar.
2. **Etapa** (:class:`FoldCheckpoint`): qué etapas de un fold ya se evaluaron y
   loguearon, con el artefacto que cada una deja en disco para la siguiente.
3. **Epoch** (:func:`training_resume_point`): el ``last.pt`` de Ultralytics que el
   fine-tune y la distillation escriben al final de cada epoch. Con él se reanuda
   el entrenamiento (``resume=True``) en vez de reiniciarlo.

Es **opt-in**: sólo se activa con ``checkpoint_dir`` en la config. Sin esa clave
todo el pipeline se comporta exactamente como antes.

Layout en disco::

    <checkpoint_dir>/
        cv_state.json                  # huella de la config, run padre, estado por fold
        fold{k}/
            stages.json                # etapas terminadas + progreso por epoch
            eval_ctx.pkl               # contexto de evaluación (GT + bbox normalizador)
            student_pruned.pt          # estudiante recién podado
            student_pruned_ft.pt       # podado + fine-tune
            pruned_finetune/           # run Ultralytics del fine-tune
            distillation/distill_student/  # run Ultralytics de la distillation (+ INT8)

Los JSON son la fuente de verdad y se escriben de forma atómica (``os.replace``).
Opcionalmente (``checkpoint_report``) se reescribe una tabla de avance en un
Markdown, entre marcadores, para seguir la corrida sin abrir MLflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.models.base import ModelMetrics
from src.utils.strict import PipelineError

logger = logging.getLogger(__name__)

# Claves que pueden cambiar entre una corrida y su reanudación sin invalidarla: sólo
# afectan la velocidad de carga de datos, no el modelo. Bajar ``workers`` tras un
# WinError 1455 (archivo de paginación agotado) no debe obligar a tirar días de cómputo.
_FINGERPRINT_IGNORED = frozenset({"workers"})

# Orden de las etapas de la Fase 5, para la tabla de avance.
STAGES = ("base", "pruned", "pruned_ft", "distilled", "int8")

_REPORT_BEGIN = "<!-- CHECKPOINTS:BEGIN (generado por src/utils/checkpoints.py, no editar) -->"
_REPORT_END = "<!-- CHECKPOINTS:END -->"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escritura atómica: un corte a mitad de escritura deja el JSON anterior intacto."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def config_fingerprint(config: dict[str, Any]) -> str:
    """Huella de la config (sin las claves de :data:`_FINGERPRINT_IGNORED`)."""
    relevant = {k: v for k, v in config.items() if k not in _FINGERPRINT_IGNORED}
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class CvCheckpoint:
    """Estado de una validación cruzada a nivel fold (``cv_state.json``)."""

    def __init__(self, root: Path, state: dict[str, Any]) -> None:
        self.root = root
        self._state = state

    @classmethod
    def open(cls, config: dict[str, Any]) -> CvCheckpoint:
        """Abre (o crea) el estado de la CV en ``config["checkpoint_dir"]``.

        Raises:
            PipelineError: si el directorio ya tiene una CV hecha con **otra** config.
                Reanudar mezclaría folds entrenados con recetas distintas en una
                misma media ± σ, sin rastro en MLflow.
        """
        root = Path(config["checkpoint_dir"])
        fingerprint = config_fingerprint(config)
        state = _read_json(root / "cv_state.json")
        if state and state.get("fingerprint") != fingerprint:
            raise PipelineError(
                f"{root} tiene una validación cruzada con otra config (huella "
                f"{state.get('fingerprint')} != {fingerprint}). Revertí la config para "
                "reanudarla, o usá otro 'checkpoint_dir' para empezar una nueva."
            )
        if not state:
            state = {"fingerprint": fingerprint, "created": _now(), "folds": {}}
            logger.info("Checkpoints de CV nuevos en %s.", root)
        else:
            done = sum(1 for f in state["folds"].values() if f.get("done"))
            logger.info("Reanudando CV desde %s (%d folds terminados).", root, done)
        state["report"] = config.get("checkpoint_report")
        ckpt = cls(root, state)
        ckpt._save()
        return ckpt

    def _save(self) -> None:
        _write_json(self.root / "cv_state.json", self._state)
        render_report(self.root)

    def _fold(self, k: int) -> dict[str, Any]:
        return self._state["folds"].setdefault(str(k), {})

    @property
    def parent_run_id(self) -> str | None:
        return self._state.get("parent_run_id")

    def set_parent_run(self, run_id: str) -> None:
        self._state["parent_run_id"] = run_id
        self._save()

    def fold_dir(self, k: int) -> Path:
        return self.root / f"fold{k}"

    def fold_run_id(self, k: int) -> str | None:
        return self._fold(k).get("run_id")

    def set_fold_run(self, k: int, run_id: str, fold_yaml: str) -> None:
        fold = self._fold(k)
        fold.update(run_id=run_id, data=fold_yaml)
        fold.setdefault("started", _now())
        self._save()

    def fold_done(self, k: int) -> bool:
        return bool(self._fold(k).get("done"))

    def fold_metrics(self, k: int) -> ModelMetrics:
        return ModelMetrics(**self._fold(k).get("metrics", {}))

    def mark_fold_done(self, k: int, metrics: ModelMetrics) -> None:
        self._fold(k).update(done=True, finished=_now(), metrics=asdict(metrics))
        self._save()

    def mark_finished(self) -> None:
        self._state["finished"] = _now()
        self._save()


class FoldCheckpoint:
    """Estado de las etapas de un fold (``fold{k}/stages.json``).

    Se reconstruye desde la ruta en cada uso (todo el estado vive en disco), así
    que los módulos de optimización lo obtienen de la config sin que el runner se
    lo pase explícitamente.
    """

    def __init__(self, fold_dir: Path) -> None:
        self.dir = fold_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._file = fold_dir / "stages.json"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FoldCheckpoint | None:
        """El checkpoint del fold, o ``None`` si la config no declara ``checkpoint_dir``."""
        fold_dir = config.get("checkpoint_dir")
        return cls(Path(fold_dir)) if fold_dir else None

    def _load(self) -> dict[str, Any]:
        return _read_json(self._file)

    def _save(self, state: dict[str, Any]) -> None:
        _write_json(self._file, state)
        render_report(self.dir.parent)

    def done(self, stage: str) -> bool:
        return bool(self._load().get(stage, {}).get("done"))

    def get(self, stage: str) -> dict[str, Any]:
        return self._load().get(stage, {})

    def metrics(self, stage: str) -> ModelMetrics:
        return ModelMetrics(**self.get(stage).get("metrics", {}))

    def mark(
        self,
        stage: str,
        *,
        metrics: ModelMetrics,
        logged: dict[str, float] | None = None,
        **artifacts: str,
    ) -> None:
        """Registra una etapa como terminada (ya logueada en MLflow).

        Args:
            stage: nombre de la etapa.
            metrics: las :class:`ModelMetrics` que la etapa aporta a ``final_metrics``.
            logged: las métricas sufijadas que la etapa logueó (``oks_pruned``, …),
                sólo para la tabla de avance.
            **artifacts: rutas que la etapa siguiente necesita (p. ej. ``pt=...``).
        """
        state = self._load()
        entry = state.setdefault(stage, {})
        entry.update(
            done=True,
            finished=_now(),
            metrics=asdict(metrics),
            logged=logged or {},
            artifacts={k: str(v) for k, v in artifacts.items()},
        )
        entry.pop("progress", None)
        self._save(state)

    def progress(self, stage: str, *, epoch: int, epochs: int, save_dir: str) -> None:
        """Registra el último epoch guardado de una etapa de entrenamiento."""
        state = self._load()
        entry = state.setdefault(stage, {})
        entry.setdefault("started", _now())
        entry["progress"] = {"epoch": epoch, "epochs": epochs, "save_dir": save_dir, "t": _now()}
        self._save(state)

    def artifact(self, stage: str, name: str) -> str:
        return self.get(stage)["artifacts"][name]

    def save_ctx(self, ctx: Any) -> None:  # noqa: ANN401
        """Persiste el contexto de evaluación del fold.

        Reusarlo al reanudar no es sólo por tiempo: el bbox de persona normalizador
        sale de inferencia en GPU, y recalcularlo podría moverlo levemente entre
        etapas del mismo fold, rompiendo la comparabilidad que el contexto garantiza.
        """
        tmp = self.dir / "eval_ctx.pkl.tmp"
        with tmp.open("wb") as fh:
            pickle.dump(ctx, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.dir / "eval_ctx.pkl")

    def load_ctx(self) -> Any | None:  # noqa: ANN401
        path = self.dir / "eval_ctx.pkl"
        if not path.exists():
            return None
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 — archivo propio de esta corrida


def training_resume_point(run_dir: Path) -> tuple[str, Path | None]:
    """Decide cómo arrancar un entrenamiento Ultralytics con run dir fijo.

    Args:
        run_dir: directorio del run (``…/pruned_finetune`` o ``…/distill_student``).

    Returns:
        ``("fresh", None)`` si no hay ningún epoch completo (un dir a medio crear se
        borra: no tiene nada que rescatar y Ultralytics le *anexaría* el
        ``results.csv``); ``("resume", last.pt)`` si quedó a mitad; ``("finished",
        best.pt)`` si terminó (al final, Ultralytics quita el optimizer de
        ``last.pt``, y reanudar un entrenamiento terminado aborta).

    Raises:
        PipelineError: si ``last.pt`` existe pero no se puede leer (corte justo
            mientras se escribía). No se borra solo: puede ser un bloqueo
            transitorio y el dir contiene epochs de trabajo.
    """
    import torch  # noqa: PLC0415

    last = run_dir / "weights" / "last.pt"
    if not last.exists():
        if run_dir.exists():
            logger.info("%s sin epochs completos: se reinicia la etapa.", run_dir)
            shutil.rmtree(run_dir)
        return "fresh", None
    try:
        ckpt = torch.load(str(last), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(
            f"No se puede leer {last} ({exc}); probablemente el corte ocurrió mientras se "
            f"escribía. Borrá {run_dir} para reiniciar sólo esta etapa."
        ) from exc
    if ckpt.get("optimizer") is None:
        best = run_dir / "weights" / "best.pt"
        return "finished", best if best.exists() else last
    logger.info("Reanudando entrenamiento desde %s (epoch %s).", last, ckpt.get("epoch", -1) + 1)
    return "resume", last


class Fp32OptimizerCheckpointMixin:
    """Mixin para trainers Ultralytics: guarda el estado del optimizer en **fp32**.

    ``BaseTrainer.save_model`` pasa el optimizer por ``convert_optimizer_state_dict_to_fp16``
    antes de escribir ``last.pt``. Con AdamW eso rompe la reanudación: el segundo momento
    (``exp_avg_sq`` ≈ g², del orden de 1e-9) queda por debajo del mínimo de fp16 y se
    guarda como 0, así que al reanudar el denominador de Adam colapsa y los pasos se
    multiplican. Medido (A/B con el mismo seed sobre el estudiante podado): el 46,5 % de
    los ``exp_avg_sq`` quedaban en 0 y el fine-tune reanudado daba ``box_loss = nan``
    desde el primer epoch; guardándolo en fp32, la curva reanudada sigue a la de una
    corrida sin cortes. Cuesta ~8 MB más por ``last.pt``.

    Va primero en las bases (``class T(Fp32OptimizerCheckpointMixin, PoseTrainer)``).
    """

    def save_model(self) -> None:
        import ultralytics.engine.trainer as trainer_module  # noqa: PLC0415

        original = trainer_module.convert_optimizer_state_dict_to_fp16
        trainer_module.convert_optimizer_state_dict_to_fp16 = lambda state_dict: state_dict
        try:
            super().save_model()  # type: ignore[misc]
        finally:
            trainer_module.convert_optimizer_state_dict_to_fp16 = original


def epoch_callback(ckpt: FoldCheckpoint, stage: str) -> Callable[[Any], None]:
    """Callback ``on_model_save`` de Ultralytics que registra el epoch en ``stages.json``."""

    def _on_model_save(trainer: Any) -> None:  # noqa: ANN401
        ckpt.progress(
            stage, epoch=trainer.epoch + 1, epochs=trainer.epochs, save_dir=str(trainer.save_dir)
        )

    return _on_model_save


# --- Tabla de avance en Markdown ------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _stage_cell(entry: dict[str, Any], stage: str) -> str:
    if entry.get("done"):
        oks = entry.get("logged", {}).get(f"oks_{stage}")
        return f"✅ {_fmt(oks)}" if oks is not None else "✅"
    progress = entry.get("progress")
    if progress:
        return f"⏳ ep {progress['epoch']}/{progress['epochs']}"
    return "·"


def _render(root: Path, cv_state: dict[str, Any]) -> str:
    lines = [
        f"_Actualizado: {_now()} — `{root.as_posix()}`_",
        "",
        "Cada celda: ✅ etapa terminada y logueada (con su OKS), ⏳ entrenando (último "
        "epoch guardado en `last.pt`), · pendiente.",
        "",
        "| Fold | Run MLflow | " + " | ".join(STAGES) + " | Fin del fold |",
        "|---|---|" + "---|" * len(STAGES) + "---|",
    ]
    per_stage: dict[str, dict[str, list[float]]] = {}
    for k in sorted(cv_state.get("folds", {}), key=int):
        fold = cv_state["folds"][k]
        stages = _read_json(root / f"fold{k}" / "stages.json")
        cells = [_stage_cell(stages.get(s, {}), s) for s in STAGES]
        run_id = fold.get("run_id", "")
        lines.append(
            f"| {k} | `{run_id[:8]}` | " + " | ".join(cells) + f" | {fold.get('finished', '')} |"
        )
        for stage in STAGES:
            for key, val in stages.get(stage, {}).get("logged", {}).items():
                per_stage.setdefault(stage, {}).setdefault(key, []).append(float(val))

    if per_stage:
        lines += [
            "",
            "**Media ± σ por etapa sobre los folds terminados** (n = folds con esa métrica):",
            "",
            "| Etapa | OKS | PCK@0.2 | PCK pierna | PCK brazo | Params | Latencia TFLite (ms) "
            "| Tamaño (MB) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for stage in STAGES:
            vals = per_stage.get(stage, {})

            def cell(name: str, digits: int = 4, vals: dict[str, list[float]] = vals) -> str:
                xs = vals.get(f"{name}_{stage}")
                if not xs:
                    return "—"
                sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
                return f"{statistics.fmean(xs):.{digits}f} ± {sd:.{digits}f} (n={len(xs)})"

            lines.append(
                f"| {stage} | {cell('oks')} | {cell('pck')} | {cell('pck_leg')} | "
                f"{cell('pck_arm')} | {cell('params', 0)} | {cell('latency_ms', 1)} | "
                f"{cell('size_mb', 2)} |"
            )
    return "\n".join(lines)


def render_report(root: Path) -> None:
    """Reescribe la tabla de avance entre los marcadores del Markdown de reporte.

    Nunca interrumpe la corrida: el reporte es una conveniencia (la fuente de verdad
    son los JSON), y un error acá no puede tirar días de entrenamiento.
    """
    try:
        cv_state = _read_json(root / "cv_state.json")
        report = cv_state.get("report")
        if not report:
            return
        path = Path(report)
        table = _render(root, cv_state)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        block = f"{_REPORT_BEGIN}\n{table}\n{_REPORT_END}"
        if _REPORT_BEGIN in text and _REPORT_END in text:
            head, rest = text.split(_REPORT_BEGIN, 1)
            tail = rest.split(_REPORT_END, 1)[1]
            text = head + block + tail
        else:
            text = text + ("\n\n" if text else "") + block + "\n"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo actualizar el reporte de checkpoints: %s", exc)
