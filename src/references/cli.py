"""CLI fina del generador de referencias: parsea argumentos y orquesta el módulo.

Uso (en venv-torch)::

    python -m src.references --config configs/references.yaml
    python -m src.references --config configs/references.yaml --exercise <uuid> --multi
    python -m src.references --config configs/references.yaml --model models/otro.pt --out DIR

VALIDACIÓN MANUAL (no saltear — cierra la invariante mismo-pipeline):

1. Correr el parser Kotlin de la app (``ReferenceJson``) sobre cada JSON
   emitido: debe pasar.
2. End-to-end: ejecutar el MISMO movimiento en vivo en ``RealtimeExerciseScreen``
   y confirmar distancia DTW baja / badge "Good". Este chequeo además detecta
   si el modelo pesado offline diverge del ``.tflite`` de runtime.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.references import cycle as cycle_mod
from src.references import report as report_mod
from src.references import schema as schema_mod
from src.references.angles import (
    AngleDef,
    angle_matrix,
    gap_runs,
    interpolate_gaps_wrapped,
    smooth_wrapped,
)
from src.references.backend import build_backend
from src.references.extract import extract_keypoints

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "kpt_conf_thr": 0.5,
    "max_gap_ms": 150.0,  # umbral en TIEMPO: mismo parpadeo físico a cualquier fps
    "smoothing": {"detect_window": 5, "store_window": 1},
    "cycle": {"rest_frames": 15, "valley_tol_rad": 0.15},  # TODO: calibrar valley_tol
    "multi": {"prominence_rad": 0.35, "min_gap_s": 0.8},  # TODO: calibrar
    "scoring": {"similarityScale": 0.5, "goodThreshold": 0.7},  # TODO: calibrar
}


class ReferenceError(RuntimeError):
    """Un ejercicio no pudo producir una referencia confiable (recapturar)."""


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in _DEFAULTS.items():
        if isinstance(value, dict):
            config[key] = {**value, **config.get(key, {})}
        else:
            config.setdefault(key, value)
    if not config.get("exercises"):
        raise ReferenceError(f"La config {path} no declara ejercicios")
    return config


def build_exercise_reference(
    exercise: dict, config: dict, *, multi: bool, backend: Any
) -> tuple[dict, str]:
    """Video → keypoints → ángulos → ciclo → JSON schema v1 + reporte.

    Claves opcionales por ejercicio: ``trimStartS``/``trimEndS`` (recorte en
    segundos, para clíps que no empiezan/terminan en reposo) y ``cycle``/
    ``multi`` (overrides de la calibración global: la excursión varía por
    ejercicio — una sentadilla parcial ronda 0.3 rad, una extensión de rodilla
    0.6 — y un solo umbral global no sirve para ambos).
    """
    video = Path(exercise["video"])
    defs = [AngleDef.from_config(d) for d in exercise["angleDefinitions"]]
    driver_idx = int(exercise["driverAngleIndex"])
    if not 0 <= driver_idx < len(defs):
        raise ReferenceError(f"driverAngleIndex {driver_idx} fuera de rango")

    kpts, fps, skipped = extract_keypoints(video, backend)
    lo = int(round(float(exercise.get("trimStartS", 0.0)) * fps))
    hi = int(round(float(exercise["trimEndS"]) * fps)) if "trimEndS" in exercise else kpts.shape[0]
    if lo or hi < kpts.shape[0]:
        if not 0 <= lo < hi <= kpts.shape[0]:
            raise ReferenceError(f"Trim [{lo}, {hi}) fuera de los {kpts.shape[0]} cuadros")
        kpts = kpts[lo:hi]
        skipped = [i for i in skipped if lo <= i < hi]
        logger.info(
            "%s: trim a cuadros %d..%d (%d cuadros); índices del reporte relativos al recorte",
            video.name,
            lo,
            hi - 1,
            kpts.shape[0],
        )
    angles, valid = angle_matrix(kpts, defs, float(config["kpt_conf_thr"]))

    # Señales continuas por dimensión (gaps interpolados) SOLO para detección.
    filled = np.stack(
        [interpolate_gaps_wrapped(angles[:, d], valid[:, d]) for d in range(len(defs))], axis=1
    )
    det_window = int(config["smoothing"]["detect_window"])
    cyc = {**config["cycle"], **exercise.get("cycle", {})}
    mult = {**config["multi"], **exercise.get("multi", {})}
    driver = filled[:, driver_idx]
    driver_valid = valid[:, driver_idx]

    if multi:
        spans = cycle_mod.detect_reps(
            driver,
            driver_valid,
            fps,
            prominence_rad=float(mult["prominence_rad"]),
            min_gap_s=float(mult["min_gap_s"]),
            valley_tol_rad=float(cyc["valley_tol_rad"]),
            rest_frames=int(cyc["rest_frames"]),
            detect_window=det_window,
        )
        if not spans:
            raise ReferenceError("Modo --multi: no se detectó ninguna rep")
        reps = [filled[s.start : s.end + 1] for s in spans]
        weights = np.array([d.weight for d in defs])
        medoid, outliers = cycle_mod.select_medoid(reps, weights)
        span = spans[medoid]
        n_reps, n_outliers = len(spans), len(outliers)
    else:
        span = cycle_mod.detect_single_cycle(
            driver,
            driver_valid,
            rest_frames=int(cyc["rest_frames"]),
            valley_tol_rad=float(cyc["valley_tol_rad"]),
            detect_window=det_window,
        )
        n_reps, n_outliers = 1, 0

    # Política de gaps DENTRO del ciclo, medida en tiempo (no en cuadros crudos).
    valid_crop = valid[span.start : span.end + 1]
    max_gap_frames = max(1, int(round(float(config["max_gap_ms"]) / 1000.0 * fps)))
    gap_notes: list[report_mod.GapNote] = []
    for d, ad in enumerate(defs):
        for run_start, run_end in gap_runs(valid_crop[:, d]):
            n = run_end - run_start + 1
            abs_start, abs_end = span.start + run_start, span.start + run_end
            ms = n / fps * 1000.0
            if n > max_gap_frames:
                raise ReferenceError(
                    f"Gap de {n} cuadros ({ms:.0f} ms) en '{ad.name}' "
                    f"(cuadros {abs_start}–{abs_end}, fase "
                    f"{report_mod.phase_of(abs_start, abs_end, span.peak)}) supera el máximo "
                    f"de {config['max_gap_ms']} ms: RECAPTURAR (mejor luz/encuadre)"
                )
            gap_notes.append(
                report_mod.GapNote(
                    angle_name=ad.name,
                    start=abs_start,
                    end=abs_end,
                    ms=ms,
                    phase=report_mod.phase_of(abs_start, abs_end, span.peak),
                )
            )

    # Trayectoria final: gaps cortos ya interpolados por arco corto; suavizado
    # de almacenamiento mínimo (default 1 = crudo; subirlo alisa el patrón pero
    # recorta los extremos reales del movimiento — documentado trade-off).
    store_window = int(config["smoothing"]["store_window"])
    crop = filled[span.start : span.end + 1]
    trajectory = np.stack(
        [smooth_wrapped(crop[:, d], store_window) for d in range(len(defs))], axis=1
    )

    doc = schema_mod.build_reference(
        exercise_id=str(exercise["exerciseId"]),
        angle_definitions=exercise["angleDefinitions"],
        driver_angle_index=driver_idx,
        trajectory=trajectory,
        fps=fps,
        video_name=video.name,
        scoring=config["scoring"],
    )
    text = report_mod.build_report(
        exercise_id=str(exercise["exerciseId"]),
        video_name=video.name,
        fps=fps,
        total_frames=kpts.shape[0],
        skipped_frames=len(skipped),
        angle_names=[d.name for d in defs],
        trajectory=trajectory,
        valid_crop=valid_crop,
        span=span,
        driver_index=driver_idx,
        n_reps=n_reps,
        n_outliers=n_outliers,
        gap_notes=gap_notes,
    )
    return doc, text


def run(
    config_path: Path, *, exercise_id: str | None, multi: bool, model: str | None, out: str | None
) -> int:
    config = load_config(config_path)
    if model:
        config["model"] = model
    output_dir = Path(out or config.get("output_dir", "references"))
    exercises = config["exercises"]
    if exercise_id:
        exercises = [e for e in exercises if str(e["exerciseId"]) == exercise_id]
        if not exercises:
            raise ReferenceError(f"exerciseId {exercise_id!r} no está en la config")

    backend = build_backend(config)
    failures = 0
    for exercise in exercises:
        try:
            doc, text = build_exercise_reference(exercise, config, multi=multi, backend=backend)
            path = schema_mod.write_reference(doc, output_dir)
            print(text)
            print(f"→ escrito {path}\n")
        except (ReferenceError, ValueError, FileNotFoundError, RuntimeError) as exc:
            failures += 1
            logger.error("Ejercicio %s FALLÓ: %s", exercise.get("exerciseId"), exc)
    if failures:
        logger.error("%d/%d ejercicios fallaron", failures, len(exercises))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML con el catálogo de ejercicios.")
    parser.add_argument("--exercise", help="Procesar sólo este exerciseId (default: todos).")
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Video con varias reps: elegir la mediana por medoide DTW (default: 1 rep).",
    )
    parser.add_argument("--model", help="Override del checkpoint de pose de la config.")
    parser.add_argument("--out", help="Override del directorio de salida.")
    args = parser.parse_args(argv)
    return run(
        Path(args.config),
        exercise_id=args.exercise,
        multi=args.multi,
        model=args.model,
        out=args.out,
    )
