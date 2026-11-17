"""Reporte diagnóstico por ejercicio — la determinación EMPÍRICA del signo.

El signo de la excursión del conductor no se asume: lo define la geometría con
el eje y hacia abajo, y este reporte lo imprime para confrontarlo con lo que
asume el segmentador en vivo de la app. También localiza cada gap (ángulo,
cuadros, fase del ciclo) para que una recaptura sepa qué falló.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.references.cycle import CycleSpan


@dataclass(frozen=True)
class GapNote:
    """Un gap interpolado dentro del ciclo (índices absolutos del video)."""

    angle_name: str
    start: int
    end: int
    ms: float
    phase: str  # subida | pico | bajada


def phase_of(start: int, end: int, peak: int) -> str:
    """Fase del ciclo en la que cae una corrida de cuadros (índices absolutos)."""
    if start <= peak <= end:
        return "pico"
    return "subida" if end < peak else "bajada"


def build_report(
    *,
    exercise_id: str,
    video_name: str,
    fps: float,
    total_frames: int,
    skipped_frames: int,
    angle_names: list[str],
    trajectory: np.ndarray,
    valid_crop: np.ndarray,
    span: CycleSpan,
    driver_index: int,
    n_reps: int,
    n_outliers: int,
    gap_notes: list[GapNote],
) -> str:
    """Texto del reporte. ``trajectory`` es el recorte final (T, D); ``valid_crop``
    la máscara ORIGINAL del recorte (pre-interpolación), que es la que delata gaps."""
    t_cycle = trajectory.shape[0]
    peak_rel = span.peak - span.start
    driver_peak = float(trajectory[peak_rel, driver_index])
    sign_txt = "POSITIVO (+)" if span.sign > 0 else "NEGATIVO (−)"
    n_invalid = int(np.count_nonzero(~valid_crop))

    lines = [
        f"=== Referencia: {exercise_id} ===",
        f"video: {video_name} | fps: {fps:.2f} | cuadros totales: {total_frames}"
        f" | sin persona: {skipped_frames}",
        f"reps detectadas: {n_reps} (outliers descartados: {n_outliers})"
        f" | ciclo: cuadros {span.start}..{span.end} (T={t_cycle}, {t_cycle / fps:.2f} s)",
        f"conductor: {angle_names[driver_index]} (dim {driver_index})",
        f"SIGNO de la excursión del conductor: {sign_txt}"
        f" — pico {driver_peak:+.3f} rad en el cuadro {span.peak}",
        "",
        f"{'ángulo':<16}{'reposo':>9}{'extremo':>10}{'min':>9}{'max':>9}{'inválidos':>11}",
    ]
    for d, name in enumerate(angle_names):
        col = trajectory[:, d]
        bad = int(np.count_nonzero(~valid_crop[:, d]))
        lines.append(
            f"{name:<16}{trajectory[0, d]:>9.3f}{trajectory[peak_rel, d]:>10.3f}"
            f"{col.min():>9.3f}{col.max():>9.3f}{bad:>11d}"
        )
    for note in gap_notes:
        lines.append(
            f"  ⚠ gap en {note.angle_name}: cuadros {note.start}–{note.end}"
            f" ({note.ms:.0f} ms, {note.phase}) → INTERPOLADO por arco corto"
        )
    if n_invalid:
        lines += [
            "",
            f"⚠⚠ {n_invalid} dimensiones-cuadro inválidas dentro del ciclo."
            " Una buena referencia debería tener 0:",
            "   considerá RECAPTURAR con mejor luz/encuadre (el detalle de cada gap está arriba).",
        ]
    else:
        lines += ["", "OK: 0 dimensiones inválidas dentro del ciclo."]
    return "\n".join(lines)
