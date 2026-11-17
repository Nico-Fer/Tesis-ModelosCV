"""Probe de señal angular sobre un video de referencia, ANTES de tocar la config.

Mide, por dimensión candidata, cuánta excursión real hay respecto del reposo
(wraparound-aware, misma matemática que el generador) y cuán confiable es la
cadena de keypoints. Sirve para: elegir driver, descartar dimensiones sin señal
(p.ej. elevación de talones ≈ traslación pura → ángulos quietos) y comparar dos
capturas del mismo ejercicio (cuadros sin persona, inválidos, jitter).

Uso (venv-torch, desde la raíz)::

    python scripts/probe_reference_angles.py --config configs/references.yaml \
        --video data/references/videos/c...0005.MOV [--video otro.MOV] \
        [--dims rodilla_izq,cadera_izq] [--cache]

Sin MLflow. ``--cache`` guarda los keypoints en ``<video>.probe_kpts.npz`` (el
backend INT8 corre en CPU ≈ 200 ms/cuadro; iterar sobre dims no debe re-inferir).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.references.angles import (  # noqa: E402
    AngleDef,
    angle_matrix,
    interpolate_gaps_wrapped,
    smooth_wrapped,
    wrap_diff,
)
from src.references.backend import build_backend  # noqa: E402
from src.references.cycle import detect_reps, excursion, rest_baseline  # noqa: E402
from src.references.extract import extract_keypoints  # noqa: E402

# Catálogo de dimensiones candidatas (superset de lo que usan las referencias).
CATALOG: dict[str, dict[str, str]] = {
    "cadera_izq": {"pointA": "LEFT_SHOULDER", "vertex": "LEFT_HIP", "pointC": "LEFT_KNEE"},
    "cadera_der": {"pointA": "RIGHT_SHOULDER", "vertex": "RIGHT_HIP", "pointC": "RIGHT_KNEE"},
    "rodilla_izq": {"pointA": "LEFT_HIP", "vertex": "LEFT_KNEE", "pointC": "LEFT_ANKLE"},
    "rodilla_der": {"pointA": "RIGHT_HIP", "vertex": "RIGHT_KNEE", "pointC": "RIGHT_ANKLE"},
    # Vértice en el tobillo contra el tobillo contrario: único ángulo que toca
    # el pie en COCO-17 (no hay dedos/talón).
    "tobillo_cruz_izq": {"pointA": "LEFT_KNEE", "vertex": "LEFT_ANKLE", "pointC": "RIGHT_ANKLE"},
    "tobillo_cruz_der": {"pointA": "RIGHT_KNEE", "vertex": "RIGHT_ANKLE", "pointC": "LEFT_ANKLE"},
    # Inclinación de tronco (lean adelante/atrás visto de perfil).
    "tronco_izq": {"pointA": "LEFT_SHOULDER", "vertex": "LEFT_HIP", "pointC": "LEFT_ANKLE"},
    "tronco_der": {"pointA": "RIGHT_SHOULDER", "vertex": "RIGHT_HIP", "pointC": "RIGHT_ANKLE"},
    "hombro_izq": {"pointA": "LEFT_HIP", "vertex": "LEFT_SHOULDER", "pointC": "LEFT_ELBOW"},
    "hombro_der": {"pointA": "RIGHT_HIP", "vertex": "RIGHT_SHOULDER", "pointC": "RIGHT_ELBOW"},
}

DETECT_WINDOW = 5  # mismo suavizado que usa la detección de ciclo del generador
REST_FRAMES = 15


def _get_keypoints(
    video: Path, backend_factory, cache: bool
) -> tuple[np.ndarray, float, int]:
    cache_path = video.with_suffix(video.suffix + ".probe_kpts.npz")
    if cache and cache_path.exists():
        data = np.load(cache_path)
        print(f"[cache] {cache_path.name}")
        return data["kpts"], float(data["fps"]), int(data["n_skipped"])
    kpts, fps, skipped = extract_keypoints(video, backend_factory())
    if cache:
        np.savez_compressed(cache_path, kpts=kpts, fps=fps, n_skipped=len(skipped))
    return kpts, fps, len(skipped)


def probe_video(video: Path, kpts: np.ndarray, fps: float, n_skipped: int,
                dim_names: list[str], conf_thr: float) -> None:
    defs = [AngleDef.from_config({"name": n, **CATALOG[n]}) for n in dim_names]
    angles, valid = angle_matrix(kpts, defs, conf_thr)
    t = kpts.shape[0]
    print(f"\n=== {video.name} ===")
    print(f"cuadros: {t} | fps: {fps:.2f} | sin persona: {n_skipped} ({n_skipped / t:.1%})")
    header = (
        f"{'dim':<18}{'inval%':>8}{'conf':>7}{'reposo':>9}{'pico':>9}"
        f"{'|exc|max':>10}{'|exc|p95':>10}{'jitter':>9}"
    )
    print(header)
    for d, ad in enumerate(defs):
        col_valid = valid[:, d]
        n_valid = int(col_valid.sum())
        chain_conf = float(kpts[:, [ad.point_a, ad.vertex, ad.point_c], 2].mean())
        if n_valid < 2 * REST_FRAMES:
            print(f"{ad.name:<18}{1 - n_valid / t:>8.1%}{chain_conf:>7.2f}"
                  f"{'—— sin cuadros válidos suficientes ——':>47}")
            continue
        filled = interpolate_gaps_wrapped(angles[:, d], col_valid)
        smooth = smooth_wrapped(filled, DETECT_WINDOW)
        base = rest_baseline(smooth, col_valid, REST_FRAMES)
        exc = excursion(smooth, base)
        abs_exc = np.abs(exc)
        peak = float(exc[int(np.argmax(abs_exc))])
        # Jitter cuadro a cuadro sobre la señal CRUDA en pares válidos consecutivos.
        pair = col_valid[:-1] & col_valid[1:]
        raw_diff = wrap_diff(angles[1:, d][pair], angles[:-1, d][pair])
        jitter = float(np.median(np.abs(raw_diff))) if np.any(pair) else float("nan")
        print(
            f"{ad.name:<18}{1 - n_valid / t:>8.1%}{chain_conf:>7.2f}{base:>9.3f}"
            f"{peak:>+9.3f}{abs_exc.max():>10.3f}{np.percentile(abs_exc, 95):>10.3f}"
            f"{jitter:>9.4f}"
        )


def probe_reps(video: Path, kpts: np.ndarray, fps: float, dim: str, conf_thr: float,
               multi_cfg: dict, cycle_cfg: dict) -> None:
    """Reps que detectaría el generador en modo --multi sobre ``dim``."""
    ad = AngleDef.from_config({"name": dim, **CATALOG[dim]})
    angles, valid = angle_matrix(kpts, [ad], conf_thr)
    filled = interpolate_gaps_wrapped(angles[:, 0], valid[:, 0])
    spans = detect_reps(
        filled,
        valid[:, 0],
        fps,
        prominence_rad=float(multi_cfg["prominence_rad"]),
        min_gap_s=float(multi_cfg["min_gap_s"]),
        valley_tol_rad=float(cycle_cfg["valley_tol_rad"]),
        rest_frames=int(cycle_cfg["rest_frames"]),
        detect_window=DETECT_WINDOW,
    )
    smooth = smooth_wrapped(filled, DETECT_WINDOW)
    base = rest_baseline(filled, valid[:, 0], int(cycle_cfg["rest_frames"]))
    exc = excursion(smooth, base)
    print(f"[reps] {video.name} · {dim} · prominence={multi_cfg['prominence_rad']}"
          f" valley_tol={cycle_cfg['valley_tol_rad']}: {len(spans)} reps")
    for s in spans:
        amp = abs(float(exc[s.peak]))
        print(f"    cuadros {s.start:>4}–{s.end:<4} (T={s.end - s.start + 1:>3},"
              f" {(s.end - s.start + 1) / fps:4.2f} s) | pico cuadro {s.peak:>4}"
              f" | amplitud {amp:.3f} rad | signo {'+' if s.sign > 0 else '−'}")


def probe_series(video: Path, kpts: np.ndarray, fps: float, dim: str, conf_thr: float,
                 rest_frames: int, bucket: int = 10) -> None:
    """Sparkline ASCII de |excursión| suavizada: ver reps, drift y reposo en extremos."""
    ad = AngleDef.from_config({"name": dim, **CATALOG[dim]})
    angles, valid = angle_matrix(kpts, [ad], conf_thr)
    filled = interpolate_gaps_wrapped(angles[:, 0], valid[:, 0])
    smooth = smooth_wrapped(filled, DETECT_WINDOW)
    base = rest_baseline(filled, valid[:, 0], rest_frames)
    exc = excursion(smooth, base)
    print(f"[series] {video.name} · {dim} · baseline {base:+.3f} rad · bucket={bucket} cuadros")
    scale = 40.0 / max(float(np.abs(exc).max()), 1e-9)
    for i in range(0, exc.size, bucket):
        chunk = exc[i : i + bucket]
        m = float(chunk[np.argmax(np.abs(chunk))])
        bar = "#" * int(round(abs(m) * scale))
        print(f"  {i:>5} {m:+.3f} {'-' if m < 0 else '+'}|{bar}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML del generador (backend + umbral).")
    parser.add_argument("--video", action="append", required=True, help="Video(s) a analizar.")
    parser.add_argument("--dims", help=f"Dims separadas por coma (default: todas: {list(CATALOG)}).")
    parser.add_argument("--cache", action="store_true", help="Cachear keypoints en .npz.")
    parser.add_argument("--reps", help="Dim conductora: detectar reps como haría --multi.")
    parser.add_argument("--series", help="Dim: sparkline de |excursión| para inspección visual.")
    parser.add_argument("--prominence", type=float, help="Override de multi.prominence_rad.")
    parser.add_argument("--valley-tol", type=float, help="Override de cycle.valley_tol_rad.")
    parser.add_argument("--trim-start-s", type=float, help="Descartar los primeros N segundos.")
    parser.add_argument("--trim-end-s", type=float, help="Descartar todo después del segundo N.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    conf_thr = float(config.get("kpt_conf_thr", 0.5))
    multi_cfg = {"prominence_rad": 0.35, "min_gap_s": 0.8, **config.get("multi", {})}
    cycle_cfg = {"rest_frames": 15, "valley_tol_rad": 0.15, **config.get("cycle", {})}
    if args.prominence is not None:
        multi_cfg["prominence_rad"] = args.prominence
    if args.valley_tol is not None:
        cycle_cfg["valley_tol_rad"] = args.valley_tol
    dim_names = [d.strip() for d in args.dims.split(",")] if args.dims else list(CATALOG)
    unknown = [d for d in dim_names if d not in CATALOG]
    if unknown:
        parser.error(f"Dims fuera del catálogo: {unknown}")

    backend = None

    def backend_factory():
        nonlocal backend
        if backend is None:
            backend = build_backend(config)
        return backend

    for v in args.video:
        video = Path(v)
        kpts, fps, n_skipped = _get_keypoints(video, backend_factory, args.cache)
        lo = int(round(args.trim_start_s * fps)) if args.trim_start_s else 0
        hi = int(round(args.trim_end_s * fps)) if args.trim_end_s else kpts.shape[0]
        if lo or hi < kpts.shape[0]:
            kpts = kpts[lo:hi]
            print(f"[trim] cuadros {lo}..{hi - 1} (quedan {kpts.shape[0]})")
        if args.reps:
            if args.reps not in CATALOG:
                parser.error(f"--reps {args.reps!r} fuera del catálogo")
            probe_reps(video, kpts, fps, args.reps, conf_thr, multi_cfg, cycle_cfg)
        elif args.series:
            if args.series not in CATALOG:
                parser.error(f"--series {args.series!r} fuera del catálogo")
            probe_series(video, kpts, fps, args.series, conf_thr, int(cycle_cfg["rest_frames"]))
        else:
            probe_video(video, kpts, fps, n_skipped, dim_names, conf_thr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
