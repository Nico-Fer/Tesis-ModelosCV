"""Exporta los resultados de la CV completa de la Fase 5 a ``resultados/fase5_cv/``.

Lee sólo lo que la corrida dejó en disco (``runs/fase5_cv/``: ``cv_state.json``,
``fold{k}/stages.json``, los ``results.csv`` de Ultralytics y los ``best_int8.tflite``)
más el file store de MLflow (``mlruns/``, sin servidor), y genera:

* ``tablas/``   CSV con las métricas por fold, el resumen media ± σ por etapa, las
  diferencias pareadas con IC 95 % y p-valor, el entrenamiento por fold y los sujetos
  de validación de cada fold; además, un JSON con el run padre de MLflow.
* ``curvas/``   el ``results.csv`` de Ultralytics del fine-tune y de la distillation de
  cada fold (métricas por epoch).
* ``figuras/``  PNG para el informe.
* ``modelos/``  el ``.tflite`` INT8 de cada fold (gitignoreados por ``*.tflite``).
* ``config/``   la config usada y un ``procedencia.json`` (run IDs, huella de la config,
  commit, versiones, MD5 de los artefactos).
* ``bitacora_ejecucion.md``: copia de ``docs/fase5_cv_bitacora.md``.

Los documentos de interpretación (``README.md``, ``RESULTADOS.md``, ``METODOLOGIA.md``)
se escriben a mano y este script no los toca. Es idempotente: regenera todo lo demás.

Uso (venv-torch, desde la raíz del repo)::

    venv-torch\\Scripts\\python.exe scripts\\export_fase5_cv_results.py
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics as st
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.uco_to_yolo import parse_stem  # noqa: E402
from src.utils.checkpoints import config_fingerprint  # noqa: E402

RUNS = REPO / "runs" / "fase5_cv"
OUT = REPO / "resultados" / "fase5_cv"
CONFIG = REPO / "configs" / "combined_yolov8n.yaml"
BITACORA = REPO / "docs" / "fase5_cv_bitacora.md"
FOLDS_DIR = REPO / "data" / "processed" / "uco_pseudo" / "folds"
N_FOLDS = 5
STAGES = ("base", "pruned", "pruned_ft", "distilled", "int8")
METRICS = ("oks", "pck", "pck_leg", "pck_arm", "params", "latency_ms", "size_mb", "n_frames")
TRANSITIONS = (
    ("base", "pruned_ft"),
    ("pruned_ft", "distilled"),
    ("distilled", "int8"),
    ("base", "int8"),
    ("pruned_ft", "int8"),
)
# Fitness de Ultralytics 8.3.253 para pose: mAP50-95 de box + mAP50-95 de pose. Es lo
# que decide el best.pt y el early stopping.
FITNESS_COLS = ("metrics/mAP50-95(B)", "metrics/mAP50-95(P)")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _results_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _epoch_seconds(times: pd.Series) -> list[float]:
    """Duración de cada epoch a partir de la columna ``time`` (acumulada por lanzamiento).

    Al reanudar, Ultralytics reinicia el reloj: un valor menor al anterior marca un
    nuevo lanzamiento y entonces la duración es el propio valor.
    """
    out, prev = [], 0.0
    for t in times:
        out.append(t - prev if t > prev else t)
        prev = t
    return out


# --- Métricas por etapa ---------------------------------------------------------------


def load_stages() -> dict[int, dict]:
    return {k: _read_json(RUNS / f"fold{k}" / "stages.json") for k in range(N_FOLDS)}


def table_per_fold(stages: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for k in range(N_FOLDS):
        for s in STAGES:
            logged = stages[k][s]["logged"]
            row = {"fold": k, "etapa": s}
            for m in METRICS:
                row[m] = logged.get(f"{m}_{s}")
            rows.append(row)
    return pd.DataFrame(rows)


def table_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in STAGES:
        sub = per_fold[per_fold.etapa == s]
        for m in METRICS:
            vals = [v for v in sub[m] if v is not None and not pd.isna(v)]
            if not vals:
                continue
            rows.append(
                {
                    "etapa": s,
                    "metrica": m,
                    "media": st.fmean(vals),
                    "sd_poblacional": st.pstdev(vals) if len(vals) > 1 else 0.0,
                    "sd_muestral": st.stdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals),
                    "max": max(vals),
                    "n_folds": len(vals),
                }
            )
    return pd.DataFrame(rows)


def table_deltas(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in TRANSITIONS:
        for m in ("oks", "pck", "pck_leg", "pck_arm"):
            va = per_fold[per_fold.etapa == a].sort_values("fold")[m].to_numpy(dtype=float)
            vb = per_fold[per_fold.etapa == b].sort_values("fold")[m].to_numpy(dtype=float)
            d = vb - va
            n = len(d)
            mean, sd = float(d.mean()), float(d.std(ddof=1))
            half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
            t_stat, p = stats.ttest_rel(vb, va) if sd > 0 else (float("nan"), float("nan"))
            row = {"transicion": f"{a} -> {b}", "metrica": m}
            row.update({f"delta_fold{k}": float(d[k]) for k in range(n)})
            row.update(
                media=mean,
                sd_muestral=sd,
                ic95_inf=mean - half,
                ic95_sup=mean + half,
                t=float(t_stat),
                p_valor_t_pareado=float(p),
                folds_mismo_signo=int(max((d > 0).sum(), (d < 0).sum())),
            )
            rows.append(row)
    return pd.DataFrame(rows)


# --- Entrenamiento por fold -----------------------------------------------------------


def fold_subjects(k: int) -> list[str]:
    stems = [
        Path(line.strip()).stem
        for line in (FOLDS_DIR / f"fold{k}_val.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(
        {parse_stem(s).subject for s in stems}, key=lambda x: int(x) if x.isdigit() else x
    )


def _count_lines(path: Path) -> int:
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def table_training(stages: dict[int, dict], cv: dict) -> pd.DataFrame:
    rows = []
    for k in range(N_FOLDS):
        fdir = RUNS / f"fold{k}"
        ft = _results_csv(fdir / "pruned_finetune" / "results.csv")
        kd = _results_csv(fdir / "distillation" / "distill_student" / "results.csv")
        fitness = kd[list(FITNESS_COLS)].sum(axis=1)
        best = int(kd.loc[fitness.idxmax(), "epoch"])
        kd_secs = _epoch_seconds(kd["time"])
        ft_secs = _epoch_seconds(ft["time"])
        n_val = _count_lines(FOLDS_DIR / f"fold{k}_val.txt")
        n_eval = int(stages[k]["base"]["logged"]["n_frames_base"])
        rows.append(
            {
                "fold": k,
                "n_train": _count_lines(FOLDS_DIR / f"fold{k}_train.txt"),
                "n_val": n_val,
                "n_frames_evaluados": n_eval,
                "frames_sin_bbox_base": n_val - n_eval,
                "sujetos_val": " ".join(fold_subjects(k)),
                "ft_epochs": int(ft["epoch"].max()),
                "ft_map50_95_pose_final": float(ft["metrics/mAP50-95(P)"].iloc[-1]),
                "ft_min_por_epoch": st.fmean(ft_secs) / 60,
                "kd_epochs_efectivos": int(kd["epoch"].max()),
                "kd_early_stopping": int(kd["epoch"].max()) < 80,
                "kd_mejor_epoch": best,
                "kd_map50_95_pose_mejor": float(kd.loc[fitness.idxmax(), "metrics/mAP50-95(P)"]),
                "kd_min_por_epoch": st.fmean(kd_secs) / 60,
                "kd_horas_entrenamiento": sum(kd_secs) / 3600,
                "inicio_fold": cv["folds"][str(k)].get("started"),
                "fin_fold": cv["folds"][str(k)].get("finished"),
                "run_id_mlflow": cv["folds"][str(k)].get("run_id"),
            }
        )
    return pd.DataFrame(rows)


# --- MLflow ---------------------------------------------------------------------------


def export_mlflow(parent_id: str) -> dict:
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    client = MlflowClient(tracking_uri=(REPO / "mlruns").resolve().as_uri())
    parent = client.get_run(parent_id)
    children = client.search_runs(
        [parent.info.experiment_id], f"tags.mlflow.parentRunId = '{parent_id}'"
    )
    return {
        "experimento_id": parent.info.experiment_id,
        "run_padre": {
            "run_id": parent_id,
            "nombre": parent.info.run_name,
            "estado": parent.info.status,
            "inicio_ms": parent.info.start_time,
            "fin_ms": parent.info.end_time,
            "tags": {k: v for k, v in parent.data.tags.items() if not k.startswith("mlflow.")},
            "params": parent.data.params,
            "metricas": dict(sorted(parent.data.metrics.items())),
        },
        "runs_fold": sorted(
            (
                {
                    "nombre": r.info.run_name,
                    "run_id": r.info.run_id,
                    "estado": r.info.status,
                    "metricas": dict(sorted(r.data.metrics.items())),
                }
                for r in children
            ),
            key=lambda r: r["nombre"],
        ),
    }


# --- Figuras --------------------------------------------------------------------------

# Paleta (skill dataviz, modo claro): una serie de acento + contexto en gris.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"
CONTEXT = "#b7b5ad"
STAGE_LABEL = {
    "base": "base\n(COCO)",
    "pruned_ft": "pruned_ft\n(poda + FT)",
    "distilled": "distilled\n(+ KD)",
    "int8": "int8\n(entregable)",
}


def _c(value: float, fmt: str = ".3f") -> str:
    """Número con coma decimal y signo menos tipográfico (convención del informe)."""
    return format(value, fmt).replace("-", "−").replace(".", ",")


def _comma_axis(axis, fmt: str) -> None:  # noqa: ANN001
    from matplotlib.ticker import FuncFormatter  # noqa: PLC0415

    axis.set_major_formatter(FuncFormatter(lambda v, _pos: _c(v, fmt)))


def _style(plt) -> None:  # noqa: ANN001
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 1,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def fig_oks_by_stage(per_fold: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    stages = ["base", "pruned_ft", "distilled", "int8"]
    x = list(range(len(stages)))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k in range(N_FOLDS):
        ys = [
            float(per_fold[(per_fold.fold == k) & (per_fold.etapa == s)].oks.iloc[0])
            for s in stages
        ]
        ax.plot(x, ys, color=CONTEXT, lw=1.2, marker="o", ms=4, zorder=2)
    means = [per_fold[per_fold.etapa == s].oks.mean() for s in stages]
    sds = [per_fold[per_fold.etapa == s].oks.std(ddof=0) for s in stages]
    ax.errorbar(
        x,
        means,
        yerr=sds,
        color=ACCENT,
        lw=2,
        marker="o",
        ms=8,
        capsize=4,
        mec=SURFACE,
        mew=2,
        zorder=3,
    )
    ticks = [f"{STAGE_LABEL[s]}\n{_c(m)} ± {_c(sd)}" for s, m, sd in zip(stages, means, sds)]
    ax.set_xticks(x, ticks)
    ax.set_xlim(-0.3, len(stages) - 0.7)
    _comma_axis(ax.yaxis, ".2f")
    ax.set_ylabel("OKS (val completo del fold)")
    ax.set_title(
        "OKS por etapa — media ± σ de los 5 folds (azul); cada línea gris es un fold", loc="left"
    )
    fig.text(
        0.0,
        -0.10,
        "La etapa 'pruned' (OKS ≈ 0 en los 5 folds) se omite para no aplastar la escala.",
        fontsize=8,
        color=MUTED,
    )
    fig.savefig(out)
    plt.close(fig)


def fig_paired_deltas(deltas: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    picks = [
        ("base -> pruned_ft", "Fine-tune post-poda\n(base → pruned_ft)"),
        ("pruned_ft -> distilled", "Etapa de distillation\n(pruned_ft → distilled)"),
        ("distilled -> int8", "Cuantización INT8\n(distilled → int8)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (key, title) in zip(axes, picks):
        row = deltas[(deltas.transicion == key) & (deltas.metrica == "oks")].iloc[0]
        vals = [row[f"delta_fold{k}"] for k in range(N_FOLDS)]
        ax.axvline(0, color=AXIS, lw=1, zorder=1)
        ax.scatter(
            vals,
            [0.0] * N_FOLDS,
            s=36,
            color=CONTEXT,
            edgecolor=SURFACE,
            lw=1.5,
            zorder=2,
            label="cada fold",
        )
        ax.errorbar(
            [row.media],
            [0.35],
            xerr=[[row.media - row.ic95_inf], [row.ic95_sup - row.media]],
            color=ACCENT,
            lw=2,
            marker="o",
            ms=8,
            capsize=4,
            mec=SURFACE,
            mew=2,
            zorder=3,
        )
        ax.annotate(
            f"{_c(row.media, '+.4f')}\nIC 95 % [{_c(row.ic95_inf, '+.4f')}; "
            f"{_c(row.ic95_sup, '+.4f')}]",
            (row.media, 0.35),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK,
        )
        ax.set_ylim(-0.4, 1.3)
        ax.set_yticks([])
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Δ OKS (pareado por fold)")
        lo, hi = min(vals + [row.ic95_inf, 0.0]), max(vals + [row.ic95_sup, 0.0])
        pad = (hi - lo) * 0.25 or 0.001
        ax.set_xlim(lo - pad, hi + pad)
        _comma_axis(ax.xaxis, ".3f")
    fig.suptitle(
        "Cuánto aporta cada etapa — media con IC 95 % (azul) y diferencia de cada "
        "fold (gris). Cada panel tiene su propia escala.",
        x=0.0,
        ha="left",
        fontsize=10,
        color=INK_2,
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_training_curves(out: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, N_FOLDS, figsize=(12, 3.0), sharey=True)
    for k, ax in enumerate(axes):
        fdir = RUNS / f"fold{k}"
        ft = _results_csv(fdir / "pruned_finetune" / "results.csv")
        kd = _results_csv(fdir / "distillation" / "distill_student" / "results.csv")
        n_ft = int(ft["epoch"].max())
        ax.plot(ft["epoch"], ft["metrics/mAP50-95(P)"], color=CONTEXT, lw=2)
        ax.plot(kd["epoch"] + n_ft, kd["metrics/mAP50-95(P)"], color=ACCENT, lw=2)
        fitness = kd[list(FITNESS_COLS)].sum(axis=1)
        b = fitness.idxmax()
        ax.plot(
            [kd.loc[b, "epoch"] + n_ft],
            [kd.loc[b, "metrics/mAP50-95(P)"]],
            "o",
            ms=7,
            color=ACCENT,
            mec=SURFACE,
            mew=2,
        )
        ax.axvline(n_ft + 0.5, color=AXIS, lw=1)
        ax.set_title(f"Fold {k}", loc="left", fontsize=10)
        ax.set_xlabel("epoch (FT + KD)")
        ax.set_ylim(0.65, 0.97)
        stop = int(kd["epoch"].max())
        note = f"KD: {stop} ep" + (" (early stop)" if stop < 80 else "")
        ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    axes[0].set_ylabel("mAP50-95 pose (val, Ultralytics)")
    _comma_axis(axes[0].yaxis, ".1f")
    fig.suptitle(
        "Curvas de entrenamiento por fold — fine-tune post-poda (gris, 10 ep) y "
        "distillation (azul); el punto marca el best.pt elegido",
        x=0.0,
        ha="left",
        fontsize=10,
        color=INK_2,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_size_vs_oks(summary: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    def get(stage: str, metric: str) -> tuple[float, float]:
        r = summary[(summary.etapa == stage) & (summary.metrica == metric)].iloc[0]
        return float(r.media), float(r.sd_poblacional)

    pts = [
        ("base", "base (COCO)"),
        ("distilled", "distilled (fp32)"),
        ("int8", "int8 (entregable)"),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for stage, label in pts:
        size, _ = get(stage, "size_mb")
        oks, sd = get(stage, "oks")
        ax.errorbar(
            [size],
            [oks],
            yerr=[sd],
            color=ACCENT,
            marker="o",
            ms=8,
            capsize=4,
            lw=2,
            mec=SURFACE,
            mew=2,
        )
        ax.annotate(
            f"{label}\n{_c(size, '.2f')} MB · OKS {_c(oks)}",
            (size, oks),
            xytext=(10, -4),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
            va="top",
        )
    ax.set_xlabel("Tamaño del archivo (MB)")
    ax.set_ylabel("OKS (media ± σ, 5 folds)")
    ax.set_xlim(1.5, 8.5)
    ax.set_ylim(0.83, 0.99)
    ax.grid(axis="x", color=GRID)
    ax.set_title("Tamaño vs. precisión", loc="left")
    _comma_axis(ax.yaxis, ".2f")
    fig.savefig(out)
    plt.close(fig)


# --- Main -----------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> int:
    cv = _read_json(RUNS / "cv_state.json")
    if not cv.get("finished"):
        raise SystemExit("La CV no figura terminada en runs/fase5_cv/cv_state.json.")
    stages = load_stages()

    for sub in ("tablas", "curvas", "figuras", "modelos", "config"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    per_fold = table_per_fold(stages)
    summary = table_summary(per_fold)
    deltas = table_deltas(per_fold)
    training = table_training(stages, cv)
    per_fold.to_csv(OUT / "tablas" / "metricas_por_fold.csv", index=False)
    summary.to_csv(OUT / "tablas" / "resumen_por_etapa.csv", index=False)
    deltas.to_csv(OUT / "tablas" / "deltas_pareados.csv", index=False)
    training.to_csv(OUT / "tablas" / "entrenamiento_por_fold.csv", index=False)

    mlflow_dump = export_mlflow(cv["parent_run_id"])
    (OUT / "tablas" / "mlflow_runs.json").write_text(
        json.dumps(mlflow_dump, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    artefactos = {}
    for k in range(N_FOLDS):
        fdir = RUNS / f"fold{k}"
        shutil.copy2(
            fdir / "pruned_finetune" / "results.csv",
            OUT / "curvas" / f"fold{k}_finetune_post_poda.csv",
        )
        shutil.copy2(
            fdir / "distillation" / "distill_student" / "results.csv",
            OUT / "curvas" / f"fold{k}_distillation.csv",
        )
        tflite = Path(stages[k]["int8"]["artifacts"]["tflite"])
        dst = OUT / "modelos" / f"fold{k}_int8.tflite"
        shutil.copy2(tflite, dst)
        artefactos[f"fold{k}"] = {
            "int8_tflite": {
                "copia": str(dst.relative_to(OUT)),
                "origen": str(tflite),
                "md5": _md5(dst),
                "bytes": dst.stat().st_size,
            },
            "distilled_best_pt": str(
                fdir / "distillation" / "distill_student" / "weights" / "best.pt"
            ),
            "pruned_ft_pt": str(fdir / "student_pruned_ft.pt"),
            "pruned_pt": str(fdir / "student_pruned.pt"),
        }

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    _style(plt)
    fig_oks_by_stage(per_fold, OUT / "figuras" / "oks_por_etapa.png")
    fig_paired_deltas(deltas, OUT / "figuras" / "deltas_pareados.png")
    fig_training_curves(OUT / "figuras" / "curvas_entrenamiento.png")
    fig_size_vs_oks(summary, OUT / "figuras" / "tamano_vs_oks.png")

    shutil.copy2(CONFIG, OUT / "config" / CONFIG.name)
    shutil.copy2(BITACORA, OUT / "bitacora_ejecucion.md")

    import torch  # noqa: PLC0415
    import ultralytics  # noqa: PLC0415

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    control = REPO / "models" / "yolov8n-pose_saved_model" / "yolov8n-pose_int8.tflite"
    procedencia = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "cv": {
            "inicio": cv.get("created"),
            "fin": cv.get("finished"),
            "huella_config": cv["fingerprint"],
            "huella_config_actual": config_fingerprint(config),
            "checkpoint_dir": str(RUNS),
        },
        "mlflow": {
            "experimento": config["mlflow"]["experiment"],
            "experimento_id": mlflow_dump["experimento_id"],
            "run_padre": cv["parent_run_id"],
            "runs_fold": {k: v["run_id"] for k, v in cv["folds"].items()},
        },
        "codigo": {
            "rama": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "HEAD"),
            "cambios_sin_commitear": _git("status", "--short").splitlines(),
        },
        "entorno": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "artefactos": artefactos,
        "control_int8_stock": {
            "ruta": str(control),
            "md5": _md5(control) if control.exists() else None,
        },
    }
    (OUT / "config" / "procedencia.json").write_text(
        json.dumps(procedencia, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Resultados exportados a {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
