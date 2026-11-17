"""Corre la compuerta OOD sobre los modelos de la CV de la Fase 5 y las referencias.

Para cada set de ``data/processed/ood_gate_coco/<split>/`` (armado por
``scripts/build_ood_gt_set.py``) evalúa con ``diagnose_kpt_ratio.gate_gt_set``:

* los 5 ``.tflite`` INT8 de la CV (``resultados/fase5_cv/modelos/``),
* los 5 ``best.pt`` fp32 de la distillation, para separar pesos de cuantización,
* las referencias: el control stock INT8, el deploy v2 (roto en la app), el v3
  (desplegado) y el deploy de la Fase 5 (``fase5-deploy``, run ``42c1c36d``).

Escribe en ``resultados/fase5_cv/gate_ood/``: ``resumen.csv``, ``por_imagen.csv``,
el JSON completo por set, overlays de ejemplo y la figura ``razon_por_modelo.png``.

Uso (venv-torch, desde la raíz del repo)::

    venv-torch\\Scripts\\python.exe scripts\\run_ood_gate_fase5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from diagnose_kpt_ratio import gate_gt_set  # noqa: E402

SETS = REPO / "data" / "processed" / "ood_gate_coco"
OUT = REPO / "resultados" / "fase5_cv" / "gate_ood"
RUNS = REPO / "runs" / "fase5_cv"
CV_MODELS = REPO / "resultados" / "fase5_cv" / "modelos"

# (etiqueta, grupo, ruta). El orden es el de las tablas y la figura.
MODELS: list[tuple[str, str, Path]] = [
    *[
        (f"cv_fold{k}_int8", "CV Fase 5 (INT8)", CV_MODELS / f"fold{k}_int8.tflite")
        for k in range(5)
    ],
    *[
        (
            f"cv_fold{k}_fp32",
            "CV Fase 5 (fp32)",
            RUNS / f"fold{k}" / "distillation" / "distill_student" / "weights" / "best.pt",
        )
        for k in range(5)
    ],
    (
        "deploy_fase5_int8",
        "Referencia",
        REPO
        / "mlartifacts"
        / "536609034621976297"
        / "42c1c36d89c54879a5ff9740c4c8b728"
        / "artifacts"
        / "model"
        / "best_int8.tflite",
    ),
    (
        "deploy_v2_int8",
        "Referencia",
        REPO
        / "models"
        / "student_pruned_ft_deploy_v2_saved_model"
        / "student_pruned_ft_deploy_v2_int8.tflite",
    ),
    (
        "deploy_v3_int8",
        "Referencia",
        REPO
        / "models"
        / "student_pruned_ft_deploy_v3_saved_model"
        / "student_pruned_ft_deploy_v3_int8.tflite",
    ),
    (
        "stock_coco_int8",
        "Referencia",
        REPO / "models" / "yolov8n-pose_saved_model" / "yolov8n-pose_int8.tflite",
    ),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    resumen, por_imagen = [], []
    for split_dir in sorted(p for p in SETS.iterdir() if p.is_dir()):
        split = split_dir.name
        reports = []
        for label, group, path in MODELS:
            if not path.exists():
                raise SystemExit(f"No existe el modelo {label}: {path}")
            vis = (
                OUT / "overlays" / split
                if label.endswith(("fold0_int8", "v2_int8", "stock_coco_int8"))
                else None
            )
            if vis:
                vis.mkdir(parents=True, exist_ok=True)
            rep = gate_gt_set(path, split_dir, label=label, save_vis=vis, max_vis=4)
            rep["grupo"] = group
            reports.append(rep)
            s = rep["resumen"]
            # Detección con el umbral de la app: el gate elige la mejor detección sin
            # umbral, así que una persona con score 0,01 cuenta como "detectada" ahí.
            det = sum(1 for r in rep["por_imagen"] if r.get("score", 0.0) >= APP_CONF)
            resumen.append(
                {
                    "set": split,
                    "modelo": label,
                    "grupo": group,
                    **s,
                    "det025": det,
                    "frac_colapsadas": s["colapsadas"] / max(s["n_evaluables"], 1),
                    "frac_det": det / max(s["n_gt_validas"], 1),
                }
            )
            por_imagen += [{"set": split, "modelo": label, **r} for r in rep["por_imagen"]]
            print(
                f"{split:10s} {label:20s} razón={s['razon_pred_gt']:.3f} "
                f"colapsadas={s['colapsadas']}/{s['n_evaluables']} err={s['error_kpt']:.3f}",
                flush=True,
            )
        (OUT / f"gate_{split}.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    pd.DataFrame(resumen).to_csv(OUT / "resumen.csv", index=False)
    pd.DataFrame(por_imagen).to_csv(OUT / "por_imagen.csv", index=False)
    figures(pd.DataFrame(por_imagen), pd.DataFrame(resumen))
    print(f"Resultados en {OUT}")
    return 0


APP_CONF = 0.25  # umbral de detección de la app (y de pose_eval)

LABELS = [
    *[f"cv_fold{k}_int8" for k in range(5)],
    "deploy_fase5_int8",
    "deploy_v2_int8",
    "deploy_v3_int8",
    "stock_coco_int8",
]
NICE = {
    **{f"cv_fold{k}_int8": f"CV F{k}" for k in range(5)},
    "deploy_fase5_int8": "deploy\nFase 5",
    "deploy_v2_int8": "v2\n(roto)",
    "deploy_v3_int8": "v3\n(desplegado)",
    "stock_coco_int8": "stock\nCOCO",
}


def figures(per_img: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from export_fase5_cv_results import (  # noqa: PLC0415
        ACCENT,
        AXIS,
        CONTEXT,
        INK,
        INK_2,
        MUTED,
        SURFACE,
        _c,
        _comma_axis,
        _style,
    )
    from matplotlib.ticker import FixedLocator  # noqa: PLC0415

    _style(plt)
    sets = sorted(per_img["set"].unique())
    x = list(range(len(LABELS)))

    # --- Figura 1: detección a la confianza de la app + error de punto por imagen ---
    fig, axes = plt.subplots(
        len(sets), 2, figsize=(12, 3.6 * len(sets)), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    for row_axes, split in zip(axes, sets):
        ax_det, ax_err = row_axes
        n_img = int(summary[summary.set == split].n_gt_validas.iloc[0])
        for i, lab in enumerate(LABELS):
            sub = per_img[(per_img.set == split) & (per_img.modelo == lab)]
            det = int((sub.score >= APP_CONF).sum())
            ax_det.barh(i, det / n_img * 100, height=0.55, color=ACCENT if i < 5 else CONTEXT)
            ax_det.text(
                det / n_img * 100 + 2, i, f"{det}/{n_img}", va="center", fontsize=8, color=INK
            )
            ev = sub[sub.evaluable & sub.error_kpt.notna()]
            vals = ev.error_kpt.clip(upper=1.5).to_numpy()
            jitter = ((pd.Series(range(len(vals))) % 7 - 3) * 0.06).to_numpy()
            ax_err.scatter(i + jitter, vals, s=10, color=CONTEXT, zorder=2)
            if len(vals):
                ax_err.plot(
                    [i],
                    [float(pd.Series(vals).median())],
                    "o",
                    ms=9,
                    color=ACCENT,
                    mec=SURFACE,
                    mew=2,
                    zorder=3,
                )
        ax_det.set_yticks(x, [NICE[lab].replace("\n", " ") for lab in LABELS], fontsize=8.5)
        ax_det.invert_yaxis()
        ax_det.set_xlim(0, 118)
        ax_det.set_xlabel(f"% de imágenes con persona detectada (conf ≥ {_c(APP_CONF, '.2f')})")
        ax_det.grid(axis="x", color="#e1e0d9")
        ax_det.grid(axis="y", visible=False)
        ax_det.set_title(f"COCO {split} — detección como en la app", loc="left", fontsize=10)
        ax_err.set_yscale("log")
        ax_err.set_ylim(0.008, 2.0)
        ticks = [0.01, 0.05, 0.1, 0.5, 1.0]
        ax_err.yaxis.set_major_locator(FixedLocator(ticks))
        ax_err.yaxis.set_minor_locator(FixedLocator([]))
        _comma_axis(ax_err.yaxis, ".2f")
        ax_err.axhline(0.05, color=AXIS, lw=1)
        ax_err.text(
            len(LABELS) - 0.5,
            0.052,
            "criterio de aprobación ≤ 0,05",
            ha="right",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
        ax_err.set_xticks(x, [NICE[lab] for lab in LABELS], fontsize=8)
        ax_err.set_ylabel("error de punto\n(fracción de la diagonal GT)")
        ax_err.set_title(
            f"COCO {split} — error de punto por imagen (mediana en azul)", loc="left", fontsize=10
        )
    fig.text(
        0.0,
        -0.02,
        "Barras azules: modelos de la CV. Error: sólo imágenes con esqueleto "
        "utilizable (≥ 4 puntos con conf > 0,5); escala logarítmica, valores > 1,5 "
        "recortados.",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(OUT / "compuerta_ood.png")
    plt.close(fig)

    # --- Figura 2: la métrica de la tabla del informe (razón pred/GT) ---
    fig, axes = plt.subplots(len(sets), 1, figsize=(9, 3.6 * len(sets)), sharex=True)
    for ax, split in zip(axes, sets):
        ax.axhline(0.6, color=AXIS, lw=1)
        ax.text(
            len(LABELS) - 0.45,
            0.62,
            "umbral de colapso 0,6",
            ha="right",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
        for i, lab in enumerate(LABELS):
            sub = per_img[(per_img.set == split) & (per_img.modelo == lab) & per_img.evaluable]
            vals = sub.razon_pred_gt.clip(upper=2.0).to_numpy()
            jitter = ((pd.Series(range(len(vals))) % 7 - 3) * 0.05).to_numpy()
            ax.scatter(i + jitter, vals, s=10, color=CONTEXT, zorder=2)
            row = summary[(summary.set == split) & (summary.modelo == lab)].iloc[0]
            ax.plot([i], [row.razon_pred_gt], "o", ms=9, color=ACCENT, mec=SURFACE, mew=2, zorder=3)
            ax.text(
                i,
                1.03,
                f"{_c(row.razon_pred_gt, '.2f')}\n{int(row.colapsadas)}/{int(row.n_evaluables)}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
        ax.set_ylim(0, 2.05)
        ax.yaxis.set_major_locator(
            FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        )
        ax.set_ylabel("extensión x pred / GT")
        _comma_axis(ax.yaxis, ".1f")
        n_img = int(summary[summary.set == split].n_gt_validas.iloc[0])
        ax.set_title(
            f"COCO {split} ({n_img} imágenes con GT real)", loc="left", fontsize=10, pad=30
        )
    axes[-1].set_xticks(x, [NICE[lab] for lab in LABELS], fontsize=8.5)
    fig.text(
        0.0,
        -0.01,
        "Arriba de cada modelo: razón media y colapsadas / evaluadas (razón < "
        "0,6). Puntos grises: cada imagen evaluable; valores > 2 recortados.",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(OUT / "razon_por_modelo.png")
    plt.close(fig)


if __name__ == "__main__":
    if "--solo-figuras" in sys.argv:
        figures(pd.read_csv(OUT / "por_imagen.csv"), pd.read_csv(OUT / "resumen.csv"))
        raise SystemExit(0)
    raise SystemExit(main())
