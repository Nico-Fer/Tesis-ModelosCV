"""E75: consolida las mediciones de ``reports/e75/`` en ``resultados.json`` y ``resultados.md``.

Lee lo que dejaron ``scripts/e75_pseudo_vs_uco.py``, ``scripts/e75_eval_model.py`` y
``scripts/run_v3_cv.py``, más la compuerta de la CV de la Fase 5
(``resultados/fase5_cv/gate_ood/resumen.csv``) y la selección de la auditoría CVAT. No
recalcula nada: se puede volver a correr cuando termine la CV del v3.

Convención de dispersión: desvío estándar **poblacional** entre particiones, el mismo que
consolida ``src/training/_kfold.py`` y que reporta el informe.

Uso::

    venv-torch\\Scripts\\python.exe scripts\\e75_build_report.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from e75_eval_model import git_commit  # noqa: E402

E75 = REPO / "reports" / "e75"
EVAL = E75 / "eval"
GATE_CV5 = REPO / "resultados" / "fase5_cv" / "gate_ood" / "resumen.csv"
AUDIT = REPO / "data" / "processed" / "uco_audit"
V3CV_STATE = REPO / "runs" / "v3_cv" / "state.json"
N_FOLDS = 5


def _load(tag: str, fold: int) -> dict[str, Any] | None:
    path = EVAL / f"{tag}_fold{fold}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _row(d: dict[str, Any]) -> dict[str, Any]:
    v = d["vs_instrumental"]
    out = {
        "modelo": d["modelo"],
        "md5": d["md5"],
        "commit": d["commit"],
        "oks_vs_pseudo": d["vs_pseudo"]["oks"],
        "pck02_vs_pseudo": d["vs_pseudo"]["pck02"],
        "deteccion": v["deteccion"],
    }
    for name, s in (("total", v["global"]), *v["por_zona"].items()):
        for key in ("pck0.05", "pck0.10", "error_mediano"):
            out[f"{name}_{key}"] = s.get(key)
    if "compuerta" in d:
        out["compuerta"] = d["compuerta"]
    return out


def _mean_sd(vals: list[float]) -> dict[str, float]:
    return {"media": statistics.fmean(vals), "desvio": statistics.pstdev(vals), "n": len(vals)}


def _gate_cv5() -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with GATE_CV5.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["set"] == "train2017" and r["modelo"].startswith("cv_fold") and "int8" in r["modelo"]:
                k = int(r["modelo"][len("cv_fold")])
                out[k] = {
                    "razon_pred_gt": float(r["razon_pred_gt"]),
                    "colapsadas": int(r["colapsadas"]),
                    "n_evaluables": int(r["n_evaluables"]),
                    "error_kpt": float(r["error_kpt"]),
                    "det025": int(r["det025"]),
                    "n_imagenes": int(r["n_imagenes"]),
                }
    return out


def _cv(tag: str, gate: dict[int, dict[str, float]] | None) -> dict[str, Any]:
    folds: dict[int, dict[str, Any]] = {}
    for k in range(N_FOLDS):
        d = _load(tag, k)
        if d is None:
            continue
        row = _row(d)
        if gate is not None and k in gate:
            row["compuerta_train2017"] = gate[k]
        elif "compuerta" in row:
            row["compuerta_train2017"] = row["compuerta"]["train2017"]
        folds[k] = row
    res: dict[str, Any] = {"particiones": folds, "completa": len(folds) == N_FOLDS}
    if res["completa"]:
        keys = ["oks_vs_pseudo", "pck02_vs_pseudo", "total_pck0.05", "total_pck0.10",
                "total_error_mediano", "pierna_pck0.05", "brazo_pck0.05"]
        res["agregado"] = {key: _mean_sd([folds[k][key] for k in folds]) for key in keys}
        if all("compuerta_train2017" in f for f in folds.values()):
            g = [f["compuerta_train2017"] for f in folds.values()]
            res["agregado"]["compuerta_error_kpt"] = _mean_sd([x["error_kpt"] for x in g])
            res["agregado"]["compuerta_razon"] = _mean_sd([x["razon_pred_gt"] for x in g])
            res["agregado"]["compuerta_colapsadas"] = {
                "suma": sum(x["colapsadas"] for x in g), "de": sum(x["n_evaluables"] for x in g)}
            res["agregado"]["compuerta_det025"] = {
                "suma": sum(x["det025"] for x in g), "de": sum(x["n_imagenes"] for x in g)}
    return res


def _audit() -> dict[str, Any]:
    with (AUDIT / "selection.csv").open(encoding="utf-8") as fh:
        sel = list(csv.DictReader(fh))
    with (AUDIT / "cvat_frames.csv").open(encoding="utf-8") as fh:
        frames = list(csv.DictReader(fh))
    jobs = Counter(r["job_id"] for r in frames)
    exported = list(REPO.glob("data/processed/uco_audit/**/auditados.txt"))
    real = [p for p in exported if "roundtrip" not in str(p)]
    return {
        "seleccionados": len(sel),
        "por_estrato": dict(Counter(r["motivo"] for r in sel).most_common()),
        "montados_en_cvat": len(frames),
        "jobs": len(jobs),
        "tarea_cvat": sorted({r["task_id"] for r in frames}),
        "auditados_exportados": sum(
            len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
            for p in real
        ),
        "audit_report": [str(p.relative_to(REPO)) for p in REPO.glob("data/processed/**/audit_report.json")],
        "nota": "Sólo existe el auditados.txt del gate de round-trip (5 cuadros de prueba).",
    }


def _fmt(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def _md(res: dict[str, Any]) -> str:
    lines = [
        "# E75 — Resultados",
        "",
        f"Generado {res['generado']} por `scripts/e75_build_report.py` (commit {res['commit']}).",
        "Distancias como fracción del alto del cuadro (720 px); dx, dy en px (pseudo − original).",
        "",
        "## 1. Pseudo-etiquetas contra la anotación original de UCO (pool completo)",
        "",
        "| Articulación | n | PCK@0,05 | PCK@0,10 | Error mediano | dx (px) | dy (px) |",
        "|---|---|---|---|---|---|---|",
    ]
    pv = res["pseudo_vs_uco"]
    for name, s in [*pv["por_articulacion"].items(), ("TOTAL", pv["global"])]:
        lines.append(
            f"| {name} | {s['n']} | {_fmt(s['pck0.05'])} | {_fmt(s['pck0.10'])} | "
            f"{_fmt(s['error_mediano'])} | {s['dx_mediano_px']:+.1f} | {s['dy_mediano_px']:+.1f} |"
        )
    lines += [
        "",
        f"Cuadros comparados: {pv['cuadros_comparados']} (sin pseudo-etiqueta: "
        f"{pv['cuadros_sin_pseudo_etiqueta']}). Error máximo por cuadro: "
        f"{pv['error_maximo_por_cuadro_px']}.",
        "",
        "## 2. Auditoría CVAT",
        "",
        "```json",
        json.dumps(res["auditoria_cvat"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## 3. Exactitud contra la referencia instrumental, partición 0 (16.149 cuadros)",
        "",
        "| Modelo | MD5 | OKS vs pseudo | PCK@0,05 | PCK@0,10 | Error | inf. 0,05 | sup. 0,05 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag, r in res["fase2_particion0"].items():
        lines.append(
            f"| {tag} | {(r['md5'] or '')[:8]} | {_fmt(r['oks_vs_pseudo'], 4)} | "
            f"{_fmt(r['total_pck0.05'])} | {_fmt(r['total_pck0.10'])} | "
            f"{_fmt(r['total_error_mediano'])} | {_fmt(r['pierna_pck0.05'])} | "
            f"{_fmt(r['brazo_pck0.05'])} |"
        )
    for name, cv in (("Receta de la Fase 5 (INT8)", res["cv_fase5"]), ("Receta v3 (INT8)", res["cv_v3"])):
        lines += ["", f"## 4. Validación cruzada — {name}", ""]
        if not cv.get("completa"):
            lines.append(f"**En ejecución** ({len(cv['particiones'])}/5 particiones). Sin parciales.")
            continue
        lines += ["| Medida | Media ± σ |", "|---|---|"]
        for key, a in cv["agregado"].items():
            if "media" in a:
                lines.append(f"| {key} | {_fmt(a['media'], 4)} ± {_fmt(a['desvio'], 4)} |")
            else:
                lines.append(f"| {key} | {a['suma']}/{a['de']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    gate5 = _gate_cv5()
    v3_state = json.loads(V3CV_STATE.read_text(encoding="utf-8")) if V3CV_STATE.exists() else {}
    res: dict[str, Any] = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "pseudo_vs_uco": json.loads((E75 / "pseudo_vs_uco.json").read_text(encoding="utf-8")),
        "auditoria_cvat": _audit(),
        "fase2_particion0": {
            t: _row(d)
            for t in ("app_stock_int8", "base_pt", "v2_fp32", "v2_int8", "v3_fp32", "v3_int8", "pseudo")
            if (d := _load(t, 0)) is not None
        },
        "base_pt_por_particion": {k: _row(d) for k in range(N_FOLDS) if (d := _load("base_pt", k))},
        "pseudo_por_particion": {k: _row(d) for k in range(N_FOLDS) if (d := _load("pseudo", k))},
        "cv_fase5": _cv("cv5_int8", gate5),
        "cv_v3": {**_cv("v3cv_int8", None), "estado_corrida": v3_state},
        "cv_v3_fp32": _cv("v3cv_fp32", None),
    }
    (E75 / "resultados.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (E75 / "resultados.md").write_text(_md(res), encoding="utf-8")
    print((E75 / "resultados.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
