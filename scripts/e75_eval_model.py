"""E75: evalúa un modelo de pose contra las dos referencias del fold de validación.

Responde a la observación del E50 sobre el origen de los datos: además de medir la
concordancia con las pseudo-etiquetas de ViTPose-H (OKS/PCK, que es lo que ya reporta
el informe), mide la exactitud contra la **anotación original de UCO**, la referencia
instrumental (OptiTrack proyectado a cada cámara, completado a mano en 2D donde el
marcador no se recuperó; Aguilar-Ortega et al., 2023).

Una sola pasada de predicciones alimenta las dos comparaciones:

* **Contra las pseudo-etiquetas**: :func:`src.evaluation.pose_eval.score_stage` con el
  ``eval_ctx.pkl`` del fold de la CV de la Fase 5 (mismo GT, mismo bbox de persona
  normalizador y mismos cuadros que la Tabla de la CV), así los números son
  comparables con los ya publicados.
* **Contra la anotación original**: sólo las articulaciones que UCO anota en ese clip
  (tres por cuadro). PCK@0,05 y @0,10 y error con la distancia normalizada por el
  **alto del cuadro**, la misma normalización de la sec. 5.4 del informe (Test Lab).
  Un cuadro sin detección cuenta como fallo en el PCK y queda fuera del error mediano
  (se reporta la tasa de detección aparte). No se calcula OKS: sus constantes por punto
  están calibradas para los 17 puntos de COCO con anotación manual, y la referencia
  instrumental tiene tres articulaciones con otra convención.

Modelos admitidos en ``--model``: una ruta ``.pt`` o ``.tflite``; ``pseudo`` (las
pseudo-etiquetas, techo de la comparación instrumental); ``base_pt`` (las predicciones
del YOLOv8n-pose COCO guardadas en el contexto, sin re-inferir).

Las predicciones se cachean en ``reports/e75/preds/<tag>_fold<k>.npz`` y se guardan
cada 500 cuadros: una evaluación INT8 (≈ 1 h en CPU) se puede cortar y retomar.

Uso (venv-torch, desde la raíz del repo)::

    venv-torch\\Scripts\\python.exe scripts\\e75_eval_model.py --model <ruta> --fold 0 --tag v3_int8
    ... --export-int8-from runs/v3_cv/fold1/student_pruned_ft.pt --fold 1 --tag v3cv_int8 --gate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.uco_to_yolo import parse_stem  # noqa: E402
from src.inference.keypoint_schema import COCO_KEYPOINT_NAMES  # noqa: E402

logger = logging.getLogger("e75_eval")

GT_LABELS = REPO / "data" / "processed" / "uco_yolo" / "labels" / "all"
PSEUDO_LABELS = REPO / "data" / "processed" / "uco_pseudo" / "labels" / "all"
CTX_DIR = REPO / "runs" / "fase5_cv"
OUT_DIR = REPO / "reports" / "e75"
GATE_SETS = REPO / "data" / "processed" / "ood_gate_coco"
LEG = frozenset({"01", "02", "03", "04", "05", "06", "07", "08"})
ARM = frozenset({"09", "13"})
THRESHOLDS = (0.05, 0.10)
SAVE_EVERY = 500


# --------------------------------------------------------------------------- utilidades


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "scripts"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("+cambios" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def md5(path: Path) -> str:
    import hashlib  # noqa: PLC0415

    h = hashlib.md5()  # noqa: S324 — huella de archivo, no seguridad
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_tflite_threads(n: int) -> None:
    """Hace que el ``tf.lite.Interpreter`` que crea Ultralytics use ``n`` hilos.

    Ultralytics 8.3 crea el intérprete sin ``num_threads``. El número de hilos reparte
    el trabajo de cada operación pero no cambia la aritmética, así que no altera las
    predicciones; sólo el tiempo de una evaluación INT8 sobre CPU.
    """
    import tensorflow as tf  # noqa: PLC0415

    original = tf.lite.Interpreter

    class _Interpreter(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            kwargs.setdefault("num_threads", n)
            super().__init__(*args, **kwargs)

    tf.lite.Interpreter = _Interpreter


def load_ctx(fold: int) -> Any:  # noqa: ANN401
    path = CTX_DIR / f"fold{fold}" / "eval_ctx.pkl"
    with path.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301 — archivo propio de la CV


def _read_label(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    first = next((ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()), None)
    if first is None:
        return None
    vals = np.array(first.split()[5 : 5 + 51], dtype=np.float64)
    return vals.reshape(17, 3) if vals.size == 51 else None


def load_instr_gt(paths: list[Path], dims: dict[Path, tuple[int, int]]) -> dict[Path, np.ndarray]:
    """Anotación original de UCO (17, 3) en píxeles; ``v=0`` donde UCO no anota."""
    out: dict[Path, np.ndarray] = {}
    for p in paths:
        lab = _read_label(GT_LABELS / f"{p.stem}.txt")
        if lab is None:
            continue
        w, h = dims[p]
        lab[:, 0] *= w
        lab[:, 1] *= h
        out[p] = lab
    return out


# ---------------------------------------------------------------------- predicciones


def _cache_path(tag: str, fold: int) -> Path:
    return OUT_DIR / "preds" / f"{tag}_fold{fold}.npz"


def _save_cache(path: Path, preds: dict[str, np.ndarray | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stems = sorted(preds)
    kpts = np.full((len(stems), 17, 3), np.nan, dtype=np.float32)
    for i, s in enumerate(stems):
        if preds[s] is not None:
            kpts[i] = preds[s]
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, stems=np.array(stems), kpts=kpts)
    os.replace(tmp, path)


def _load_cache(path: Path) -> dict[str, np.ndarray | None]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    out: dict[str, np.ndarray | None] = {}
    for s, k in zip(data["stems"], data["kpts"], strict=True):
        out[str(s)] = None if np.isnan(k).all() else k.astype(np.float64)
    return out


def predict(model: str, ctx: Any, tag: str, fold: int, limit: int | None) -> dict[Path, Any]:  # noqa: ANN401
    """``{path: kpts (17,3) px | None}`` para los cuadros del contexto."""
    paths = ctx.img_paths[:limit] if limit else ctx.img_paths
    if model == "base_pt":
        return {p: ctx.base_preds[p]["kpts"] for p in paths}
    if model == "pseudo":
        out: dict[Path, Any] = {}
        for p in paths:
            lab = _read_label(PSEUDO_LABELS / f"{p.stem}.txt")
            if lab is None:
                out[p] = None
                continue
            w, h = ctx.dims[p]
            lab[:, 0] *= w
            lab[:, 1] *= h
            lab[lab[:, 2] <= 0, :2] = np.nan  # punto descartado por el maestro: sin coordenada
            out[p] = lab
        return out

    from src.evaluation import pose_eval  # noqa: PLC0415

    cache = _cache_path(tag, fold) if not limit else None
    done = _load_cache(cache) if cache else {}
    todo = [p for p in paths if p.stem not in done]
    logger.info("%s fold %d: %d cuadros en caché, %d por predecir.", tag, fold, len(done), len(todo))
    from ultralytics import YOLO  # noqa: PLC0415

    yolo = YOLO(str(model), task="pose")
    t0 = time.perf_counter()
    for i in range(0, len(todo), SAVE_EVERY):
        chunk = todo[i : i + SAVE_EVERY]
        res = pose_eval._predict_each(yolo, chunk, 640, 0.25)  # noqa: SLF001 — mismo camino que la CV
        for p, r in res.items():
            done[p.stem] = r["kpts"]
        if cache:
            _save_cache(cache, done)
        rate = (time.perf_counter() - t0) / (i + len(chunk))
        logger.info(
            "  %d/%d (%.0f ms/cuadro, faltan ~%.0f min)",
            i + len(chunk),
            len(todo),
            rate * 1e3,
            rate * (len(todo) - i - len(chunk)) / 60,
        )
    return {p: done.get(p.stem) for p in paths}


# --------------------------------------------------------------------------- métricas


def score_vs_pseudo(pred: dict[Path, Any], ctx: Any) -> dict[str, float]:  # noqa: ANN401
    from src.evaluation import pose_eval  # noqa: PLC0415

    paths = [p for p in ctx.img_paths if p in pred]
    sub = pose_eval.EvalContext(
        gt_kpts={p: ctx.gt_kpts[p] for p in paths},
        person_boxes={p: ctx.person_boxes[p] for p in paths},
        dims={p: ctx.dims[p] for p in paths},
        img_paths=paths,
        base_preds={},
    )
    pred_map = {
        p: {"kpts": None if k is None else np.nan_to_num(k, nan=-1e6)} for p, k in pred.items()
    }
    s = pose_eval.score_stage(pred_map, sub)
    return {"oks": s.oks, "pck02": s.pck, "pck02_pierna": s.pck_leg, "pck02_brazo": s.pck_arm,
            "n_cuadros": s.n_frames}


def score_vs_instrumental(
    pred: dict[Path, Any], gt: dict[Path, np.ndarray], dims: dict[Path, tuple[int, int]]
) -> dict[str, Any]:
    """Error por articulación normalizado por el alto del cuadro."""
    rows: list[tuple[str, int, float, float, float, bool]] = []  # zona, slot, err, dx, dy, det
    n_frames = n_det = 0
    for p, g in gt.items():
        if p not in pred:
            continue
        n_frames += 1
        k = pred[p]
        has_det = k is not None
        n_det += int(has_det)
        h = float(dims[p][1])
        zone = "pierna" if parse_stem(p.stem).exercise in LEG else "brazo"
        for j in np.flatnonzero(g[:, 2] > 0):
            if not has_det or np.isnan(k[j, :2]).any():
                rows.append((zone, int(j), np.nan, np.nan, np.nan, False))
                continue
            dx = (k[j, 0] - g[j, 0]) / h
            dy = (k[j, 1] - g[j, 1]) / h
            rows.append((zone, int(j), float(np.hypot(dx, dy)), float(dx), float(dy), True))

    def summ(sel: list[tuple[str, int, float, float, float, bool]]) -> dict[str, Any]:
        if not sel:
            return {}
        err = np.array([r[2] for r in sel])
        ok = ~np.isnan(err)
        out: dict[str, Any] = {"n_articulaciones": len(sel), "con_prediccion": float(ok.mean())}
        for t in THRESHOLDS:
            out[f"pck{t:.2f}"] = float(np.mean(np.where(ok, err <= t, False)))
        if ok.any():
            out["error_mediano"] = float(np.median(err[ok]))
            out["error_p90"] = float(np.percentile(err[ok], 90))
            out["dx_mediano"] = float(np.median(np.array([r[3] for r in sel])[ok]))
            out["dy_mediano"] = float(np.median(np.array([r[4] for r in sel])[ok]))
        return out

    return {
        "n_cuadros": n_frames,
        "deteccion": n_det / n_frames if n_frames else None,
        "global": summ(rows),
        "por_zona": {z: summ([r for r in rows if r[0] == z]) for z in ("pierna", "brazo")},
        "por_articulacion": {
            COCO_KEYPOINT_NAMES[j]: summ([r for r in rows if r[1] == j])
            for j in sorted({r[1] for r in rows})
        },
    }


# ------------------------------------------------------------------------ export INT8


def export_int8(ckpt: Path, fold: int) -> Path:
    """INT8 con la misma receta que ``configs/quantize_deploy_v3.yaml`` (fold ``k``)."""
    from src.evaluation import pose_eval  # noqa: PLC0415
    from src.models.registry import build_model  # noqa: PLC0415
    from src.optimization import quantization as quant_utils  # noqa: PLC0415

    data = REPO / "data" / "processed" / f"pose_dataset_pseudo_fold{fold}.yaml"
    config = {"family": "yolo", "model": str(ckpt), "data": str(data), "imgsz": 640, "device": 0}
    quant_utils.prepare_calibration_dataset(config)
    model = build_model(config)
    quant_utils.cap_calibration_fraction(model.config, len(pose_eval.read_val_image_paths(data)))
    out = quant_utils.quantize_to_int8(model, config, ckpt.parent)
    logger.info("INT8 exportado: %s", out)
    return out


# ------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="ruta .pt/.tflite, 'pseudo' o 'base_pt'")
    ap.add_argument("--export-int8-from", type=Path, help="checkpoint .pt a cuantizar y evaluar")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=4, help="hilos del intérprete TFLite")
    ap.add_argument("--limit", type=int, default=None, help="sólo los primeros N cuadros (prueba)")
    ap.add_argument("--gate", action="store_true", help="correr también la compuerta OOD")
    ap.add_argument("--mlflow-run-id", default=None, help="loguear las métricas en ese run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    model = args.model
    if args.export_int8_from:
        model = str(export_int8(args.export_int8_from, args.fold))
    if not model:
        ap.error("falta --model o --export-int8-from")
    if model.endswith(".tflite"):
        patch_tflite_threads(args.threads)

    ctx = load_ctx(args.fold)
    pred = predict(model, ctx, args.tag, args.fold, args.limit)
    gt = load_instr_gt(list(pred), ctx.dims)
    result: dict[str, Any] = {
        "tag": args.tag,
        "fold": args.fold,
        "modelo": model,
        "md5": md5(Path(model)) if Path(model).exists() else None,
        "commit": git_commit(),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "limit": args.limit,
        "vs_pseudo": score_vs_pseudo(pred, ctx),
        "vs_instrumental": score_vs_instrumental(pred, gt, ctx.dims),
    }
    if args.gate and Path(model).exists():
        sys.path.insert(0, str(REPO / "scripts"))
        from diagnose_kpt_ratio import gate_gt_set  # noqa: PLC0415

        result["compuerta"] = {}
        for s in ("train2017", "val2017"):
            rep = gate_gt_set(Path(model), GATE_SETS / s, label=args.tag)
            # Detección con el umbral de la app (0,25), como scripts/run_ood_gate_fase5.py:
            # el gate toma la mejor detección sin umbral.
            det = sum(1 for r in rep["por_imagen"] if r.get("score", 0.0) >= 0.25)
            result["compuerta"][s] = {**rep["resumen"], "det025": det}

    out = OUT_DIR / "eval" / f"{args.tag}_fold{args.fold}{'_limit' if args.limit else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    logger.info("Resultado en %s", out)
    logger.info("vs pseudo: %s", result["vs_pseudo"])
    logger.info("vs instrumental (global): %s", result["vs_instrumental"]["global"])

    if args.mlflow_run_id and not args.limit:
        from mlflow.tracking import MlflowClient  # noqa: PLC0415

        client = MlflowClient()
        pre = f"e75_{args.tag}"
        flat = {f"{pre}_{k}": v for k, v in result["vs_pseudo"].items() if isinstance(v, float)}
        g = result["vs_instrumental"]["global"]
        flat |= {f"{pre}_instr_{k}": v for k, v in g.items() if isinstance(v, float)}
        for s, res in result.get("compuerta", {}).items():
            for key in ("razon_pred_gt", "colapsadas", "n_evaluables", "error_kpt", "det025"):
                if isinstance(res.get(key), (int, float)):
                    flat[f"{pre}_gate_{s}_{key}"] = float(res[key])
        for k, v in flat.items():
            client.log_metric(args.mlflow_run_id, k.replace(".", ""), v)
        client.log_artifact(args.mlflow_run_id, str(out), artifact_path="e75")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
