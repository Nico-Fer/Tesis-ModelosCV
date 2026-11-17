"""Diagnóstico: ¿el OKS 0.000 tras podar yolov8n-pose a sparsity 0.3 es real o un bug?

Aísla, sobre una muestra fija del val del fold, cada eslabón del camino de poda de
:mod:`src.optimization.pruning` (sin tocarlo) y mide OKS/PCK con el evaluador del
pipeline (:mod:`src.evaluation.pose_eval`):

    | ID | Variante                                   | Qué responde                           |
    |----|--------------------------------------------|----------------------------------------|
    | B0 | base                                       | cordura (≈0.846 en el split completo)  |
    | B1 | sólo C2f → C2f_v2                          | ¿la cirugía es equivalente?            |
    | B2 | B1 + eps/momentum de BN como Ultralytics   | aísla el eps de los BN nuevos          |
    | B3 | _prune_yolo_module(sparsity=0)             | integridad del camino de poda          |
    | B4 | barrido de sparsity (con y sin fix de eps) | curva gradual (real) vs precipicio     |
    | B5 | conf 0.25 vs 0.001, tasa de detección      | ¿umbral o keypoints destruidos?        |
    | B6 | recalibración de BN post-poda              | ¿estadísticas o features?              |
    | B7 | método DeepLearningCourse (L1 unstructured)| "30 %" de pesos vs "30 %" de canales   |
    | B8 | nº de BN antes/después de predecir en RAM  | fusión in-place de combined.py (fixed) |

Resultado (2026-09-29, 600 frames fold0): el colapso es real — B3 ≈ base, pero cualquier
sparsity ≥ 0.05 da OKS 0 aun con conf 0.001. B1/B8 destaparon dos bugs colaterales ya
corregidos (eps de los BN de C2f_v2; fusión in-place en run_predictions): tras el fix, B1
y B3 deben igualar a B0 y B8 debe dar Δparams = 0.

Cada variante parte de un checkpoint recién cargado y se evalúa sobre una
``deepcopy`` del módulo: ``YOLO.predict`` fusiona Conv+BN *in-place* (AutoBackend
``fuse=True``), así que evaluar el original contaminaría la variante siguiente.

Uso (venv-torch, sin MLflow):
    venv-torch/Scripts/python.exe scripts/diagnose_pruning.py [--sample 600]
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diagnose_kpt_ratio import letterbox_app, uco_frames_by_camera  # noqa: E402

IMGSZ = 640
_BASE_MODEL = REPO_ROOT / "models" / "yolov8n-pose.pt"
_DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "pose_dataset_pseudo_fold0.yaml"
_PRUNED_FT = REPO_ROOT / "models" / "exports" / "student_pruned_ft.pt"
_SWEEP = (0.05, 0.1, 0.2, 0.3)


# --------------------------------------------------------------------------- utils


def fresh_module() -> Any:  # noqa: ANN401
    """``nn.Module`` recién cargado del checkpoint base (CPU)."""
    from ultralytics import YOLO  # noqa: PLC0415

    return YOLO(str(_BASE_MODEL)).model


def wrap(module: Any) -> Any:  # noqa: ANN401
    """Wrapper ``YOLO`` sobre una **copia** del módulo (el predict la fusiona)."""
    from ultralytics import YOLO  # noqa: PLC0415

    yolo = YOLO(str(_BASE_MODEL))
    yolo.model = copy.deepcopy(module)
    return yolo


def n_params(module: Any) -> int:  # noqa: ANN401
    return int(sum(p.numel() for p in module.parameters()))


def bn_layers(module: Any) -> list[Any]:  # noqa: ANN401
    import torch.nn as nn  # noqa: PLC0415

    return [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]


def to_tensor(paths: list[Path], device: Any) -> Any:  # noqa: ANN401
    """Batch NCHW float32 [0,1] RGB letterboxeado a 640 (mismo preproceso que la app)."""
    import torch  # noqa: PLC0415

    arrs = [letterbox_app(cv2.imread(str(p)))[0][0] for p in paths]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous().to(device)


def raw_output(module: Any, batch: Any) -> Any:  # noqa: ANN401
    """Salida cruda ``(B, 56, 8400)`` del módulo en eval, sin fusionar."""
    import torch  # noqa: PLC0415

    m = copy.deepcopy(module).eval().to(batch.device)
    with torch.no_grad():
        out = m(batch)
    return out[0] if isinstance(out, (tuple, list)) else out


# --------------------------------------------------------------------------- eval


def build_ctx(data_yaml: Path, n: int) -> Any:  # noqa: ANN401
    """Contexto de evaluación sobre una muestra estratificada por cámara.

    Mismo armado que ``diagnose_kpt_ratio.cmd_oks_sample``: GT pseudo + bbox de persona
    del detector base (normalizador fijo, idéntico al de combined.py).
    """
    from src.evaluation import pose_eval  # noqa: PLC0415

    sample: list[Path] = []
    for paths in uco_frames_by_camera(data_yaml, per_cam=max(1, n // 5)).values():
        sample += paths
    gt_norm = pose_eval.load_gt_normalized(sample)
    base_preds = pose_eval.run_predictions(str(_BASE_MODEL), list(gt_norm), IMGSZ)
    gt_kpts, boxes, dims, kept_preds, kept = {}, {}, {}, {}, []
    for path, gt in gt_norm.items():
        pred = base_preds.get(path)
        if pred is None or pred["box"] is None:
            continue
        w, h = pred["wh"]
        gt_px = gt.copy()
        gt_px[:, 0] *= w
        gt_px[:, 1] *= h
        gt_kpts[path] = gt_px
        boxes[path] = np.asarray(pred["box"], dtype=np.float64)
        dims[path] = (w, h)
        kept_preds[path] = pred
        kept.append(path)
    return pose_eval.EvalContext(
        gt_kpts=gt_kpts, person_boxes=boxes, dims=dims, img_paths=kept, base_preds=kept_preds
    )


def predict(module: Any, paths: list[Path], conf: float) -> tuple[dict, list[float]]:  # noqa: ANN401
    """Como ``pose_eval.run_predictions`` (mismo predict, batch=1) + conf de la mejor det."""
    from src.evaluation import pose_eval  # noqa: PLC0415

    yolo = wrap(module)
    out: dict[Path, dict[str, Any]] = {}
    best_conf: list[float] = []
    for path in paths:
        res = yolo.predict(source=str(path), imgsz=IMGSZ, conf=conf, verbose=False)[0]
        h, w = res.orig_shape
        kpts, box = pose_eval._pick_best_detection(res)
        out[path] = {"kpts": kpts, "box": box, "wh": (int(w), int(h))}
        best_conf.append(float(res.boxes.conf.max()) if len(res.boxes) else 0.0)
    return out, best_conf


ROWS: list[str] = []


def score(tag: str, module: Any, ctx: Any, note: str = "") -> dict[str, float]:  # noqa: ANN401
    """OKS/PCK a conf 0.25 (la del pipeline) y 0.001, tasa de detección y conf mediana."""
    from src.evaluation import pose_eval  # noqa: PLC0415

    res: dict[str, float] = {}
    for conf in (0.25, 0.001):
        preds, confs = predict(module, ctx.img_paths, conf)
        s = pose_eval.score_stage(preds, ctx)
        det = float(np.mean([c > 0 for c in confs]))
        res[f"oks@{conf}"] = float(s.oks or 0.0)
        res[f"pck@{conf}"] = float(s.pck or 0.0)
        res[f"det@{conf}"] = det
        if conf == 0.001:
            res["conf_med"] = float(np.median(confs))
            res["conf_p90"] = float(np.percentile(confs, 90))
    row = (
        f"{tag:<34} params={n_params(module):>9,} "
        f"OKS@.25={res['oks@0.25']:.4f} PCK@.25={res['pck@0.25']:.4f} "
        f"det@.25={res['det@0.25']:.3f} | "
        f"OKS@.001={res['oks@0.001']:.4f} PCK@.001={res['pck@0.001']:.4f} "
        f"conf_med={res['conf_med']:.4f} conf_p90={res['conf_p90']:.4f}  {note}"
    )
    print(row, flush=True)
    ROWS.append(row)
    return res


# --------------------------------------------------------------------------- variantes


def c2f_v2_only(fix_eps: bool) -> Any:  # noqa: ANN401
    from ultralytics.utils.torch_utils import initialize_weights  # noqa: PLC0415

    from src.optimization._c2f_v2 import replace_c2f_with_v2  # noqa: PLC0415

    m = fresh_module()
    replace_c2f_with_v2(m)
    if fix_eps:
        # Lo que hace el ejemplo oficial de torch_pruning tras el reemplazo: BN eps=1e-3,
        # momentum=0.03 (valores de Ultralytics). No toca pesos de Conv2d.
        initialize_weights(m)
    return m.eval()


def pruned(sparsity: float, fix_eps: bool) -> Any:  # noqa: ANN401
    """Camino de poda del pipeline, tal cual; ``fix_eps`` sólo inserta el fix previo."""
    from src.optimization.pruning import _prune_yolo_module  # noqa: PLC0415

    m = c2f_v2_only(fix_eps) if fix_eps else fresh_module()
    # Si ya está reemplazado, replace_c2f_with_v2 dentro de _prune_yolo_module es no-op.
    _prune_yolo_module(m, sparsity=sparsity, imgsz=IMGSZ)
    return m


def recalibrate_bn(module: Any, train_paths: list[Path], batch: int = 16) -> Any:  # noqa: ANN401
    """Re-estima las running stats de BN con imágenes de TRAIN (media acumulada exacta)."""
    import torch  # noqa: PLC0415

    m = copy.deepcopy(module)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = m.to(device)
    bns = bn_layers(m)
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # promedio acumulado sobre todos los batches
    m.train()
    with torch.no_grad():
        for i in range(0, len(train_paths), batch):
            m(to_tensor(train_paths[i : i + batch], device))
    for bn in bns:
        bn.momentum = 0.03
    return m.eval().cpu()


def dlc_unstructured(skip_dfl: bool) -> Any:  # noqa: ANN401
    """Réplica de DeepLearningCourse/.../training/prune.py sobre yolov8n-pose (sin fine-tune)."""
    import torch.nn as nn  # noqa: PLC0415
    import torch.nn.utils.prune as tprune  # noqa: PLC0415

    m = fresh_module()
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            if skip_dfl and name.endswith(".dfl.conv"):
                continue
            tprune.l1_unstructured(mod, name="weight", amount=0.3)
            tprune.remove(mod, "weight")
    return m.eval()


def zero_fraction(module: Any) -> float:  # noqa: ANN401
    import torch.nn as nn  # noqa: PLC0415

    tot = zeros = 0
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            tot += mod.weight.numel()
            zeros += int((mod.weight == 0).sum())
    return zeros / max(tot, 1)


def train_sample(data_yaml: Path, n: int, seed: int = 0) -> list[Path]:
    """``n`` imágenes del split TRAIN (sin fuga hacia el val puntuado)."""
    import yaml  # noqa: PLC0415

    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    train = spec["train"]
    manifests = train if isinstance(train, list) else [train]
    paths: list[Path] = []
    for mf in manifests:
        mf_path = Path(mf)
        if not mf_path.is_absolute():
            mf_path = Path(spec.get("path", data_yaml.parent)) / mf
        paths += [
            Path(ln.strip())
            for ln in mf_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    return random.Random(seed).sample(sorted(set(paths)), n)


# --------------------------------------------------------------------------- main


def main() -> None:
    import torch  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default=str(_DEFAULT_DATA))
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--calib", type=int, default=512, help="imgs de train para recalibrar BN (B6)")
    args = ap.parse_args()
    data = Path(args.data)

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    ctx = build_ctx(data, args.sample)
    print(f"muestra: {len(ctx.img_paths)} frames con bbox base, data={data.name}\n")
    probe = to_tensor(ctx.img_paths[:16], torch.device("cpu"))

    # ---- B0 / B1 / B2 / B3: integridad del camino, sin podar nada -------------------
    base = fresh_module().eval()
    y_base = raw_output(base, probe)
    score("B0 base", base, ctx)

    from src.optimization._c2f_v2 import C2f_v2  # noqa: PLC0415

    b1 = c2f_v2_only(fix_eps=False)
    v2s = [m for m in b1.modules() if isinstance(m, C2f_v2)]
    eps_pairs = sorted(
        {(m.cv0.bn.eps, m.cv0.bn.momentum, m.cv2.bn.eps, m.cv2.bn.momentum) for m in v2s}
    )
    print(f"\n[B1] {len(v2s)} C2f_v2; (eps,momentum) BN nuevos cv0 vs BN original cv2: {eps_pairs}")
    orig_eps = sorted({bn.eps for bn in bn_layers(base)})
    print(f"[B1] eps de TODOS los BN del checkpoint original: {orig_eps}")
    d1 = (raw_output(b1, probe) - y_base).abs()
    print(
        f"[B1] salida cruda vs base (16 imgs): max|Δ|={float(d1.max()):.4g}  "
        f"mean|Δ|={float(d1.mean()):.4g}  (ref mean|y|={float(y_base.abs().mean()):.4g})"
    )
    score("B1 C2f_v2 (eps pipeline)", b1, ctx)

    b2 = c2f_v2_only(fix_eps=True)
    d2 = (raw_output(b2, probe) - y_base).abs()
    print(
        f"[B2] salida cruda vs base: max|Δ|={float(d2.max()):.4g}  mean|Δ|={float(d2.mean()):.4g}"
    )
    score("B2 C2f_v2 + eps fix", b2, ctx)

    b3 = pruned(0.0, fix_eps=False)
    d3 = (raw_output(b3, probe) - y_base).abs()
    print(
        f"[B3] sparsity=0 salida cruda vs base: max|Δ|={float(d3.max()):.4g}  "
        f"params {n_params(b3):,} vs base {n_params(base):,}"
    )
    score("B3 pipeline sparsity=0.0", b3, ctx)
    print()

    # ---- B4 / B5: barrido ------------------------------------------------------------
    models_03: dict[str, Any] = {}
    for fix in (False, True):
        for sp in _SWEEP:
            m = pruned(sp, fix_eps=fix)
            if sp == 0.3:
                models_03["fix" if fix else "pipe"] = m
            score(f"B4 sparsity={sp:<4} {'eps fix' if fix else 'pipeline'}", m, ctx)
        print()

    # ---- B6: recalibración de BN -----------------------------------------------------
    calib = train_sample(data, args.calib)
    print(f"[B6] recalibrando BN con {len(calib)} imgs de train")
    score("B6 base + BN recal (control)", recalibrate_bn(base, calib), ctx)
    score("B6 sp=0.3 pipeline + BN recal", recalibrate_bn(models_03["pipe"], calib), ctx)
    score("B6 sp=0.3 eps fix + BN recal", recalibrate_bn(models_03["fix"], calib), ctx)
    print()

    # ---- B7: método DeepLearningCourse ------------------------------------------------
    for skip in (False, True):
        m = dlc_unstructured(skip_dfl=skip)
        score(
            f"B7 L1-unstr 30% {'sin DFL' if skip else 'fiel (todas)'}",
            m,
            ctx,
            note=f"ceros={zero_fraction(m):.3f}",
        )
    print()

    # ---- B8: fusión in-place al predecir sobre el wrapper en memoria -----------------
    from ultralytics import YOLO  # noqa: PLC0415

    from src.evaluation import pose_eval  # noqa: PLC0415

    live = YOLO(str(_BASE_MODEL))
    live.model = pruned(0.3, fix_eps=False)
    p0, bn0 = n_params(live.model), bn_layers(live.model)
    ch0 = sum(bn.num_features for bn in bn0)
    pose_eval.run_predictions(live, ctx.img_paths[:2], IMGSZ)  # como combined.py:287
    p1, bn1 = n_params(live.model), bn_layers(live.model)
    print(
        f"[B8] podado 0.3 ANTES de run_predictions en RAM: params={p0:,} BN={len(bn0)} "
        f"canales_BN={ch0:,}"
    )
    print(
        f"[B8] DESPUÉS: params={p1:,} BN={len(bn1)}  Δparams={p0 - p1:,} "
        f"(esperado 0 desde el fix de run_predictions; antes fusionaba: Δ = canales_BN = {ch0:,})"
    )
    print("[B8] histórico AGENTS.md: params_pruned=1,907,230  params_pruned_ft=1,903,083  Δ=4,147")
    if _PRUNED_FT.exists():
        ft = YOLO(str(_PRUNED_FT)).model
        print(f"[B8] {_PRUNED_FT.name}: params={n_params(ft):,} BN={len(bn_layers(ft))}")

    print("\n=== RESUMEN ===")
    for r in ROWS:
        print(r)


if __name__ == "__main__":
    main()
