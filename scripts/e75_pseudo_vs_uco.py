"""E75: pseudo-etiquetas de ViTPose-H contra la anotación original de UCO, por articulación.

La anotación original de UCO es la referencia instrumental del conjunto: un sistema
OptiTrack de seis cámaras infrarrojas con marcadores en hombro, codo y muñeca (o cadera,
rodilla y tobillo), proyectado a cada una de las cinco cámaras RGB de 1280×720, y
completado a mano en 2D en los cuadros donde no se recuperó alguna articulación
(Aguilar-Ortega et al., 2023). Cubre las tres articulaciones del ejercicio de cada clip.

Este script compara, sobre todos los cuadros del pool donde existen las dos fuentes, el
punto que produjo el maestro con el de la anotación original:

* PCK@0,05 y @0,10 y error con la distancia normalizada por el **alto del cuadro** (la
  normalización de la sec. 5.4 del informe);
* **desplazamiento con signo** mediano (dx, dy), que separa una diferencia sistemática
  de convención (el marcador señala un punto de la superficie del cuerpo, que puede no
  coincidir con la articulación que define COCO) del error aleatorio del anotador;
* cobertura: fracción de los puntos anotados por UCO que el maestro conservó (``v > 0``).

Sólo lee etiquetas; no usa modelos. Reutiliza :func:`src.evaluation.metrics.compute_pck`.

Uso::

    venv-torch\\Scripts\\python.exe scripts\\e75_pseudo_vs_uco.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from e75_eval_model import _read_label, git_commit  # noqa: E402

from src.data.uco_to_yolo import parse_stem  # noqa: E402
from src.evaluation.metrics import compute_pck  # noqa: E402
from src.inference.keypoint_schema import COCO_KEYPOINT_NAMES  # noqa: E402

GT = REPO / "data" / "processed" / "uco_yolo" / "labels" / "all"
PSEUDO = REPO / "data" / "processed" / "uco_pseudo" / "labels" / "all"
OUT = REPO / "reports" / "e75" / "pseudo_vs_uco.json"
W, H = 1280, 720  # todos los videos UCO (se verifica en configs/audit_cvat.yaml)
LEG = frozenset({"01", "02", "03", "04", "05", "06", "07", "08"})


def main() -> int:
    per_joint: dict[int, list[tuple[float, float, bool]]] = defaultdict(list)  # dx, dy, cubierto
    per_zone: dict[str, list[tuple[float, float, bool]]] = defaultdict(list)
    per_cam: dict[str, list[tuple[float, float, bool]]] = defaultdict(list)
    frame_max_px: list[float] = []
    n_gt = n_pseudo_vacio = 0
    for gt_path in sorted(GT.glob("*.txt")):
        g = _read_label(gt_path)
        if g is None:
            continue
        n_gt += 1
        p = _read_label(PSEUDO / gt_path.name)
        if p is None:
            n_pseudo_vacio += 1  # el maestro no detectó persona: el cuadro no está en el pool
            continue
        parts = parse_stem(gt_path.stem)
        zone = "pierna" if parts.exercise in LEG else "brazo"
        errs = []
        for j in np.flatnonzero(g[:, 2] > 0):
            if p[j, 2] <= 0:
                rec = (np.nan, np.nan, False)
            else:
                rec = ((p[j, 0] - g[j, 0]) * W / H, (p[j, 1] - g[j, 1]), True)  # fracción del alto
                errs.append(float(np.hypot(*rec[:2])) * H)
            per_joint[int(j)].append(rec)
            per_zone[zone].append(rec)
            per_cam[f"cam{parts.cam}"].append(rec)
        if errs:
            frame_max_px.append(max(errs))

    def summ(recs: list[tuple[float, float, bool]]) -> dict[str, float]:
        arr = np.array([(r[0], r[1]) for r in recs], dtype=np.float64)
        cov = np.array([r[2] for r in recs])
        n = len(recs)
        # compute_pck sobre (N, 1, ·): referencia = alto del cuadro, ya normalizado (=1).
        pred = np.nan_to_num(arr, nan=1e6).reshape(n, 1, 2)
        gt = np.zeros((n, 1, 3))
        gt[:, 0, 2] = 2
        ref = np.ones(n)
        err = np.hypot(arr[cov, 0], arr[cov, 1])
        return {
            "n": n,
            "cobertura": float(cov.mean()),
            "pck0.05": compute_pck(pred, gt, 0.05, reference_size=ref),
            "pck0.10": compute_pck(pred, gt, 0.10, reference_size=ref),
            "error_mediano": float(np.median(err)),
            "error_p90": float(np.percentile(err, 90)),
            "dx_mediano": float(np.median(arr[cov, 0])),
            "dy_mediano": float(np.median(arr[cov, 1])),
            "dx_mediano_px": float(np.median(arr[cov, 0]) * H),
            "dy_mediano_px": float(np.median(arr[cov, 1]) * H),
        }

    all_recs = [r for recs in per_joint.values() for r in recs]
    fm = np.array(frame_max_px)
    result = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "normalizacion": "distancia / alto del cuadro (720 px); dx, dy = pseudo - original",
        "cuadros_con_anotacion_original": n_gt,
        "cuadros_sin_pseudo_etiqueta": n_pseudo_vacio,
        "cuadros_comparados": n_gt - n_pseudo_vacio,
        "global": summ(all_recs),
        "por_zona": {z: summ(r) for z, r in sorted(per_zone.items())},
        "por_camara": {c: summ(r) for c, r in sorted(per_cam.items())},
        "por_articulacion": {COCO_KEYPOINT_NAMES[j]: summ(r) for j, r in sorted(per_joint.items())},
        "error_maximo_por_cuadro_px": {
            "p50": float(np.percentile(fm, 50)),
            "p95": float(np.percentile(fm, 95)),
            "p99": float(np.percentile(fm, 99)),
            "mayor_a_100px": int((fm > 100).sum()),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("cuadros_comparados", "global")}, indent=2))
    for name, s in result["por_articulacion"].items():
        print(
            f"{name:15s} n={s['n']:6d} cob={s['cobertura']:.4f} pck05={s['pck0.05']:.3f} "
            f"pck10={s['pck0.10']:.3f} med={s['error_mediano']:.4f} "
            f"dx={s['dx_mediano_px']:+.1f}px dy={s['dy_mediano_px']:+.1f}px"
        )
    print(result["error_maximo_por_cuadro_px"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
