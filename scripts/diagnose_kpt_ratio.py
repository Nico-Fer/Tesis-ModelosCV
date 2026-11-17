"""Diagnóstico del colapso de la coordenada x de keypoints en exports TFLite.

Mide, sobre el tensor de salida CRUDO ``[1, 56, 8400]`` (decode idéntico al de la
app Android, sin Ultralytics), el cociente entre el span de los keypoints y el
ancho/alto del box de la mejor detección:

    ratio_x = (max(kpt_x) - min(kpt_x)) / box_w      (sano: >= 0.75 de frente)
    ratio_y = (max(kpt_y) - min(kpt_y)) / box_h

Corre una matriz modelo x frame que discrimina dónde vive el defecto:

    | Resultado en desktop                                  | Veredicto            |
    |-------------------------------------------------------|----------------------|
    | .pt, _float32 e _int8 colapsan en bus.jpg; en UCO no  | dominio (retrain)    |
    | _int8 colapsa; _float32 y .pt no                      | cuantización         |
    | nada colapsa (incl. bytes desplegados, decode crudo)  | runtime Android      |
    | el control stock también colapsa                      | bug de este script   |

Modos extra:
    --per-camera N   ratio_x GT (labels pseudo) y predicho por cámara UCO.
    --oks-sample N   OKS/PCK muestreado (estratificado por cámara) vía pose_eval.
    --oks-full       OKS/PCK sobre todo el val de --data (p.ej. el subset gold).

Uso (venv-torch, sin MLflow):
    venv-torch/Scripts/python.exe scripts/diagnose_kpt_ratio.py [--per-camera 20]
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMGSZ = 640
# Límites de anchors por nivel FPN con imgsz=640: P3 80x80, P4 40x40, P5 20x20.
_P3_END = 80 * 80
_P4_END = _P3_END + 40 * 40
_KPT_CONF_THR = 0.5

_DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "pose_dataset_pseudo_fold0.yaml"
_BASE_MODEL = REPO_ROOT / "models" / "yolov8n-pose.pt"
_DEPLOY_DIR = REPO_ROOT / "models" / "student_pruned_ft_deploy_v2_saved_model"

# (etiqueta, ruta). El control stock valida el letterbox/decode de este script:
# debe dar ratio_x ~0.8-0.9 en bus.jpg o el bug está acá y no en los modelos.
_DEFAULT_MODELS: list[tuple[str, Path]] = [
    ("deploy_v2_int8", _DEPLOY_DIR / "student_pruned_ft_deploy_v2_int8.tflite"),
    ("deploy_v2_float32", _DEPLOY_DIR / "student_pruned_ft_deploy_v2_float32.tflite"),
    ("deploy_v2_pt", REPO_ROOT / "models" / "student_pruned_ft_deploy_v2.pt"),
    (
        "stock_int8_CONTROL",
        REPO_ROOT / "models" / "yolov8n-pose_saved_model" / "yolov8n-pose_int8.tflite",
    ),
]


def letterbox_app(
    img_bgr: np.ndarray, size: int = IMGSZ
) -> tuple[np.ndarray, tuple[float, int, int]]:
    """Letterbox centrado idéntico al de la app: pad gris 114, RGB, float32/255, NHWC.

    Returns:
        ``(tensor (1,size,size,3) float32, (escala, pad_x, pad_y))``.
    """
    h, w = img_bgr.shape[:2]
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py : py + nh, px : px + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)[None] / 255.0, (r, px, py)


class TfliteModel:
    """Wrapper mínimo de ``tf.lite.Interpreter`` con E/S cuantizada opcional."""

    def __init__(self, path: Path) -> None:
        import tensorflow as tf  # noqa: PLC0415  # diferido: sólo venv-torch

        self.path = path
        self.interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=4)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

    def contract(self) -> str:
        """Describe la E/S real del archivo (chequeo de contrato de la app)."""

        def fmt(d: dict[str, Any]) -> str:
            scale, zp = d["quantization"]
            quant = f" quant(scale={scale:.6g}, zp={zp})" if scale else ""
            return f"{list(d['shape'])} {np.dtype(d['dtype']).name}{quant}"

        return f"input {fmt(self.inp)} -> output {fmt(self.out)}"

    def infer_crop(self, crop_bgr: np.ndarray) -> dict[str, Any] | None:
        """Infiere sobre un recorte en memoria con el letterbox y decode de la app."""
        tensor, _ = letterbox_app(crop_bgr)
        det = decode_best(self(tensor))
        if det is None:
            return None
        rx, ry = ratios(det["kx"], det["ky"], det["kc"], det["box"][2], det["box"][3])
        det["ratio_x"], det["ratio_y"] = rx, ry
        det["tensor"] = tensor
        return det

    def __call__(self, tensor_f32: np.ndarray) -> np.ndarray:
        """Infiere y devuelve la salida ``(56, 8400)`` en float32 desescalado."""
        x: np.ndarray = tensor_f32
        if self.inp["dtype"] in (np.int8, np.uint8, np.int16):
            scale, zp = self.inp["quantization"]
            x = (x / scale + zp).astype(self.inp["dtype"])
        self.interpreter.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])
        if self.out["dtype"] in (np.int8, np.uint8, np.int16):
            scale, zp = self.out["quantization"]
            y = (y.astype(np.float32) - zp) * scale
        y = np.squeeze(y, axis=0)
        if y.shape[0] != 56 and y.shape[-1] == 56:  # guard de layout channel-last
            y = y.T
        return y.astype(np.float32)


def decode_best(out56: np.ndarray) -> dict[str, Any] | None:
    """Mejor detección del tensor crudo: box, kpts, score y nivel FPN del anchor.

    Layout por columna (anchor): ``[cx, cy, w, h, score, (x, y, conf) x 17]``,
    coordenadas normalizadas 0-1 del tensor de entrada (contrato del export).
    """
    conf = out56[4]
    best = int(np.argmax(conf))
    score = float(conf[best])
    if score <= 0.0:
        return None
    level = "P3" if best < _P3_END else ("P4" if best < _P4_END else "P5")
    return {
        "score": score,
        "anchor": best,
        "level": level,
        "box": out56[0:4, best].astype(np.float64),  # cx, cy, w, h
        "kx": out56[5::3, best].astype(np.float64),
        "ky": out56[6::3, best].astype(np.float64),
        "kc": out56[7::3, best].astype(np.float64),
    }


def ratios(
    kx: np.ndarray, ky: np.ndarray, kc: np.ndarray, box_w: float, box_h: float
) -> tuple[float, float]:
    """``(ratio_x, ratio_y)`` sobre los keypoints con conf > umbral (fallback: los 17)."""
    mask = kc > _KPT_CONF_THR
    if int(mask.sum()) < 2:
        mask = np.ones_like(kc, dtype=bool)
    span_x = float(kx[mask].max() - kx[mask].min())
    span_y = float(ky[mask].max() - ky[mask].min())
    return span_x / max(box_w, 1e-9), span_y / max(box_h, 1e-9)


def run_tflite_frame(
    model: TfliteModel, image_path: Path, save_vis: Path | None
) -> dict[str, Any] | None:
    """Camino crudo: letterbox de la app + interpreter + decode sin Ultralytics."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [WARN] no se pudo leer {image_path}")
        return None
    tensor, _ = letterbox_app(img)
    det = decode_best(model(tensor))
    if det is None:
        return None
    rx, ry = ratios(det["kx"], det["ky"], det["kc"], det["box"][2], det["box"][3])
    det["ratio_x"], det["ratio_y"] = rx, ry
    if save_vis is not None:
        _save_overlay(tensor, det, save_vis / f"{model.path.stem}__{image_path.stem}.jpg")
    return det


def _save_overlay(tensor: np.ndarray, det: dict[str, Any], out_path: Path) -> None:
    """Dibuja box + esqueleto sobre el canvas letterboxeado (evidencia visual)."""
    from src.inference.keypoint_schema import COCO_SKELETON  # noqa: PLC0415

    canvas = cv2.cvtColor((tensor[0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cx, cy, w, h = det["box"] * IMGSZ
    p1 = (int(cx - w / 2), int(cy - h / 2))
    p2 = (int(cx + w / 2), int(cy + h / 2))
    cv2.rectangle(canvas, p1, p2, (0, 200, 255), 2)
    pts = np.stack([det["kx"] * IMGSZ, det["ky"] * IMGSZ], axis=1).astype(int)
    for a, b in COCO_SKELETON:
        if det["kc"][a] > _KPT_CONF_THR and det["kc"][b] > _KPT_CONF_THR:
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        if det["kc"][i] > _KPT_CONF_THR:
            cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


class PtModel:
    """Camino PyTorch/Ultralytics para checkpoints ``.pt`` (testea los pesos)."""

    def __init__(self, path: Path) -> None:
        from ultralytics import YOLO  # noqa: PLC0415

        self.path = path
        self.yolo = YOLO(str(path), task="pose")

    def run_frame(self, image_path: Path) -> dict[str, Any] | None:
        """Infiere sobre una imagen de disco (ratios en px de la imagen original)."""
        return self._predict(str(image_path))

    def infer_crop(self, crop_bgr: np.ndarray) -> dict[str, Any] | None:
        """Infiere sobre un recorte en memoria (Ultralytics letterboxea internamente)."""
        return self._predict(crop_bgr)

    def _predict(self, source: Any) -> dict[str, Any] | None:  # noqa: ANN401
        from src.evaluation.pose_eval import _pick_best_detection  # noqa: PLC0415

        result = self.yolo.predict(source=source, imgsz=IMGSZ, conf=0.25, verbose=False)[0]
        kpts, box = _pick_best_detection(result)
        if kpts is None or box is None:
            return None
        rx, ry = ratios(kpts[:, 0], kpts[:, 1], kpts[:, 2], float(box[2]), float(box[3]))
        return {
            "score": float(result.boxes.conf.max()),
            "level": "-",
            "box": np.asarray(box, dtype=np.float64),
            "kpts_px": np.asarray(kpts, dtype=np.float64),
            "ratio_x": rx,
            "ratio_y": ry,
        }


def load_model(path: Path) -> TfliteModel | PtModel:
    """Despacha por extensión: ``.tflite`` al decode crudo, ``.pt`` a Ultralytics."""
    return TfliteModel(path) if path.suffix == ".tflite" else PtModel(path)


def bus_jpg() -> Path:
    """``bus.jpg`` de los assets de Ultralytics (proxy OOD portrait, cuerpo entero)."""
    from ultralytics.utils import ASSETS  # noqa: PLC0415

    return Path(ASSETS) / "bus.jpg"


def cmd_gt_set(args: argparse.Namespace) -> None:
    """GATE cuantitativo: predicho vs GT real sobre un set OOD ``images/`` + ``labels/``.

    Es el criterio de aceptación del fix. A diferencia de ``--crop-image``, acá hay
    ground truth, así que el span de keypoints se normaliza por el **ancho del bbox
    GT** (no por el box predicho, que en un modelo roto también está mal y falsea el
    cociente):

        span_x_rel = (max(kpt_x) - min(kpt_x)) / w_gt

    Se reporta el del GT como techo alcanzable en cada imagen y la razón
    ``pred/GT``: 1.0 = span correcto, <0.6 = colapso. Además el error medio de los
    keypoints (px, normalizado por la diagonal GT) para detectar un esqueleto
    "ancho pero mal puesto".

    El set esperado: ``<dir>/images/*.jpg`` con ``<dir>/labels/<stem>.txt`` en
    formato YOLO-pose de una sola persona.
    """
    root = Path(args.gt_set)
    if args.model:
        paths = [(Path(args.model).stem, Path(args.model))]
    else:
        paths = [_DEFAULT_MODELS[0], _DEFAULT_MODELS[3]]
    save_vis = Path(args.save_vis) if args.save_vis else None

    n_images = len(_gt_set_images(root))
    print(f"\n=== GATE contra GT: {n_images} imágenes OOD de {root.name} ===")
    print("span_x_rel = span keypoints / ancho bbox GT   |   razón = pred / GT")
    reports = []
    for label, model_path in paths:
        rep = gate_gt_set(model_path, root, label=label, save_vis=save_vis)
        reports.append(rep)
        s = rep["resumen"]
        print(
            f"\n[{label}]  n={s['n_evaluables']}/{s['n_imagenes']} (det: {s['n_detectadas']})\n"
            f"  span_x_rel GT   = {s['span_x_rel_gt']:.3f}\n"
            f"  span_x_rel pred = {s['span_x_rel_pred']:.3f}\n"
            f"  razón pred/GT   = {s['razon_pred_gt']:.3f}   <- 1.0 sano, <0.6 colapso\n"
            f"  colapsadas      = {s['colapsadas']}/{s['n_evaluables']} imágenes con razón < 0.6\n"
            f"  error kpt medio = {s['error_kpt']:.3f} (fracción de la diagonal GT)"
        )
    if args.json_out:
        import json  # noqa: PLC0415

        Path(args.json_out).write_text(json.dumps(reports, indent=2), encoding="utf-8")


# Umbral de colapso de la razón pred/GT (calibrado con v2: roto en la app, 0,70 medio).
_COLLAPSE_THR = 0.6


def _gt_set_images(root: Path) -> list[Path]:
    return sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in {".jpg", ".png"})


def gate_gt_set(
    model_path: Path,
    root: Path,
    *,
    label: str | None = None,
    save_vis: Path | None = None,
    max_vis: int = 3,
) -> dict[str, Any]:
    """Corre la compuerta OOD de un modelo sobre un set ``images/`` + ``labels/``.

    Ver :func:`cmd_gt_set` para la definición de las métricas.

    Returns:
        ``{"modelo", "ruta", "resumen": {...}, "por_imagen": [...]}``. Una imagen entra
        al resumen sólo si su GT tiene ≥ 4 keypoints, el modelo detecta a alguien y la
        detección tiene ≥ 4 keypoints con conf > 0,5 (``evaluable``). El error de
        keypoints además exige ≥ 4 keypoints presentes en ambos.
    """
    label = label or model_path.stem
    model = load_model(model_path)
    rows: list[dict[str, Any]] = []
    vis_done = 0
    for img_path in _gt_set_images(root):
        row: dict[str, Any] = {"imagen": img_path.stem, "evaluable": False}
        rows.append(row)
        img = cv2.imread(str(img_path))
        lab = root / "labels" / f"{img_path.stem}.txt"
        if img is None or not lab.exists():
            row["motivo"] = "sin imagen o label"
            continue
        first = next((ln for ln in lab.read_text().splitlines() if ln.strip()), None)
        if first is None:
            row["motivo"] = "label vacío"
            continue
        vals = np.array(first.split(), dtype=np.float64)
        if vals.size < 5 + 17 * 3:
            row["motivo"] = "label sin keypoints"
            continue
        hh, ww = img.shape[:2]
        gt = vals[5 : 5 + 17 * 3].reshape(17, 3).copy()
        gt[:, 0] *= ww
        gt[:, 1] *= hh
        w_gt, h_gt = vals[3] * ww, vals[4] * hh
        mask = gt[:, 2] > 0
        if int(mask.sum()) < 4:
            row["motivo"] = "GT con < 4 keypoints"
            continue
        row["span_x_rel_gt"] = float(gt[mask, 0].max() - gt[mask, 0].min()) / max(w_gt, 1e-9)

        det = model.infer_crop(img)  # imagen completa; el modelo elige su detección
        row["detectada"] = det is not None
        if det is None:
            row["motivo"] = "sin detección"
            continue
        row["score"] = float(det["score"])
        if "kx" in det:  # tflite: normalizado al tensor letterboxeado -> a px
            scale = max(ww, hh)
            pad_x, pad_y = (scale - ww) / 2, (scale - hh) / 2
            kx = det["kx"] * scale - pad_x
            ky = det["ky"] * scale - pad_y
            kc = det["kc"]
        else:  # .pt: Ultralytics ya devuelve px de la imagen original
            kx, ky, kc = det["kpts_px"][:, 0], det["kpts_px"][:, 1], det["kpts_px"][:, 2]
        pmask = kc > _KPT_CONF_THR
        row["kpts_confiables"] = int(pmask.sum())
        if int(pmask.sum()) < 4:
            row["motivo"] = "< 4 keypoints con conf > 0,5"
            continue
        row["evaluable"] = True
        row["span_x_rel_pred"] = float(kx[pmask].max() - kx[pmask].min()) / max(w_gt, 1e-9)
        row["razon_pred_gt"] = row["span_x_rel_pred"] / max(row["span_x_rel_gt"], 1e-9)
        row["colapsada"] = bool(row["razon_pred_gt"] < _COLLAPSE_THR)
        both = mask & pmask
        if int(both.sum()) >= 4:
            dist = np.hypot(kx[both] - gt[both, 0], ky[both] - gt[both, 1])
            row["error_kpt"] = float(np.mean(dist)) / max(np.hypot(w_gt, h_gt), 1e-9)
        if save_vis is not None and vis_done < max_vis:
            _save_gt_overlay(img, kx, ky, kc, gt, save_vis / f"gt_{label}_{img_path.stem}.jpg")
            vis_done += 1

    def m(key: str, rows_: list[dict[str, Any]]) -> float:
        vals = [r[key] for r in rows_ if key in r]
        return float(np.mean(vals)) if vals else float("nan")

    ev = [r for r in rows if r["evaluable"]]
    n_gt = sum(1 for r in rows if "span_x_rel_gt" in r)
    # El techo GT se promedia sobre todas las imágenes con GT válido (como antes del
    # refactor), no sólo sobre las evaluables.
    resumen = {
        "n_imagenes": len(rows),
        "n_gt_validas": n_gt,
        "n_detectadas": sum(1 for r in rows if r.get("detectada")),
        "n_evaluables": len(ev),
        # Imágenes con GT válido donde el modelo no produjo un esqueleto utilizable (sin
        # detección o < 4 keypoints con conf > 0,5): no entran a la razón ni al error,
        # pero son fallas, así que se reportan aparte.
        "n_sin_esqueleto": n_gt - len(ev),
        "span_x_rel_gt": m("span_x_rel_gt", rows),
        "span_x_rel_pred": m("span_x_rel_pred", ev),
        "razon_pred_gt": m("razon_pred_gt", ev),
        "razon_pred_gt_mediana": float(np.median([r["razon_pred_gt"] for r in ev]))
        if ev
        else float("nan"),
        "colapsadas": sum(1 for r in ev if r["colapsada"]),
        "error_kpt": m("error_kpt", ev),
    }
    return {"modelo": label, "ruta": str(model_path), "resumen": resumen, "por_imagen": rows}


def _save_gt_overlay(
    img: np.ndarray, kx: np.ndarray, ky: np.ndarray, kc: np.ndarray, gt: np.ndarray, out: Path
) -> None:
    """Dibuja predicción (verde/rojo) y GT (azul) sobre la imagen original."""
    from src.inference.keypoint_schema import COCO_SKELETON  # noqa: PLC0415

    canvas = img.copy()
    for a, b in COCO_SKELETON:
        if gt[a, 2] > 0 and gt[b, 2] > 0:
            p1 = (int(gt[a, 0]), int(gt[a, 1]))
            p2 = (int(gt[b, 0]), int(gt[b, 1]))
            cv2.line(canvas, p1, p2, (255, 160, 0), 2)
        if kc[a] > _KPT_CONF_THR and kc[b] > _KPT_CONF_THR:
            cv2.line(canvas, (int(kx[a]), int(ky[a])), (int(kx[b]), int(ky[b])), (0, 255, 0), 2)
    for j in range(17):
        if kc[j] > _KPT_CONF_THR:
            cv2.circle(canvas, (int(kx[j]), int(ky[j])), 4, (0, 0, 255), -1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)


def uco_frames_by_camera(data_yaml: Path, per_cam: int, seed: int = 0) -> dict[str, list[Path]]:
    """Muestra determinista de ``per_cam`` frames por cámara del split val del fold."""
    from src.data.uco_to_yolo import parse_stem  # noqa: PLC0415
    from src.evaluation.pose_eval import read_val_image_paths  # noqa: PLC0415

    by_cam: dict[str, list[Path]] = defaultdict(list)
    for path in read_val_image_paths(data_yaml):
        try:
            by_cam[parse_stem(path.stem).cam].append(path)
        except ValueError:
            continue
    rng = random.Random(seed)
    return {
        cam: rng.sample(sorted(paths), min(per_cam, len(paths)))
        for cam, paths in sorted(by_cam.items())
    }


def _read_first_label(image_path: Path) -> np.ndarray | None:
    """Primera línea del label YOLO-pose pseudo como array de 56 floats, o ``None``."""
    from src.evaluation.pose_eval import _label_path_for  # noqa: PLC0415

    label = _label_path_for(image_path)
    if not label.exists():
        return None
    first = next((ln for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()), None)
    if first is None:
        return None
    vals = np.array(first.split(), dtype=np.float64)
    return vals if vals.size >= 5 + 17 * 3 else None


def gt_ratios_from_label(image_path: Path) -> tuple[float, float] | None:
    """``(ratio_x, ratio_y)`` del GT pseudo de un frame, desde su label YOLO-pose."""
    vals = _read_first_label(image_path)
    if vals is None:
        return None
    kpts = vals[5 : 5 + 17 * 3].reshape(17, 3)
    mask = kpts[:, 2] > 0
    if int(mask.sum()) < 2:
        return None
    span_x = kpts[mask, 0].max() - kpts[mask, 0].min()
    span_y = kpts[mask, 1].max() - kpts[mask, 1].min()
    return float(span_x / max(vals[3], 1e-9)), float(span_y / max(vals[4], 1e-9))


def cmd_zoom(args: argparse.Namespace) -> None:
    """Colapso vs encuadre: mismo frame UCO, completo vs recortado alrededor de la persona.

    Recorta alrededor del bbox GT (margen 1.25) con aspecto forzado: 3:4 imita la
    cámara portrait del celular (barras laterales tras el letterbox), 16:9 imita un
    acercamiento que conserva barras horizontales como las del training. La persona
    pasa a ocupar ~0.5-0.8 del ancho del tensor, como en la app, con contenido 100 %
    in-domain (uco_pseudo) — aísla escala/encuadre de apariencia.
    """
    per_cam = max(1, args.zoom // 5)
    frames = [
        p
        for paths in uco_frames_by_camera(Path(args.data), per_cam=per_cam).values()
        for p in paths
    ]
    if args.model:
        models = [(Path(args.model).stem, Path(args.model))]
    else:
        models = [_DEFAULT_MODELS[0], _DEFAULT_MODELS[3]]  # desplegado + control stock
    aspects = [("portrait_3x4", 3 / 4), ("landscape_16x9", 16 / 9)]
    save_vis = Path(args.save_vis) if args.save_vis else None

    print(f"\n=== Zoom-crop ({len(frames)} frames UCO; encuadre tipo app, contenido in-domain) ===")
    for label, path in models:
        model = TfliteModel(path)
        rows: dict[str, list[dict[str, Any]]] = {"full_frame": []}
        for name, _ in aspects:
            rows[name] = []
        for i, frame in enumerate(frames):
            img = cv2.imread(str(frame))
            vals = _read_first_label(frame)
            if img is None or vals is None:
                continue
            hh, ww = img.shape[:2]
            det = run_tflite_frame(model, frame, None)
            if det is not None:
                rows["full_frame"].append(det)
            for name, aspect in aspects:
                cw, ch = vals[3] * ww * 1.25, vals[4] * hh * 1.25
                if cw / ch < aspect:
                    cw = ch * aspect
                else:
                    ch = cw / aspect
                x1 = max(0, round(vals[1] * ww - cw / 2))
                y1 = max(0, round(vals[2] * hh - ch / 2))
                x2, y2 = min(ww, round(x1 + cw)), min(hh, round(y1 + ch))
                crop = img[y1:y2, x1:x2]
                if crop.shape[0] < 64 or crop.shape[1] < 64:
                    continue
                tensor, _ = letterbox_app(crop)
                cdet = decode_best(model(tensor))
                if cdet is None:
                    continue
                rx, ry = ratios(cdet["kx"], cdet["ky"], cdet["kc"], cdet["box"][2], cdet["box"][3])
                cdet["ratio_x"], cdet["ratio_y"] = rx, ry
                rows[name].append(cdet)
                if save_vis is not None and i < 3:
                    _save_overlay(tensor, cdet, save_vis / f"zoom_{label}_{name}_{frame.stem}.jpg")

        print(f"\n[{label}]")
        print(f"{'encuadre':<16} {'n':>3} {'box_w':>6} {'score':>6} {'ratio_x':>8} {'ratio_y':>8}")
        for name, dets in rows.items():
            if not dets:
                print(f"{name:<16} {0:>3}")
                continue
            bw = float(np.mean([d["box"][2] for d in dets]))
            sc = float(np.mean([d["score"] for d in dets]))
            rx = float(np.mean([d["ratio_x"] for d in dets]))
            ry = float(np.mean([d["ratio_y"] for d in dets]))
            print(f"{name:<16} {len(dets):>3} {bw:>6.3f} {sc:>6.2f} {rx:>8.3f} {ry:>8.3f}")


def cmd_crop_image(args: argparse.Namespace) -> None:
    """Comparación por persona sobre una imagen OOD: crops individuales, modelo vs control.

    Detecta personas con el base COCO (``yolov8n-pose.pt``), recorta cada una en
    portrait 3:4 (margen 1.25) y corre cada modelo sobre el MISMO recorte: elimina
    la ambigüedad de "cada modelo eligió otra persona" de la matriz. Es el gate del
    fix: ``ratio_x`` acá es lo que se ve en la app.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    image_path = Path(args.crop_image)
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"No se pudo leer {image_path}")
        return
    hh, ww = img.shape[:2]
    base = YOLO(str(_BASE_MODEL), task="pose")
    result = base.predict(source=str(image_path), imgsz=IMGSZ, conf=0.5, verbose=False)[0]
    boxes = result.boxes.xywh.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))

    if args.model:
        paths = [(Path(args.model).stem, Path(args.model))]
    else:
        paths = [_DEFAULT_MODELS[0], _DEFAULT_MODELS[3]]  # desplegado + control stock
    models = [(label, load_model(path)) for label, path in paths]
    save_vis = Path(args.save_vis) if args.save_vis else None

    print(f"\n=== Por persona en {image_path.name} ({len(boxes)} detecciones del base COCO) ===")
    for i, (cx, cy, w, h) in enumerate(boxes):
        cw, ch = w * 1.25, h * 1.25
        if cw / ch < 3 / 4:
            cw = ch * 3 / 4
        else:
            ch = cw / (3 / 4)
        x1, y1 = max(0, round(cx - cw / 2)), max(0, round(cy - ch / 2))
        x2, y2 = min(ww, round(x1 + cw)), min(hh, round(y1 + ch))
        crop = img[y1:y2, x1:x2]
        if crop.shape[0] < 64 or crop.shape[1] < 64:
            continue
        line = f"persona {i} (w={w:.0f}px h={h:.0f}px):"
        for label, model in models:
            det = model.infer_crop(crop)
            if det is None:
                line += f"  {label}: SIN DET"
                continue
            rx, ry, box_w, score = det["ratio_x"], det["ratio_y"], det["box"][2], det["score"]
            line += f"  {label}: rx={rx:.3f} ry={ry:.3f} (box_w={box_w:.2f} s={score:.2f})"
            if save_vis is not None and "tensor" in det:
                _save_overlay(
                    det["tensor"], det, save_vis / f"crop_{image_path.stem}_p{i}_{label}.jpg"
                )
        print(line)


def _fmt_row(label: str, frame: str, det: dict[str, Any] | None) -> str:
    if det is None:
        return f"{label:<24} {frame:<28} {'SIN DETECCION':>13}"
    return (
        f"{label:<24} {frame:<28} score={det['score']:.2f} {det['level']:<3} "
        f"box_w={det['box'][2]:.3f} ratio_x={det['ratio_x']:.3f} ratio_y={det['ratio_y']:.3f}"
    )


def cmd_matrix(args: argparse.Namespace) -> None:
    """Matriz modelo x frame con decode crudo (tflite) o Ultralytics (.pt)."""
    models = _DEFAULT_MODELS if args.model is None else [(Path(args.model).stem, Path(args.model))]
    frames: list[Path] = [bus_jpg()]
    if args.app_frames:
        frames += sorted(
            p
            for p in Path(args.app_frames).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    for cam, paths in uco_frames_by_camera(Path(args.data), per_cam=2).items():
        frames += paths

    save_vis = Path(args.save_vis) if args.save_vis else None
    print(f"\n=== Matriz de diagnóstico (kpt_conf>{_KPT_CONF_THR}; sano: ratio_x >= 0.75) ===")
    for label, path in models:
        if not path.exists():
            print(f"{label:<24} NO EXISTE: {path}")
            continue
        if path.suffix == ".tflite":
            model = TfliteModel(path)
            print(f"\n[{label}] contrato: {model.contract()}")
            for frame in frames:
                print("  " + _fmt_row(label, frame.stem, run_tflite_frame(model, frame, save_vis)))
        else:
            pt = PtModel(path)
            print(f"\n[{label}] camino PyTorch/Ultralytics ({path.name})")
            for frame in frames:
                print("  " + _fmt_row(label, frame.stem, pt.run_frame(frame)))

    print(
        "\nInterpretación:\n"
        "  - .pt+_float32+_int8 colapsan en bus.jpg, ninguno en UCO  -> dominio (retrain)\n"
        "  - _int8 colapsa, _float32/.pt no                          -> cuantización\n"
        "  - nada colapsa en desktop                                 -> runtime Android\n"
        "  - el control stock también colapsa                        -> bug del script"
    )


def cmd_per_camera(args: argparse.Namespace) -> None:
    """ratio_x GT vs predicho por cámara UCO (¿las vistas laterales enseñan x estrecho?)."""
    model_path = Path(args.model) if args.model else _DEFAULT_MODELS[0][1]
    model = TfliteModel(model_path) if model_path.suffix == ".tflite" else None
    pt = PtModel(model_path) if model is None else None
    print(f"\n=== Por cámara (n={args.per_camera}/cam, modelo={model_path.name}) ===")
    print(
        f"{'cam':<6} {'n':>3} {'gt_rx':>7} {'gt_ry':>7} {'pred_rx':>8} {'pred_ry':>8} {'score':>6}"
    )
    for cam, paths in uco_frames_by_camera(Path(args.data), per_cam=args.per_camera).items():
        gt_rx, gt_ry, pr_rx, pr_ry, scores = [], [], [], [], []
        for p in paths:
            gt = gt_ratios_from_label(p)
            if gt is not None:
                gt_rx.append(gt[0])
                gt_ry.append(gt[1])
            det = run_tflite_frame(model, p, None) if model is not None else pt.run_frame(p)  # type: ignore[union-attr]
            if det is not None:
                pr_rx.append(det["ratio_x"])
                pr_ry.append(det["ratio_y"])
                scores.append(det["score"])

        def m(vals: list[float]) -> str:
            return f"{np.mean(vals):.3f}" if vals else "  -  "

        row = f"{cam:<6} {len(paths):>3} {m(gt_rx):>7} {m(gt_ry):>7}"
        print(f"{row} {m(pr_rx):>8} {m(pr_ry):>8} {m(scores):>6}")


def cmd_oks_sample(args: argparse.Namespace) -> None:
    """OKS/PCK sobre una muestra estratificada por cámara, o sobre todo el val (--oks-full).

    ``--oks-full`` puntúa **todo** el split val de ``--data``: es la forma de
    evaluar un checkpoint sobre otro dataset YAML sin reentrenar (p.ej. el subset
    verificado por humano ``pose_dataset_pseudo_corrected_gold_fold0.yaml``).
    """
    from src.evaluation import pose_eval  # noqa: PLC0415

    model_path = Path(args.model) if args.model else _DEFAULT_MODELS[0][1]
    if args.oks_full:
        sample = pose_eval.read_val_image_paths(Path(args.data))
        titulo = "OKS sobre el val completo"
    else:
        per_cam = max(1, args.oks_sample // 5)
        sample = []
        for paths in uco_frames_by_camera(Path(args.data), per_cam=per_cam).values():
            sample += paths
        titulo = "OKS muestreado"
    print(
        f"\n=== {titulo}: {len(sample)} frames, modelo={model_path.name}, "
        f"data={Path(args.data).name} ==="
    )

    gt_norm = pose_eval.load_gt_normalized(sample)
    base_preds = pose_eval.run_predictions(str(_BASE_MODEL), list(gt_norm), IMGSZ)
    gt_kpts, person_boxes, dims, kept_preds, kept = {}, {}, {}, {}, []
    for path, gt in gt_norm.items():
        pred = base_preds.get(path)
        if pred is None or pred["box"] is None:
            continue
        w, h = pred["wh"]
        gt_px = gt.copy()
        gt_px[:, 0] *= w
        gt_px[:, 1] *= h
        gt_kpts[path] = gt_px
        person_boxes[path] = np.asarray(pred["box"], dtype=np.float64)
        dims[path] = (w, h)
        kept_preds[path] = pred
        kept.append(path)
    ctx = pose_eval.EvalContext(
        gt_kpts=gt_kpts, person_boxes=person_boxes, dims=dims, img_paths=kept, base_preds=kept_preds
    )
    preds = pose_eval.run_predictions(str(model_path), kept, IMGSZ)
    score = pose_eval.score_stage(preds, ctx)
    print(
        f"oks={score.oks:.4f} pck={score.pck:.4f} "
        f"pck_leg={score.pck_leg if score.pck_leg is None else round(score.pck_leg, 4)} "
        f"pck_arm={score.pck_arm if score.pck_arm is None else round(score.pck_arm, 4)} "
        f"n_frames={score.n_frames}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", help="Evaluar sólo este modelo (.tflite o .pt); default: matriz completa."
    )
    parser.add_argument(
        "--data", default=str(_DEFAULT_DATA), help="Dataset YAML del fold (val = uco_pseudo)."
    )
    parser.add_argument("--app-frames", help="Directorio con frames reales de la app (opcional).")
    parser.add_argument(
        "--per-camera", type=int, default=0, help="N frames/cámara: GT vs predicho por cámara."
    )
    parser.add_argument(
        "--oks-sample", type=int, default=0, help="OKS/PCK sobre una muestra de N frames."
    )
    parser.add_argument(
        "--oks-full",
        action="store_true",
        help="OKS/PCK sobre TODO el split val de --data (sin muestreo).",
    )
    parser.add_argument("--save-vis", help="Directorio donde guardar overlays (evidencia visual).")
    parser.add_argument(
        "--zoom", type=int, default=0, help="N frames UCO: completo vs recorte tipo app."
    )
    parser.add_argument(
        "--crop-image", help="Imagen OOD: comparación por persona sobre crops individuales."
    )
    parser.add_argument(
        "--gt-set", help="Dir con images/ + labels/ (YOLO-pose): GATE contra ground truth."
    )
    parser.add_argument(
        "--json-out", help="Con --gt-set: guarda resumen y detalle por imagen en este JSON."
    )
    args = parser.parse_args()

    if args.gt_set:
        cmd_gt_set(args)
        return

    if args.oks_sample or args.oks_full:
        cmd_oks_sample(args)
        return
    if args.zoom:
        cmd_zoom(args)
        return
    if args.crop_image:
        cmd_crop_image(args)
        return
    cmd_matrix(args)
    if args.per_camera:
        cmd_per_camera(args)


if __name__ == "__main__":
    main()
