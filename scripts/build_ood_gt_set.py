"""Arma los sets de la compuerta OOD (``diagnose_kpt_ratio.py --gt-set``) desde COCO.

La compuerta fuera de dominio exige un set con GT **real** de personas en condiciones
parecidas a las de la app (una persona grande, de frente, de cuerpo entero). El set
original de 79 imágenes (julio de 2026) se perdió. Este script lo reconstruye con
criterios explícitos y deja la lista de IDs **versionada** en ``data/ood_gate/``, así
el set es reproducible byte a byte.

Criterios (todos sobre las anotaciones oficiales ``person_keypoints_<split>.json``):

1. La imagen tiene **exactamente una** anotación de persona, y no es ``iscrowd``.
2. Los **17 keypoints** están anotados (``num_keypoints == 17``).
3. Los 12 keypoints del cuerpo (hombros a tobillos) están **visibles** (``v == 2``):
   cuerpo entero.
4. **Grande:** la altura del bbox es ≥ 50 % de la altura de la imagen.
5. **De pie:** el bbox es más alto que ancho.
6. **De frente:** el hombro y la cadera izquierdos de la persona aparecen a la derecha
   de los derechos en la imagen, y el ancho de hombros es ≥ 30 % del ancho del bbox
   (descarta perfiles).

Dos splits:

* ``train2017``: 578 candidatas; se toman 80 con semilla 0. Es el split del set
  original, y sobre él están calibrados los umbrales de la compuerta (v2 = 0,70).
  Nota: el YOLOv8n-pose de partida se pre-entrenó en COCO train2017.
* ``val2017``: las 37 candidatas. Ni el modelo stock ni los modelos UCO-only las
  vieron. El linaje v3 sí: su fine-tune mezcla COCO val2017 (ver ``AGENTS.md``).

Salida: ``<out>/<split>/images/*.jpg`` + ``labels/*.txt`` en formato YOLO-pose
(``0 cx cy w h`` + 17 × ``x y v``, normalizado). Las imágenes de train2017 se descargan
de ``images.cocodataset.org``; las de val2017 se copian de
``data/processed/coco_pose/images/val2017`` si están, y si no se descargan.

Uso (cualquier venv con ``requests`` o sólo stdlib)::

    python scripts/build_ood_gt_set.py --annotations-dir <dir con person_keypoints_*.json>

``--annotations-dir`` sólo hace falta la primera vez (para elegir los IDs) o para
regenerar los labels; con las listas de ``data/ood_gate/`` ya versionadas, el script
descarga las imágenes y reescribe los labels desde las anotaciones.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IDS_DIR = REPO / "data" / "ood_gate"
DEFAULT_OUT = REPO / "data" / "processed" / "ood_gate_coco"
LOCAL_VAL = REPO / "data" / "processed" / "coco_pose" / "images" / "val2017"
URL = "http://images.cocodataset.org/{split}/{id:012d}.jpg"

CRITERIA = {
    "personas_por_imagen": 1,
    "num_keypoints": 17,
    "cuerpo_visible_v2": "keypoints 5-16",
    "alto_bbox_min_frac_imagen": 0.5,
    "bbox_vertical": True,
    "frontal": "x(hombro_izq) > x(hombro_der) y x(cadera_izq) > x(cadera_der)",
    "ancho_hombros_min_frac_bbox": 0.3,
}
N_TRAIN, SEED = 80, 0


def _candidates(ann_path: Path) -> tuple[list[int], dict[int, dict], dict[int, dict]]:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in data["images"]}
    by_img: dict[int, list[dict]] = defaultdict(list)
    for a in data["annotations"]:
        by_img[a["image_id"]].append(a)
    out, ann_of = [], {}
    for iid, anns in by_img.items():
        if len(anns) != 1 or anns[0]["iscrowd"]:
            continue
        a = anns[0]
        k = a["keypoints"]
        xs, vs = k[0::3], k[2::3]
        if a["num_keypoints"] < 17 or not all(vs[i] == 2 for i in range(5, 17)):
            continue
        x, y, w, h = a["bbox"]
        if h < CRITERIA["alto_bbox_min_frac_imagen"] * imgs[iid]["height"] or h <= w:
            continue
        if not (xs[5] > xs[6] and xs[11] > xs[12]):
            continue
        if (xs[5] - xs[6]) < CRITERIA["ancho_hombros_min_frac_bbox"] * w:
            continue
        out.append(iid)
        ann_of[iid] = a
    return sorted(out), ann_of, imgs


def _label_line(a: dict, img: dict) -> str:
    w_img, h_img = img["width"], img["height"]
    x, y, w, h = a["bbox"]
    parts = [0, (x + w / 2) / w_img, (y + h / 2) / h_img, w / w_img, h / h_img]
    k = a["keypoints"]
    for i in range(17):
        kx, ky, v = k[3 * i], k[3 * i + 1], k[3 * i + 2]
        parts += [kx / w_img if v else 0.0, ky / h_img if v else 0.0, int(v)]
    return " ".join(f"{p:.6f}" if isinstance(p, float) else str(p) for p in parts)


def _fetch(split: str, iid: int, dst: Path) -> None:
    if dst.exists():
        return
    local = LOCAL_VAL / f"{iid:012d}.jpg"
    if split == "val2017" and local.exists():
        shutil.copy2(local, dst)
        return
    with urllib.request.urlopen(URL.format(split=split, id=iid), timeout=60) as resp:
        dst.write_bytes(resp.read())


def build(split: str, ann_dir: Path | None, out: Path) -> Path:
    ids_file = IDS_DIR / f"ids_{split}.txt"
    ann_path = ann_dir / f"person_keypoints_{split}.json" if ann_dir else None
    if ann_path is None or not ann_path.exists():
        raise SystemExit(f"Hace falta {ann_path or '--annotations-dir'} para escribir los labels.")
    cands, ann_of, imgs = _candidates(ann_path)
    if ids_file.exists():
        ids = [int(x) for x in ids_file.read_text(encoding="utf-8").split()]
        missing = [i for i in ids if i not in ann_of]
        if missing:
            raise SystemExit(f"{ids_file}: IDs que ya no cumplen los criterios: {missing[:5]}")
    else:
        ids = cands if split == "val2017" else sorted(random.Random(SEED).sample(cands, N_TRAIN))
        IDS_DIR.mkdir(parents=True, exist_ok=True)
        ids_file.write_text("\n".join(str(i) for i in ids) + "\n", encoding="utf-8")
        meta = {
            "split": split,
            "criterios": CRITERIA,
            "candidatas": len(cands),
            "elegidas": len(ids),
            "semilla": SEED if split == "train2017" else None,
        }
        (IDS_DIR / f"ids_{split}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    root = out / split
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for n, iid in enumerate(ids, 1):
        _fetch(split, iid, root / "images" / f"{iid:012d}.jpg")
        (root / "labels" / f"{iid:012d}.txt").write_text(
            _label_line(ann_of[iid], imgs[iid]) + "\n", encoding="utf-8"
        )
        if n % 20 == 0:
            print(f"  {split}: {n}/{len(ids)}", flush=True)
    print(f"{split}: {len(ids)} imágenes ({len(cands)} candidatas) -> {root}")
    return root


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--annotations-dir", type=Path, help="Dir con person_keypoints_{train,val}2017.json"
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--splits", nargs="+", default=["train2017", "val2017"])
    args = p.parse_args()
    for split in args.splits:
        build(split, args.annotations_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
