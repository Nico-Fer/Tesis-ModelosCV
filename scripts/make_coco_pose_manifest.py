"""Manifest de COCO-pose val2017 para mezclarlo en el fine-tune de recuperación.

Escanea ``data/processed/coco_pose/labels/val2017/*.txt`` (labels YOLO-pose de
Ultralytics, ~2.3k imágenes con personas anotadas), valida que exista el JPEG
par en ``images/val2017/`` y escribe un manifest de rutas ABSOLUTAS —el mismo
formato que los ``folds/fold{k}_train.txt`` de uco_pseudo— listo para usarse
como segunda fuente de ``train:`` en un dataset YAML de Ultralytics.

``--repeat K`` repite cada línea K veces (oversampling): con K=8, ~18.8k líneas
COCO ≈ 24 % del mix contra las 57.8k de uco_pseudo fold-0. Ultralytics conserva
duplicados en la lista de imágenes, así que el ratio del mix es literal.

Uso (cualquier venv, sólo stdlib):
    python scripts/make_coco_pose_manifest.py --repeat 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_COCO_ROOT = REPO_ROOT / "data" / "processed" / "coco_pose"
_IMG_EXTS = (".jpg", ".jpeg", ".png")


def build_manifest(coco_root: Path, repeat: int) -> tuple[Path, int, int]:
    """Escribe el manifest y devuelve ``(ruta, imágenes únicas, líneas totales)``."""
    labels_dir = coco_root / "labels" / "val2017"
    images_dir = coco_root / "images" / "val2017"
    if not labels_dir.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError(
            f"Falta {labels_dir} o {images_dir}. Extraer coco2017labels-pose.zip "
            "(labels) y val2017.zip (imágenes) bajo data/processed/coco_pose/."
        )

    images: list[Path] = []
    sin_par = 0
    for label in sorted(labels_dir.glob("*.txt")):
        if not label.read_text(encoding="utf-8").strip():
            continue  # label vacío: imagen sin personas anotadas, no aporta al kpt-loss
        for ext in _IMG_EXTS:
            img = images_dir / (label.stem + ext)
            if img.exists():
                images.append(img.resolve())
                break
        else:
            sin_par += 1

    if sin_par:
        print(f"[WARN] {sin_par} labels sin imagen par (val2017.zip incompleto?)")
    if not images:
        raise RuntimeError(f"Ningún par imagen/label válido bajo {coco_root}.")

    out = coco_root / "coco_val2017_pose.txt"
    lines = [str(p) for p in images for _ in range(repeat)]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out, len(images), len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=_COCO_ROOT)
    parser.add_argument(
        "--repeat", type=int, default=8, help="Veces que se repite cada imagen (oversampling)."
    )
    args = parser.parse_args()
    out, unique, total = build_manifest(args.coco_root, max(1, args.repeat))
    print(f"Manifest: {out}\n{unique} imágenes únicas x{args.repeat} = {total} líneas")


if __name__ == "__main__":
    main()
