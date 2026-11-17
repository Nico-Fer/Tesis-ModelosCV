"""Empaqueta la selección de auditoría para una task de CVAT.

Produce, bajo ``pack.out_dir``:

* ``images/<subset>/`` — las imágenes seleccionadas (hardlinks), que
  ``scripts/cvat_audit.py montar`` sube a la task **en el orden de**
  ``orden.txt``;
* ``orden.txt`` — los stems en un orden **mezclado y determinista**
  (``seleccion.seed``): el anotador no debe poder distinguir los frames de
  control de los marcados, o la tasa de corrección de cada grupo quedaría
  sesgada por cuánta atención recibe cada uno;
* ``<out_dir>_anotaciones.zip`` — los pseudo-labels como **pre-anotaciones** en
  formato ``Ultralytics YOLO Pose 1.0`` (``data.yaml`` + ``<subset>.txt`` +
  ``labels/``), para importar a nivel task;
* ``cvat_skeleton_labels.json`` (junto al pack) — la definición del label
  ``person`` como SKELETON de 17 puntos COCO (con las aristas de
  ``COCO_SKELETON``); ``montar`` crea el proyecto desde ahí.

Dos usos::

    # pack completo de la selección
    python -m src.audit.pack_for_cvat --config configs/audit_cvat.yaml

    # pack chico para el gate de round-trip (frames con v=0/1/2 mezclados)
    python -m src.audit.pack_for_cvat --config configs/audit_cvat.yaml --limit 5

Con ``--limit`` el pack sale a ``<out_dir>_roundtrip`` para no pisar el pack
real, y se priorizan frames con las tres visibilidades presentes: un round-trip
sin ``v=1``/``v=0`` no probaría el mapeo occluded/outside, que es el punto
frágil que el gate existe para verificar.
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import zipfile
from pathlib import Path

from src.audit.common import AuditConfig, build_config, load_config, read_selection
from src.data.pseudo_label_vitpose import link_image
from src.inference.keypoint_schema import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    NUM_COCO_KEYPOINTS,
)

logger = logging.getLogger(__name__)

# Posiciones de plantilla (viewport 100×100 del editor de skeletons de CVAT)
# para dibujar el esqueleto COCO en pose anatómica. Solo afectan la plantilla
# de dibujo en la UI, nunca las anotaciones.
_TEMPLATE_XY: dict[str, tuple[float, float]] = {
    "nose": (50.0, 10.0),
    "left_eye": (53.0, 8.0),
    "right_eye": (47.0, 8.0),
    "left_ear": (57.0, 10.5),
    "right_ear": (43.0, 10.5),
    "left_shoulder": (61.0, 24.0),
    "right_shoulder": (39.0, 24.0),
    "left_elbow": (66.0, 40.0),
    "right_elbow": (34.0, 40.0),
    "left_wrist": (68.0, 54.0),
    "right_wrist": (32.0, 54.0),
    "left_hip": (57.0, 52.0),
    "right_hip": (43.0, 52.0),
    "left_knee": (57.0, 71.0),
    "right_knee": (43.0, 71.0),
    "left_ankle": (57.0, 90.0),
    "right_ankle": (43.0, 90.0),
}


def skeleton_label_spec(label_name: str) -> list[dict]:
    """Definición del label SKELETON para el editor *Raw* de labels de CVAT.

    Los ``data-node-id`` del SVG son 1-based y su orden DEBE coincidir con el
    orden COCO de los sublabels: el importer de CVAT mapea los puntos del
    ``.txt`` a los sublabels **por posición**.
    """
    svg_parts: list[str] = []
    for i, j in COCO_SKELETON:
        x1, y1 = _TEMPLATE_XY[COCO_KEYPOINT_NAMES[i]]
        x2, y2 = _TEMPLATE_XY[COCO_KEYPOINT_NAMES[j]]
        svg_parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'data-type="edge" data-node-from="{i + 1}" data-node-to="{j + 1}" '
            f'stroke="black" stroke-width="0.5"></line>'
        )
    for idx, name in enumerate(COCO_KEYPOINT_NAMES):
        x, y = _TEMPLATE_XY[name]
        svg_parts.append(
            f'<circle r="1.5" cx="{x}" cy="{y}" data-type="element node" '
            f'data-element-id="{idx + 1}" data-node-id="{idx + 1}" '
            f'data-label-name="{name}" stroke="black" fill="#b3b3b3" '
            f'stroke-width="0.1"></circle>'
        )
    return [
        {
            "name": label_name,
            "type": "skeleton",
            "attributes": [],
            "svg": "".join(svg_parts),
            "sublabels": [
                {"name": name, "type": "points", "attributes": []} for name in COCO_KEYPOINT_NAMES
            ],
        }
    ]


def _prioridad_roundtrip(fila: dict[str, str]) -> tuple:
    """Orden de preferencia para el pack de round-trip.

    Primero frames con las 3 visibilidades presentes y al menos una persona
    escrita; el desempate por stem mantiene la elección determinista.
    """
    con_labels = int(fila["n_personas_escritas"]) > 0
    diversidad = sum(int(fila[k]) > 0 for k in ("n_v2", "n_v1", "n_v0"))
    return (not con_labels, -diversidad, fila["stem"])


def orden_mezclado(stems: list[str], seed: int) -> list[str]:
    """Permutación determinista de los stems (independiente del orden de entrada)."""
    orden = sorted(stems)
    random.Random(seed).shuffle(orden)
    return orden


def roundtrip_dir(cfg: AuditConfig) -> Path:
    """Directorio del pack chico del gate (no pisa el pack real)."""
    return cfg.pack_dir.parent / f"{cfg.pack_dir.name}_roundtrip"


def anotaciones_zip(pack_dir: Path) -> Path:
    """ZIP de pre-anotaciones asociado a un directorio de pack."""
    return pack_dir.parent / f"{pack_dir.name}_anotaciones.zip"


def build_pack(cfg: AuditConfig, limit: int | None = None) -> Path:
    """Arma el directorio del pack y el ZIP de pre-anotaciones.

    Returns:
        Ruta del directorio del pack.

    Raises:
        FileNotFoundError: si falta una imagen o un label de la selección
            (fail-fast: un pack incompleto produciría una auditoría parcial
            indistinguible de una completa).
    """
    filas = read_selection(cfg.selection_csv)
    if limit is not None:
        filas = sorted(filas, key=_prioridad_roundtrip)[:limit]
        pack_dir = roundtrip_dir(cfg)
    else:
        pack_dir = cfg.pack_dir

    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    img_dir = pack_dir / "images" / cfg.subset
    lbl_dir = pack_dir / "labels" / cfg.subset
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    orden = orden_mezclado([f["stem"] for f in filas], cfg.seed)
    for stem in orden:
        src_img = cfg.images_dir / f"{stem}.jpg"
        src_lbl = cfg.labels_dir / f"{stem}.txt"
        if not src_img.exists():
            raise FileNotFoundError(f"Falta la imagen de la selección: {src_img}")
        if not src_lbl.exists():
            raise FileNotFoundError(f"Falta el label de la selección: {src_lbl}")
        link_image(src_img, img_dir / f"{stem}.jpg", cfg.link_mode)
        shutil.copyfile(src_lbl, lbl_dir / f"{stem}.txt")

    (pack_dir / "orden.txt").write_text("\n".join(orden) + "\n", encoding="utf-8")
    (pack_dir / f"{cfg.subset}.txt").write_text(
        "\n".join(f"images/{cfg.subset}/{stem}.jpg" for stem in orden) + "\n", encoding="utf-8"
    )
    (pack_dir / "data.yaml").write_text(
        "path: .\n"
        f"{cfg.subset}: {cfg.subset}.txt\n"
        f"kpt_shape: [{NUM_COCO_KEYPOINTS}, 3]\n"
        "names:\n"
        f"  0: {cfg.label_name}\n",
        encoding="utf-8",
    )

    import json  # noqa: PLC0415

    spec_path = pack_dir.parent / "cvat_skeleton_labels.json"
    spec_path.write_text(
        json.dumps(skeleton_label_spec(cfg.label_name), indent=2), encoding="utf-8"
    )

    # Solo anotaciones: las imágenes se suben aparte, en orden, al crear la task.
    zip_path = anotaciones_zip(pack_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pack_dir / "data.yaml", "data.yaml")
        zf.write(pack_dir / f"{cfg.subset}.txt", f"{cfg.subset}.txt")
        for stem in orden:
            zf.write(lbl_dir / f"{stem}.txt", f"labels/{cfg.subset}/{stem}.txt")

    n_vacios = sum(1 for f in filas if int(f["n_personas_escritas"]) == 0)
    logger.info(
        "Pack armado: %d frames (%d sin pre-anotación) → %s", len(filas), n_vacios, pack_dir
    )
    logger.info("Pre-anotaciones: %s", zip_path)
    logger.info("Definición del skeleton para CVAT: %s", spec_path)
    return pack_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML de auditoría.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Empaquetar solo N frames (pack de round-trip, no pisa el pack real).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = build_config(load_config(args.config))
    build_pack(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
