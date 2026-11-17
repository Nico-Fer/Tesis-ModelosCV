"""Aplica el gold set humano sobre los pseudo-labels: dataset corregido + folds.

Cierra el loop de la auditoría: las correcciones hechas en CVAT (reconciliadas
en ``reconcile.gold_root`` por :mod:`src.audit.reconcile_corrections`) vuelven
al entrenamiento. Produce ``aplicar.out_root`` con el mismo layout que
``uco_pseudo`` — y por lo tanto consumible por los runners cambiando solo
``data:`` —:

* ``images/all/`` — hardlinks a las 74.107 imágenes del pool pseudo;
* ``labels/all/`` — copia de los pseudo-labels donde los frames del gold **con
  correcciones** se reemplazan por su versión humana; los auditados sin
  correcciones conservan el pseudo byte a byte (ver :func:`gold_reemplaza`).
  ``uco_pseudo`` nunca se edita en el lugar: es el baseline de la comparación y
  tiene su propio ``.cache``;
* ``provenance.csv`` — ``stem, fuente ∈ {pseudo, humano} (¿lo verificó un
  humano?), label_de ∈ {pseudo, gold} (¿de dónde salen los bytes?), motivo``;
* ``folds/fold{k}_{train,val}.txt`` + ``<out_yaml>_fold{k}.yaml``;
* ``folds/fold{k}_val_gold.txt`` + ``<out_yaml>_gold_fold{k}.yaml`` — el subset
  verificado por humano del val de cada fold (sesgado a frames difíciles: el
  estrato ``control`` es la única muestra aleatoria);
* ``<out_yaml>_coco_fold{k}.yaml`` — el mix v3 (uco + COCO-pose) sobre el
  dataset corregido, si ``aplicar.mix_coco`` está configurado.

**Folds.** El reparto sujeto → fold se lee de los manifests de ``uco_pseudo``
(no se re-sortea con la seed: así "mismo reparto" es un hecho, no una
suposición). La membresía sigue la regla ``skip_empty`` del pseudo-labeling: un
frame entra a los manifests solo si su label tiene al menos una persona. Por eso
la corrección puede **sumar** frames (los ``sin_deteccion`` anotados de cero,
que el pseudo dejaba afuera) y **quitar** frames (el anotador borró todo).
Cualquier otro cambio de membresía respecto de ``uco_pseudo`` aborta.

Con ``aplicar.usar_en_train: false`` los stems del gold se excluyen de todos
los manifests de train (held-out por frame); con ``true`` entran al train de
los folds donde su sujeto es train y el held-out es por sujeto.

Uso::

    python -m src.audit.apply_corrections --config configs/audit_cvat.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.audit.common import AuditConfig, build_config, load_config
from src.data.pseudo_label_vitpose import link_image
from src.data.uco_to_yolo import COCO_FLIP_IDX, parse_stem, write_dataset_yaml
from src.inference.keypoint_schema import NUM_COCO_KEYPOINTS

logger = logging.getLogger(__name__)

PROVENANCE_FIELDS = ["stem", "fuente", "label_de", "motivo"]


def subject_folds(folds_dir: Path, n_folds: int) -> dict[str, int]:
    """Reparto sujeto → fold según los ``fold{k}_val.txt`` existentes.

    Raises:
        FileNotFoundError: si falta el manifest de val de algún fold.
        ValueError: si un sujeto aparece en el val de dos folds.
    """
    out: dict[str, int] = {}
    for k in range(n_folds):
        manifest = folds_dir / f"fold{k}_val.txt"
        if not manifest.exists():
            raise FileNotFoundError(f"Falta {manifest}: no se puede heredar el reparto de folds.")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            subject = parse_stem(Path(line.strip()).stem).subject
            if out.setdefault(subject, k) != k:
                raise ValueError(f"Sujeto {subject} en el val de los folds {out[subject]} y {k}.")
    return out


def _stems_de_manifest(path: Path) -> set[str]:
    lineas = path.read_text(encoding="utf-8").splitlines()
    return {Path(line.strip()).stem for line in lineas if line.strip()}


def _n_personas(texto: str) -> int:
    return sum(1 for line in texto.splitlines() if line.strip())


def read_gold(gold_root: Path) -> dict[str, dict[str, str]]:
    """Filas de ``gold_manifest.csv`` por stem.

    Raises:
        FileNotFoundError: si no se corrió la reconciliación o falta un label.
    """
    manifest = gold_root / "gold_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"No existe {manifest}; correr antes src.audit.reconcile_corrections."
        )
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        gold = {row["stem"]: row for row in csv.DictReader(fh)}
    for stem in gold:
        if not (gold_root / "labels" / "all" / f"{stem}.txt").exists():
            raise FileNotFoundError(f"El gold lista {stem} pero no tiene label en {gold_root}.")
    return gold


def gold_reemplaza(fila_gold: dict[str, str], pseudo_texto: str) -> bool:
    """¿El label humano reemplaza al pseudo, o el humano confirmó el pseudo?

    Un frame auditado **sin correcciones** (ningún keypoint movido más allá del
    umbral, agregado ni eliminado, y la misma cantidad de personas) conserva el
    pseudo-label byte a byte. Reemplazarlo cambiaría igual los bytes: el gold
    reconstruye el bbox como hull de keypoints + 5 % (CVAT no preserva el del
    detector), y eso mezclaría convenciones de bbox en frames que el humano
    confirmó como correctos.
    """
    return int(fila_gold["n_kpts_corregidos"]) > 0 or int(
        fila_gold["n_personas_gold"]
    ) != _n_personas(pseudo_texto)


def _guardar_destino(cfg: AuditConfig) -> None:
    """El destino se borra entero: nunca puede ser el pool pseudo ni el gold."""
    destino = cfg.corrected_root.resolve()
    protegidos = {
        cfg.images_dir.resolve().parent.parent,
        cfg.labels_dir.resolve().parent.parent,
        cfg.gold_root.resolve(),
    }
    if destino in protegidos or any(p.is_relative_to(destino) for p in protegidos):
        raise ValueError(
            f"aplicar.out_root={cfg.corrected_root} contiene o es un dataset de origen; "
            "se borraría al regenerar."
        )


def _escribir_yaml_mix(out_yaml: Path, root: Path, fold: int, coco_manifest: Path) -> Path:
    """Dataset YAML del mix uco-corregido + COCO (misma forma que el mix v3)."""
    import yaml  # noqa: PLC0415

    spec = {
        "path": str(root.parent),
        "train": [str(root / "folds" / f"fold{fold}_train.txt"), str(coco_manifest.resolve())],
        "val": str(root / "folds" / f"fold{fold}_val.txt"),
        "kpt_shape": [NUM_COCO_KEYPOINTS, 3],
        "flip_idx": COCO_FLIP_IDX,
        "names": {0: "person"},
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Mix v3 sobre el dataset CORREGIDO por humano (src.audit.apply_corrections).\n"
            "# El val es 100 % uco corregido del fold, comparable con el mix pseudo.\n"
        )
        yaml.safe_dump(spec, fh, sort_keys=False, allow_unicode=True)
    return out_yaml


def apply(cfg: AuditConfig, workers: int = 16) -> dict[str, Any]:
    """Genera el dataset corregido completo.

    Args:
        cfg: configuración de la auditoría.
        workers: hilos para materializar el pool (hardlinks + labels).

    Returns:
        El resumen escrito en ``<out_root>/apply_summary.json``.

    Raises:
        FileNotFoundError: gold, folds pseudo o manifest COCO faltantes.
        ValueError: membresía de folds que cambia más allá de lo explicable
            por el gold, o un destino que pisaría un dataset de origen.
    """
    _guardar_destino(cfg)
    gold = read_gold(cfg.gold_root)
    folds_de = subject_folds(cfg.folds_dir, cfg.n_folds)
    if cfg.coco_manifest is not None and not cfg.coco_manifest.exists():
        raise FileNotFoundError(
            f"No existe {cfg.coco_manifest}; generarlo con scripts/make_coco_pose_manifest.py "
            "o quitar aplicar.mix_coco."
        )

    root = cfg.corrected_root.resolve()
    if root.exists():
        shutil.rmtree(root)
    img_dir = root / "images" / "all"
    lbl_dir = root / "labels" / "all"
    folds_dir = root / "folds"
    for d in (img_dir, lbl_dir, folds_dir):
        d.mkdir(parents=True)

    stems = sorted(p.stem for p in cfg.images_dir.glob("*.jpg"))
    if not stems:
        raise FileNotFoundError(f"No hay imágenes en {cfg.images_dir}.")
    ajenos = sorted(set(gold) - set(stems))
    if ajenos:
        raise ValueError(f"{len(ajenos)} stems del gold no están en el pool (p.ej. {ajenos[0]}).")
    logger.info(
        "Pool: %d frames; gold: %d frames. Materializando %s ...", len(stems), len(gold), root
    )

    def _materializar(stem: str) -> tuple[bool, bool]:
        """Escribe label + imagen de un frame; devuelve ``(con_personas, reemplazado)``."""
        pseudo_lbl = cfg.labels_dir / f"{stem}.txt"
        if not pseudo_lbl.exists():
            raise FileNotFoundError(f"Falta el pseudo-label de {stem}: {pseudo_lbl}")
        texto = pseudo_lbl.read_text(encoding="utf-8")
        reemplazado = stem in gold and gold_reemplaza(gold[stem], texto)
        fuente = cfg.gold_root / "labels" / "all" / f"{stem}.txt" if reemplazado else pseudo_lbl
        # copyfile, no write_text: en Windows write_text convierte \n → \r\n y
        # el label dejaría de ser idéntico al de origen.
        shutil.copyfile(fuente, lbl_dir / f"{stem}.txt")
        if reemplazado:
            texto = fuente.read_text(encoding="utf-8")
        link_image(cfg.images_dir / f"{stem}.jpg", img_dir / f"{stem}.jpg", cfg.link_mode)
        return _n_personas(texto) > 0, reemplazado

    # Hilos: domina la latencia de crear 148k archivos en NTFS, no la CPU
    # (secuencial tardaba ~11 min).
    with ThreadPoolExecutor(max_workers=workers) as pool:
        resultados = dict(zip(stems, pool.map(_materializar, stems)))

    con_personas = {s for s, (personas, _) in resultados.items() if personas}
    reemplazados = {s for s, (_, reemp) in resultados.items() if reemp}
    provenance = [
        {
            "stem": stem,
            "fuente": "humano" if stem in gold else "pseudo",
            "label_de": "gold" if stem in reemplazados else "pseudo",
            "motivo": gold[stem]["motivo"] if stem in gold else "",
        }
        for stem in stems
    ]

    with (root / "provenance.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROVENANCE_FIELDS)
        writer.writeheader()
        writer.writerows(provenance)

    # --- folds ---------------------------------------------------------------
    val: dict[int, list[str]] = {k: [] for k in range(cfg.n_folds)}
    for stem in sorted(con_personas):
        subject = parse_stem(stem).subject
        if subject not in folds_de:
            raise ValueError(f"El sujeto {subject} ({stem}) no está en ningún fold de uco_pseudo.")
        val[folds_de[subject]].append(stem)

    def _ruta(stem: str) -> str:
        return str(img_dir / f"{stem}.jpg")

    resumen_folds: dict[str, Any] = {}
    yamls: list[str] = []
    for k in range(cfg.n_folds):
        val_k = val[k]
        train_k = [
            s
            for j in range(cfg.n_folds)
            if j != k
            for s in val[j]
            if cfg.usar_en_train or s not in gold
        ]
        train_k.sort()
        (folds_dir / f"fold{k}_train.txt").write_text(
            "".join(f"{_ruta(s)}\n" for s in train_k), encoding="utf-8"
        )
        (folds_dir / f"fold{k}_val.txt").write_text(
            "".join(f"{_ruta(s)}\n" for s in val_k), encoding="utf-8"
        )
        _verificar_membresia(cfg, k, set(train_k), set(val_k), set(gold))
        fold_yaml = cfg.corrected_yaml.with_name(
            f"{cfg.corrected_yaml.stem}_fold{k}{cfg.corrected_yaml.suffix}"
        )
        write_dataset_yaml(fold_yaml, root, f"folds/fold{k}_train.txt", f"folds/fold{k}_val.txt")
        yamls.append(str(fold_yaml))

        gold_val = [s for s in val_k if s in gold]
        if gold_val:
            (folds_dir / f"fold{k}_val_gold.txt").write_text(
                "".join(f"{_ruta(s)}\n" for s in gold_val), encoding="utf-8"
            )
            gold_yaml = cfg.corrected_yaml.with_name(
                f"{cfg.corrected_yaml.stem}_gold_fold{k}{cfg.corrected_yaml.suffix}"
            )
            write_dataset_yaml(
                gold_yaml, root, f"folds/fold{k}_train.txt", f"folds/fold{k}_val_gold.txt"
            )
            yamls.append(str(gold_yaml))
        resumen_folds[f"fold{k}"] = {
            "train": len(train_k),
            "val": len(val_k),
            "val_gold": len(gold_val),
        }

    if cfg.coco_manifest is not None:
        mix_yaml = cfg.corrected_yaml.with_name(
            f"{cfg.corrected_yaml.stem}_coco_fold{cfg.coco_fold}{cfg.corrected_yaml.suffix}"
        )
        yamls.append(str(_escribir_yaml_mix(mix_yaml, root, cfg.coco_fold, cfg.coco_manifest)))

    summary = {
        "generado": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "usar_en_train": cfg.usar_en_train,
        "n_pool": len(stems),
        "n_con_personas": len(con_personas),
        "n_gold": len(gold),
        "n_gold_reemplazados": len(reemplazados),
        "n_gold_confirmados": len(gold) - len(reemplazados),
        "definiciones": {
            "reemplazado": "label humano en lugar del pseudo (hubo corrección)",
            "confirmado": "auditado sin correcciones: se conserva el pseudo byte a byte",
        },
        "folds": resumen_folds,
        "yamls": yamls,
    }
    (root / "apply_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "Dataset corregido: %d frames con persona; gold %d (%d labels reemplazados, %d "
        "confirmados sin cambios); folds (train, val, val_gold) %s.",
        len(con_personas),
        len(gold),
        len(reemplazados),
        len(gold) - len(reemplazados),
        {k: (v["train"], v["val"], v["val_gold"]) for k, v in resumen_folds.items()},
    )
    for y in yamls:
        logger.info("  YAML: %s", y)
    return summary


def _verificar_membresia(
    cfg: AuditConfig, k: int, train: set[str], val: set[str], gold: set[str]
) -> None:
    """Solo el gold puede cambiar la membresía de un fold respecto de ``uco_pseudo``.

    Raises:
        ValueError: si train y val se solapan o cambia un frame que no es gold.
    """
    if train & val:
        raise ValueError(f"fold{k}: {len(train & val)} frames en train y val a la vez.")
    if not cfg.usar_en_train and train & gold:
        raise ValueError(f"fold{k}: usar_en_train=false pero hay stems del gold en train.")
    for split, mine in (("val", val), ("train", train)):
        pseudo = _stems_de_manifest(cfg.folds_dir / f"fold{k}_{split}.txt")
        cambios = (mine ^ pseudo) - gold
        if cambios:
            raise ValueError(
                f"fold{k}_{split}: {len(cambios)} frames cambian de membresía sin ser del gold "
                f"(p.ej. {sorted(cambios)[0]}): el reparto no es el de uco_pseudo."
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML de auditoría.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    apply(build_config(load_config(args.config)))


if __name__ == "__main__":
    main()
