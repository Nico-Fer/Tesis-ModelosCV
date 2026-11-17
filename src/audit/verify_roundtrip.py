"""Gate de round-trip: la visibilidad debe sobrevivir CVAT ida y vuelta, intacta.

El punto frágil del flujo es el mapeo de visibilidad — YOLO usa ``v ∈ {0,1,2}``;
CVAT usa flags ``outside``/``occluded`` internos (``v=0 ⇒ outside``,
``v=1 ⇒ occluded``, ``v=2 ⇒ visible``). Antes de corregir miles de frames a
mano: importar el pack de ``--limit 5``, exportarlo de CVAT **sin tocar nada**
y correr este script. Si algo no vuelve idéntico, se para y se resuelve.

Además mide el **noise floor**: el desplazamiento máximo (px) introducido por
el viaje norm → px (CVAT) → norm en keypoints no tocados. Ese piso define la
tolerancia mínima razonable de ``reconcile.corregido_thr_px``.

Uso::

    python -m src.audit.verify_roundtrip --config configs/audit_cvat.yaml \\
        --export data/processed/uco_audit/cvat_export_roundtrip.zip
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import zipfile
from pathlib import Path

from src.audit.common import (
    AuditConfig,
    PoseLabel,
    build_config,
    image_size,
    load_config,
    match_persons,
    read_label_file,
)

logger = logging.getLogger(__name__)


def labels_from_export(export: Path, workdir: Path) -> dict[str, Path]:
    """Indexa por stem los ``.txt`` de labels de un export de CVAT (ZIP o dir).

    Busca bajo cualquier componente ``labels/`` para no depender del nombre de
    subset que CVAT haya usado al exportar.

    Raises:
        ValueError: si el export no contiene ningún label.
    """
    root = export
    if export.is_file():
        with zipfile.ZipFile(export) as zf:
            zf.extractall(workdir)
        root = workdir
    out = {p.stem: p for p in root.rglob("*.txt") if "labels" in p.parts}
    if not out:
        raise ValueError(f"El export {export} no contiene labels (¿formato correcto?).")
    return out


def compare_frame(
    stem: str,
    originales: list[PoseLabel],
    exportadas: list[PoseLabel],
    img_w: int,
    img_h: int,
    tol_px: float,
) -> tuple[list[str], float]:
    """Compara las personas de un frame; devuelve ``(errores, max_delta_px)``."""
    errores: list[str] = []
    max_delta = 0.0
    if len(originales) != len(exportadas):
        errores.append(
            f"{stem}: {len(originales)} personas originales vs {len(exportadas)} exportadas."
        )
        return errores, max_delta

    pares, _, _ = match_persons(originales, exportadas)
    for i, j in pares:
        for k, (ka, kb) in enumerate(zip(originales[i].kpts, exportadas[j].kpts)):
            if ka[2] != kb[2]:
                errores.append(
                    f"{stem} persona {i} kpt {k}: visibilidad {ka[2]} → {kb[2]} "
                    "(el mapeo outside/occluded NO sobrevive el round-trip)."
                )
                continue
            if ka[2] == 0:
                # v=0 no lleva coordenadas útiles; el invariante coords=0 se
                # normaliza al reconstruir el gold, no se exige al export.
                continue
            dx = abs(ka[0] - kb[0]) * img_w
            dy = abs(ka[1] - kb[1]) * img_h
            delta = max(dx, dy)
            max_delta = max(max_delta, delta)
            if delta > tol_px:
                errores.append(
                    f"{stem} persona {i} kpt {k}: desplazamiento {delta:.2f} px "
                    f"(tolerancia {tol_px} px)."
                )
    return errores, max_delta


def verify(cfg: AuditConfig, original_dir: Path, export: Path) -> float:
    """Corre el gate completo.

    Returns:
        Noise floor medido (máximo desplazamiento en px sobre kpts no tocados).

    Raises:
        SystemExit: con código 1 si el round-trip no es fiel — NO auditar hasta
            resolverlo.
    """
    originales = {p.stem: p for p in original_dir.glob("*.txt")}
    if not originales:
        raise FileNotFoundError(f"No hay labels originales en {original_dir}.")

    errores: list[str] = []
    noise_floor = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        exportadas = labels_from_export(export, Path(tmp))

        faltantes = sorted(set(originales) - set(exportadas))
        for stem in faltantes:
            if read_label_file(originales[stem]):
                errores.append(f"{stem}: presente en el pack pero ausente del export.")

        for stem in sorted(set(originales) & set(exportadas)):
            img_w, img_h = image_size(cfg.images_dir / f"{stem}.jpg")
            errs, delta = compare_frame(
                stem,
                read_label_file(originales[stem]),
                read_label_file(exportadas[stem]),
                img_w,
                img_h,
                cfg.roundtrip_tol_px,
            )
            errores.extend(errs)
            noise_floor = max(noise_floor, delta)

    if errores:
        for err in errores:
            logger.error(err)
        logger.error(
            "GATE FALLIDO: %d discrepancias. NO corregir frames a mano hasta resolverlo.",
            len(errores),
        )
        raise SystemExit(1)

    logger.info(
        "GATE OK: %d frames verificados; visibilidad idéntica en el round-trip.",
        len(originales),
    )
    logger.info(
        "Noise floor medido: %.3f px. reconcile.corregido_thr_px (%.1f px) debe superarlo.",
        noise_floor,
        cfg.corregido_thr_px,
    )
    if cfg.corregido_thr_px <= noise_floor:
        logger.warning(
            "corregido_thr_px (%.1f px) NO supera el noise floor (%.3f px): "
            "la tasa de corrección contaría ruido como corrección. Subirlo.",
            cfg.corregido_thr_px,
            noise_floor,
        )
    return noise_floor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML de auditoría.")
    parser.add_argument(
        "--export", type=Path, required=True, help="ZIP (o dir) exportado de CVAT sin ediciones."
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=None,
        help="Dir de labels del pack importado (default: el pack de round-trip).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = build_config(load_config(args.config))

    original = args.original
    if original is None:
        roundtrip = cfg.pack_dir.parent / f"{cfg.pack_dir.name}_roundtrip"
        base = roundtrip if roundtrip.exists() else cfg.pack_dir
        original = base / "labels" / cfg.subset
    verify(cfg, original, args.export)


if __name__ == "__main__":
    main()
