"""Reensamblado y extracción del dataset UCO (zip dividido en 12 partes).

El dataset crudo se distribuye como un **ZIP64 partido** con ``split`` en
``Datasets/UCO/dataset.zip.001 … .012`` (~11.9 GB). Las partes son fragmentos
contiguos de un único zip (la ``.001`` arranca con la firma ``PK\\x03\\x04``),
así que **concatenarlas en orden** reconstruye un zip válido que cualquier
extractor estándar puede abrir.

Estructura del dataset una vez extraído::

    dataset/
        clips_mp4/{sujeto 0..26}/{accion NN}/
            cam0.mp4 … cam4.mp4          # 5 vistas sincronizadas
            cam0_p2d.txt … cam4_p2d.txt  # pose 2D por cámara (px), 1 fila/frame
            p3d.txt                      # pose 3D
        dataset_2d.json                  # anotaciones 2D consolidadas (fuente principal)
        dataset_3d.json                  # anotaciones 3D consolidadas

Este módulo NO se trackea en git como dato: sólo orquesta la extracción. El
dataset crudo y el extraído van por DVC (ver CLAUDE.md › Data Versioning).

Entorno requerido: cualquiera de los dos venvs (sólo usa la stdlib).
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

# Nombre base de las partes y cantidad esperada (ajustar si cambia el origen).
DEFAULT_PART_GLOB = "dataset.zip.*"


def concat_parts(parts: Iterable[Path], output_zip: Path, chunk_size: int = 1 << 26) -> Path:
    """Concatena las partes de un zip dividido en un único archivo .zip.

    Copia byte-a-byte en orden. Usa un buffer de 64 MB por defecto para no
    cargar las partes completas en memoria (cada una pesa ~1 GB).

    Args:
        parts: rutas de las partes, **ya ordenadas** (``.001`` primero).
        output_zip: ruta del .zip reconstruido.
        chunk_size: tamaño del buffer de copia en bytes.

    Returns:
        La ruta ``output_zip``.
    """
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with output_zip.open("wb") as dst:
        for part in parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, dst, chunk_size)
    return output_zip


def find_parts(raw_dir: Path, glob: str = DEFAULT_PART_GLOB) -> list[Path]:
    """Devuelve las partes del zip dividido ordenadas por su sufijo numérico.

    Args:
        raw_dir: carpeta que contiene ``dataset.zip.001 … .NNN``.
        glob: patrón de búsqueda de las partes.

    Returns:
        Lista de rutas ordenada ascendente por número de parte.

    Raises:
        FileNotFoundError: si no se encuentra ninguna parte.
    """
    parts = sorted(raw_dir.glob(glob), key=lambda p: int(p.suffix.lstrip(".")))
    if not parts:
        raise FileNotFoundError(f"No se encontraron partes '{glob}' en {raw_dir}")
    return parts


def extract_uco(
    raw_dir: str | Path,
    dest_dir: str | Path,
    *,
    keep_zip: bool = False,
) -> Path:
    """Reensambla y extrae el dataset UCO partido.

    Flujo:
        1. Localiza ``dataset.zip.*`` en ``raw_dir`` y las ordena.
        2. Las concatena en un ``dataset.zip`` temporal dentro de ``dest_dir``.
        3. Extrae el zip a ``dest_dir``.
        4. Borra el zip temporal salvo que ``keep_zip`` sea True.

    Necesita ~12 GB para el zip temporal + lo que ocupe el contenido extraído.

    Args:
        raw_dir: carpeta con las partes (p.ej. ``../Datasets/UCO``).
        dest_dir: carpeta destino del contenido extraído.
        keep_zip: si True conserva el ``dataset.zip`` reconstruido.

    Returns:
        Ruta de ``dest_dir`` (raíz del dataset extraído).
    """
    raw_dir = Path(raw_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    parts = find_parts(raw_dir)
    merged_zip = dest_dir / "dataset.zip"
    concat_parts(parts, merged_zip)

    with zipfile.ZipFile(merged_zip) as zf:
        # TODO(data): si la extracción completa es muy pesada para iterar,
        # exponer un parámetro `members` para extraer sólo un subconjunto de
        # sujetos/clips y prototipar el conversor más rápido.
        zf.extractall(dest_dir)

    if not keep_zip:
        merged_zip.unlink(missing_ok=True)
    return dest_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reensambla y extrae el dataset UCO partido.")
    parser.add_argument(
        "--raw-dir",
        default="../Datasets/UCO",
        help="Carpeta con dataset.zip.001 … .NNN (relativa a la raíz del repo).",
    )
    parser.add_argument(
        "--dest-dir",
        default="data/raw/uco",
        help="Carpeta destino del dataset extraído.",
    )
    parser.add_argument("--keep-zip", action="store_true", help="Conservar el zip reconstruido.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = extract_uco(args.raw_dir, args.dest_dir, keep_zip=args.keep_zip)
    print(f"Dataset UCO extraído en: {out}")
