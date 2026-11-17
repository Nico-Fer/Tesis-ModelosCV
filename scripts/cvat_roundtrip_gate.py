"""GATE de round-trip por el camino real de la auditoría (task por jobs + export por job).

Monta el pack chico (``pack_for_cvat --limit 5``) exactamente como se monta la
auditoría real — proyecto nuevo, task con ``segment_size`` chico para que haya
varios jobs, imágenes subidas en orden, pre-anotaciones importadas a nivel task
— lo exporta job por job **sin tocar nada** y corre
:func:`src.audit.verify_roundtrip.verify`: la visibilidad tiene que volver
idéntica y el desplazamiento por debajo de ``roundtrip.tol_px``.

Verifica además lo que el camino nuevo agrega: el mapa frame → job cubre el pack
completo y la cantidad de jobs es ``ceil(n / segment_size)``. Sin un gate OK no
se monta la auditoría real.

Uso (cualquier venv; solo requests + stdlib + ``src.audit``)::

    python -m src.audit.pack_for_cvat --config configs/audit_cvat.yaml --limit 5
    python scripts/cvat_roundtrip_gate.py --config configs/audit_cvat.yaml

El proyecto del gate queda en CVAT para inspección; se puede borrar desde la UI.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cvat_client as cc  # noqa: E402
from cvat_audit import exportar, montar  # noqa: E402

from src.audit.common import build_config, load_config  # noqa: E402
from src.audit.pack_for_cvat import roundtrip_dir  # noqa: E402
from src.audit.verify_roundtrip import verify  # noqa: E402

logger = logging.getLogger("cvat_roundtrip_gate")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML de auditoría.")
    parser.add_argument(
        "--project",
        default=None,
        help="Nombre del proyecto del gate (default: gate-roundtrip-<timestamp>).",
    )
    parser.add_argument(
        "--segment-size", type=int, default=2, help="Frames por job (chico: fuerza varios jobs)."
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=Path("cvat_credentials.local"),
        help="Archivo KEY=VALUE con CVAT_URL/CVAT_USER/CVAT_PASS.",
    )
    parser.add_argument("--timeout", type=float, default=1800.0, help="Timeout (s) por request.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = build_config(load_config(args.config))
    pack_dir = roundtrip_dir(cfg)
    if not pack_dir.exists():
        raise FileNotFoundError(
            f"No existe {pack_dir}; correr antes: "
            "python -m src.audit.pack_for_cvat --config configs/audit_cvat.yaml --limit 5"
        )
    session, base = cc.sesion(args.credentials)

    proyecto_id, task_id, mapa = montar(
        cfg,
        session,
        base,
        pack_dir=pack_dir,
        proyecto_id=None,
        proyecto_nombre=args.project or f"gate-roundtrip-{time.strftime('%Y%m%d-%H%M%S')}",
        task_nombre="gate",
        segment_size=args.segment_size,
        frames_csv=pack_dir / "cvat_frames.csv",
        sobrescribir_mapa=True,
        timeout_s=args.timeout,
    )
    export_dir = pack_dir.parent / f"{pack_dir.name}_export"
    meta = exportar(
        cfg,
        session,
        base,
        frames_csv=pack_dir / "cvat_frames.csv",
        out_dir=export_dir,
        todos=True,
        timeout_s=args.timeout,
    )
    if meta["n_auditados"] != 0:
        # Nadie tocó los jobs del gate: si algo cuenta como auditado, la señal
        # de estado no es la que se cree.
        raise SystemExit(
            f"GATE FALLIDO: {meta['n_auditados']} frames cuentan como auditados sin que "
            "ningún job se haya completado."
        )

    noise_floor = verify(cfg, pack_dir / "labels" / cfg.subset, export_dir / "labels")
    logger.info(
        "GATE OK por el camino nuevo: proyecto %d, task %d, %d frames en %d jobs, "
        "noise floor %.3f px.",
        proyecto_id,
        task_id,
        len(mapa),
        len({f["job_id"] for f in mapa}),
        noise_floor,
    )


if __name__ == "__main__":
    main()
