"""Construcción y validación del JSON de referencia — schema v1 EXACTO.

Campos (ni más ni menos)::

    schemaVersion: 1
    exerciseId
    angleDefinitions: [{name, pointA, vertex, pointC, weight}]
    driverAngleIndex
    referenceTrajectory: [[D floats rad] × T]   (redondeada a 4 decimales)
    metadata: {fps, version, source}            (source = "video:<archivo>")
    scoring: {similarityScale, goodThreshold}   (placeholders, calibrar en app)

Sólo stdlib + numpy: puro y testeable en cualquier venv. El consumidor es el
parser Kotlin ``ReferenceJson`` de la app — validar cada JSON contra él antes
de embarcarlo (ver docstring de :mod:`src.references.cli`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.references.angles import wrap_diff

SCHEMA_VERSION = 1
# Versión del generador que queda en metadata.version (bump al cambiar la emisión).
GENERATOR_VERSION = 1
_TOP_KEYS = {
    "schemaVersion",
    "exerciseId",
    "angleDefinitions",
    "driverAngleIndex",
    "referenceTrajectory",
    "metadata",
    "scoring",
}
_ANGLE_KEYS = {"name", "pointA", "vertex", "pointC", "weight"}
_METADATA_KEYS = {"fps", "version", "source"}
_SCORING_KEYS = {"similarityScale", "goodThreshold"}


def build_reference(
    exercise_id: str,
    angle_definitions: list[dict],
    driver_angle_index: int,
    trajectory: np.ndarray,
    fps: float,
    video_name: str,
    scoring: dict,
) -> dict:
    """Arma el documento schema v1. ``trajectory`` es ``(T, D)`` en radianes.

    ``angle_definitions`` van tal cual la config (nombres de keypoints como
    strings COCO); ``scoring`` trae los placeholders de la config.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError(f"Trayectoria debe ser (T, D); llegó shape {trajectory.shape}")
    if not np.all(np.isfinite(trajectory)):
        bad = int(np.count_nonzero(~np.isfinite(trajectory)))
        raise ValueError(f"Trayectoria con {bad} valores no finitos: no se emite el JSON")
    if trajectory.shape[1] != len(angle_definitions):
        raise ValueError(
            f"D={trajectory.shape[1]} no coincide con {len(angle_definitions)} angleDefinitions"
        )
    if not 0 <= driver_angle_index < len(angle_definitions):
        raise ValueError(f"driverAngleIndex {driver_angle_index} fuera de rango")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "exerciseId": exercise_id,
        "angleDefinitions": [
            {
                "name": str(d["name"]),
                "pointA": str(d["pointA"]),
                "vertex": str(d["vertex"]),
                "pointC": str(d["pointC"]),
                "weight": float(d.get("weight", 1.0)),
            }
            for d in angle_definitions
        ],
        "driverAngleIndex": int(driver_angle_index),
        "referenceTrajectory": [[round(float(v), 4) for v in row] for row in trajectory],
        "metadata": {
            "fps": float(fps),
            "version": GENERATOR_VERSION,
            "source": f"video:{Path(video_name).name}",
        },
        # TODO: calibrar — placeholders, no valores definitivos.
        "scoring": {
            "similarityScale": float(scoring["similarityScale"]),
            "goodThreshold": float(scoring["goodThreshold"]),
        },
    }


def validate_reference(doc: dict, *, first_last_tol_rad: float = 0.35) -> list[str]:
    """Errores del documento contra el schema v1 (lista vacía = válido).

    ``first_last_tol_rad`` acota |primera − última| fila por dimensión (con
    wraparound): el ciclo es reposo→extremo→reposo.  # TODO: calibrar
    """
    errors: list[str] = []
    if set(doc) != _TOP_KEYS:
        errors.append(f"Claves top-level: {sorted(doc)} != {sorted(_TOP_KEYS)}")
        return errors
    if doc["schemaVersion"] != SCHEMA_VERSION:
        errors.append(f"schemaVersion {doc['schemaVersion']} != {SCHEMA_VERSION}")
    if not isinstance(doc["exerciseId"], str) or not doc["exerciseId"]:
        errors.append("exerciseId vacío o no-string")
    defs = doc["angleDefinitions"]
    if not defs:
        errors.append("angleDefinitions vacío")
    for i, d in enumerate(defs):
        if set(d) != _ANGLE_KEYS:
            errors.append(f"angleDefinitions[{i}]: claves {sorted(d)} != {sorted(_ANGLE_KEYS)}")
    if not 0 <= doc["driverAngleIndex"] < len(defs):
        errors.append(f"driverAngleIndex {doc['driverAngleIndex']} fuera de [0, {len(defs)})")
    if set(doc["metadata"]) != _METADATA_KEYS:
        errors.append(f"metadata: claves {sorted(doc['metadata'])} != {sorted(_METADATA_KEYS)}")
    elif not str(doc["metadata"]["source"]).startswith("video:"):
        errors.append(f"metadata.source {doc['metadata']['source']!r} no empieza con 'video:'")
    if set(doc["scoring"]) != _SCORING_KEYS:
        errors.append(f"scoring: claves {sorted(doc['scoring'])} != {sorted(_SCORING_KEYS)}")

    traj = doc["referenceTrajectory"]
    if not traj:
        errors.append("referenceTrajectory vacía")
        return errors
    n_dims = len(defs)
    for t, row in enumerate(traj):
        if len(row) != n_dims:
            errors.append(f"fila {t}: {len(row)} valores != D={n_dims}")
            break
        for v in row:
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                errors.append(f"fila {t}: valor no finito {v!r}")
                break
    first, last = np.asarray(traj[0]), np.asarray(traj[-1])
    gap = np.abs(wrap_diff(first, last))
    if np.any(gap > first_last_tol_rad):
        worst = int(np.argmax(gap))
        errors.append(
            f"primera≈última falla en dim {worst}: |Δ|={float(gap[worst]):.3f} rad "
            f"> {first_last_tol_rad} (el ciclo no cierra en reposo)"
        )
    return errors


def write_reference(doc: dict, output_dir: Path) -> Path:
    """Escribe ``{exerciseId(lower)}.json``; valida antes y aborta si hay errores.

    El filename va en minúscula (contrato de assets de la app, case-sensitive);
    el campo ``exerciseId`` DENTRO del JSON conserva su mayúscula.
    """
    errors = validate_reference(doc)
    if errors:
        raise ValueError("JSON inválido:\n  - " + "\n  - ".join(errors))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{doc['exerciseId'].lower()}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
