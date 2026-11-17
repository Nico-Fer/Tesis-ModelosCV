"""Lanza o REANUDA la validación cruzada completa de la Fase 5 (5 folds × 5 etapas).

Uso, desde la raíz del repo y en cualquier terminal (PowerShell, cmd, Git Bash)::

    venv-torch\\Scripts\\python.exe scripts\\run_fase5_cv.py

Es idempotente: tras un corte de luz se vuelve a correr **el mismo comando**. El
runner (``src.training.combined`` con ``checkpoint_dir`` en la config) saltea los
folds y etapas ya terminados y retoma el entrenamiento en curso desde su último
epoch guardado (ver ``src/utils/checkpoints.py`` y ``docs/fase5_cv_bitacora.md``).

Este script sólo resuelve lo operativo, que es lo que se olvida a las 3 a.m.:

* levanta el servidor MLflow si no está arriba (los artefactos se suben con el
  esquema ``mlflow-artifacts:``, que exige el servidor);
* fija ``MLFLOW_TRACKING_URI`` (la variable de entorno de la máquina está rota, apunta
  a otro proyecto);
* muestra en qué checkpoint quedó la corrida antes de lanzar;
* duplica la salida a ``runs/fase5_cv/train.log`` (append: cada lanzamiento queda
  separado por un encabezado con fecha).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/combined_yolov8n.yaml"
CKPT_DIR = REPO / "runs" / "fase5_cv"
LOG = CKPT_DIR / "train.log"
MLFLOW_URI = "http://127.0.0.1:5000"
STAGES = ("base", "pruned", "pruned_ft", "distilled", "int8")


def _mlflow_up() -> bool:
    try:
        with urllib.request.urlopen(f"{MLFLOW_URI}/health", timeout=3) as resp:
            return resp.status == 200
    except OSError:
        return False


def _ensure_mlflow() -> None:
    if _mlflow_up():
        print(f"MLflow ya está arriba en {MLFLOW_URI}.")
        return
    exe = REPO / "venv-torch" / "Scripts" / "mlflow.exe"
    log = (REPO / "runs" / "mlflow_server.log").open("ab")
    # Proceso desacoplado: un Ctrl+C sobre la corrida no tiene que tirar el servidor.
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    cmd = [str(exe), "server", "--backend-store-uri", "./mlruns"]
    cmd += ["--host", "127.0.0.1", "--port", "5000"]
    subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=log, creationflags=flags)
    for _ in range(60):
        if _mlflow_up():
            print(f"MLflow levantado en {MLFLOW_URI}.")
            return
        time.sleep(2)
    raise SystemExit("El servidor MLflow no levantó en 2 minutos (ver runs/mlflow_server.log).")


def _print_status() -> None:
    state_file = CKPT_DIR / "cv_state.json"
    if not state_file.exists():
        print("Sin checkpoints previos: arranca la CV desde el fold 0.")
        return
    state = json.loads(state_file.read_text(encoding="utf-8"))
    print(f"Reanudando la CV (run padre MLflow {state.get('parent_run_id')}):")
    for k in sorted(state.get("folds", {}), key=int):
        fold = state["folds"][k]
        stages_file = CKPT_DIR / f"fold{k}" / "stages.json"
        stages = json.loads(stages_file.read_text(encoding="utf-8")) if stages_file.exists() else {}
        cells = []
        for stage in STAGES:
            entry = stages.get(stage, {})
            if entry.get("done"):
                cells.append(f"{stage}=OK")
            elif entry.get("progress"):
                prog = entry["progress"]
                cells.append(f"{stage}=ep{prog['epoch']}/{prog['epochs']}")
        estado = "TERMINADO" if fold.get("done") else "en curso"
        print(f"  fold {k}: {estado}  " + "  ".join(cells))


def main() -> int:
    if "venv-torch" not in sys.executable:
        raise SystemExit(f"Correr con venv-torch\\Scripts\\python.exe (no {sys.executable}).")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_mlflow()
    _print_status()

    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": MLFLOW_URI,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [sys.executable, "-m", "src.training.combined", "--config", CONFIG]
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} lanzamiento =====\n")
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
            fh.flush()
        rc = proc.wait()
        fh.write(f"===== {datetime.now().isoformat(timespec='seconds')} fin (rc={rc}) =====\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
