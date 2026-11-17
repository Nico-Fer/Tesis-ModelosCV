"""E75: validación cruzada de cinco particiones de la receta del modelo desplegable (v3).

La CV de la Fase 5 (``scripts/run_fase5_cv.py``) valida otra receta, entrenada sólo con
UCO, cuyos modelos no pasan la compuerta fuera de dominio. El artefacto que se despliega
es el linaje v3 y, hasta acá, sus métricas eran de la partición 0 sola. Este script repite
**exactamente** la receta de ``configs/prune_yolov8n_deploy_v3.yaml`` en cada partición:

1. poda estructurada 0,3 + ajuste fino de 25 épocas sobre la mezcla
   ``uco_pseudo`` fold k + COCO-Pose val2017 ×8 (``pose_dataset_pseudo_coco_fold{k}.yaml``,
   que se genera acá para k = 1..4 con la misma forma que el de k = 0), con
   ``src.training.prune`` en modo de partición única (``cross_validation: false``,
   ``fold: k``) y el checkpoint en ``runs/v3_cv/fold{k}/`` (nunca en
   ``models/student_pruned_ft_deploy_v3.pt``, que es el desplegado);
2. en segundo plano, mientras entrena la partición siguiente: evaluación FP32, exportación
   INT8 con la receta de ``configs/quantize_deploy_v3.yaml`` y evaluación INT8, ambas con
   ``scripts/e75_eval_model.py`` (contra pseudo-etiquetas, contra la anotación original de
   UCO y la compuerta fuera de dominio).

Es reanudable: se vuelve a correr el mismo comando y saltea lo terminado
(``runs/v3_cv/state.json``). Cada partición queda en un run del experimento MLflow
``fase2-v3-cv`` con la etiqueta ``fold``.

Uso::

    venv-torch\\Scripts\\python.exe scripts\\run_v3_cv.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from run_fase5_cv import MLFLOW_URI, _ensure_mlflow  # noqa: E402

BASE_CFG = REPO / "configs" / "prune_yolov8n_deploy_v3.yaml"
OUT = REPO / "runs" / "v3_cv"
STATE = OUT / "state.json"
LOG = OUT / "train.log"
EXPERIMENT = "fase2-v3-cv"
N_FOLDS = 5
DATA = REPO / "data" / "processed"
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_state() -> dict[str, Any]:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"creado": _now(), "experimento": EXPERIMENT, "folds": {}}


def _save_state(state: dict[str, Any]) -> None:
    with _LOCK:
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, STATE)


def _log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with _LOCK, LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def ensure_fold_yaml(k: int) -> Path:
    """``pose_dataset_pseudo_coco_fold{k}.yaml``: el de k = 0 con el fold cambiado."""
    path = DATA / f"pose_dataset_pseudo_coco_fold{k}.yaml"
    if path.exists():
        return path
    base = yaml.safe_load((DATA / "pose_dataset_pseudo_coco_fold0.yaml").read_text("utf-8"))
    base["train"] = [f"uco_pseudo/folds/fold{k}_train.txt", "coco_pose/coco_val2017_pose.txt"]
    base["val"] = f"uco_pseudo/folds/fold{k}_val.txt"
    header = (
        f"# Mezcla de la receta v3 para la partición {k}: uco_pseudo fold{k} + COCO-Pose\n"
        "# val2017 x8. Generado por scripts/run_v3_cv.py con la misma forma que el fold 0.\n"
    )
    path.write_text(header + yaml.safe_dump(base, sort_keys=False, allow_unicode=True), "utf-8")
    return path


def fold_config(k: int) -> Path:
    cfg = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    fold_dir = OUT / f"fold{k}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    cfg["cross_validation"] = False
    cfg["fold"] = k
    cfg["pruning"]["save_checkpoint"] = str(fold_dir / "student_pruned_ft.pt")
    cfg["export"] = {"format": "none"}  # el INT8 lo produce la evaluación, como quantize_deploy_v3
    cfg["mlflow"] = {"experiment": EXPERIMENT}
    path = fold_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _run(cmd: list[str], env: dict[str, str], tag: str, log: Path) -> int:
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {_now()} {tag}: {' '.join(cmd)} =====\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True
        )
        return proc.wait()


def _find_run_id(k: int) -> str | None:
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    client = MlflowClient(MLFLOW_URI)
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.fold = '{k}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def evaluate_fold(k: int, state: dict[str, Any], env: dict[str, str]) -> None:
    """FP32 (GPU) y luego INT8 (export + CPU) de la partición ``k``."""
    fold = state["folds"][str(k)]
    ckpt = Path(fold["checkpoint"])
    py = sys.executable
    script = str(REPO / "scripts" / "e75_eval_model.py")
    run_id = fold.get("run_id")
    mlflow_args = ["--mlflow-run-id", run_id] if run_id else []
    if not fold.get("eval_fp32"):
        cmd = [py, script, "--model", str(ckpt), "--fold", str(k), "--tag", "v3cv_fp32", "--gate"]
        rc = _run(cmd + mlflow_args, env, f"eval FP32 fold {k}", ckpt.parent / "eval_fp32.log")
        _log(f"fold {k}: evaluación FP32 rc={rc}")
        if rc == 0:
            fold["eval_fp32"] = _now()
            _save_state(state)
    if not fold.get("eval_int8"):
        int8 = ckpt.parent / f"{ckpt.stem}_saved_model" / f"{ckpt.stem}_int8.tflite"
        src = ["--model", str(int8)] if int8.exists() else ["--export-int8-from", str(ckpt)]
        cmd = [py, script, *src, "--fold", str(k), "--tag", "v3cv_int8", "--gate", "--threads", "4"]
        rc = _run(cmd + mlflow_args, env, f"eval INT8 fold {k}", ckpt.parent / "eval_int8.log")
        _log(f"fold {k}: evaluación INT8 rc={rc}")
        if rc == 0:
            fold["eval_int8"] = _now()
            fold["int8"] = str(int8)
            _save_state(state)


def main() -> int:
    if "venv-torch" not in sys.executable:
        raise SystemExit(f"Correr con venv-torch\\Scripts\\python.exe (no {sys.executable}).")
    OUT.mkdir(parents=True, exist_ok=True)
    _ensure_mlflow()
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": MLFLOW_URI,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    for k in range(N_FOLDS):
        ensure_fold_yaml(k)
    state = _load_state()
    evals: list[threading.Thread] = []
    for k in range(N_FOLDS):
        fold = state["folds"].setdefault(str(k), {})
        if not fold.get("train_done"):
            cfg = fold_config(k)
            fold["inicio"] = _now()
            _save_state(state)
            _log(f"fold {k}: entrenamiento (poda 0,3 + 25 épocas sobre la mezcla)")
            cmd = [sys.executable, "-m", "src.training.prune", "--config", str(cfg)]
            rc = _run(cmd, env, f"entrenamiento fold {k}", cfg.parent / "train.log")
            if rc != 0:
                _log(f"fold {k}: el entrenamiento falló (rc={rc}); se corta la CV")
                for t in evals:
                    t.join()
                return rc
            fold["train_done"] = _now()
            fold["checkpoint"] = str(OUT / f"fold{k}" / "student_pruned_ft.pt")
            fold["run_id"] = _find_run_id(k)
            _save_state(state)
            _log(f"fold {k}: entrenado, run MLflow {fold['run_id']}")
        if not (fold.get("eval_fp32") and fold.get("eval_int8")):
            t = threading.Thread(target=evaluate_fold, args=(k, state, env), name=f"eval{k}")
            t.start()
            evals.append(t)
    for t in evals:
        t.join()
    state["terminado"] = _now()
    _save_state(state)
    _log("CV v3 terminada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
