# TODO / Pendientes

Trabajo pendiente del proyecto, ordenado por prioridad. Cada item enlaza al
código relevante y deja claro qué bloquea su implementación.

---

## 1. Harness de benchmark unificado (PRIORITARIO)

**Objetivo:** un único script/runner que evalúe cualquier modelo exportado
(`.tflite` o `.pt`) sobre un dataset y produzca un set de métricas comparable
entre familias (YOLO / MoveNet / BlazePose) y entre fases (1–5), logueado a
MLflow. Es la pieza central para sacar las conclusiones de la tesis.

**Qué debe hacer (flujo):**
1. Recibir `--config` (familia, ruta al modelo, dataset) o `--model` + `--data`.
2. Cargar el dataset de validación (imágenes + anotaciones de keypoints COCO).
3. Correr inferencia frame a frame y extraer **pred_keypoints**.
4. Alinear con el **ground-truth** (gt_keypoints + gt_bboxes) por instancia.
5. Calcular las 4 métricas y loguearlas a MLflow.

**Reutilizar lo que YA existe** (no reimplementar):
- `src/evaluation/metrics.py::compute_oks` — OKS (fórmula COCO), ya implementada.
- `src/evaluation/metrics.py::compute_pck` — PCK@thr, ya implementada.
- `src/evaluation/metrics.py::measure_latency` — latencia del `.tflite` (ms),
  con entrada sintética; no necesita dataset.
- `src/evaluation/metrics.py::model_size_mb` — tamaño en MB.
- `src/inference/pose_estimator.py` + `normalizer.py` — ya unifican la salida de
  las 3 familias a keypoints COCO; el harness debería apoyarse en esto para
  extraer `pred_keypoints` de forma agnóstica a la familia.
- `src/utils/mlflow_utils.py` — `start_run` / `log_metrics` para el tracking.

**Métricas a loguear por run:** `oks`, `pck`, `latency_ms`, `size_mb`
(ya existen como campos en `ModelMetrics`).

**BLOQUEANTE — falta el dataset:**
- Hoy no hay dataset de validación disponible.
- `src/data/` está vacío: falta el loader que lea imágenes + anotaciones COCO
  keypoints y devuelva los arrays `(N, K, 3)` que esperan `compute_oks/_pck`.
- Mientras tanto, lo único corrible sin dataset es `measure_latency` (velocidad)
  y, para precisión rápida, `yolo val ... data=coco8-pose.yaml` (built-in de
  Ultralytics, auto-descarga 8 imágenes de muestra).

**Cuando haya dataset:** decidir formato (COCO keypoints JSON vs. formato pose de
Ultralytics) y escribir el loader en `src/data/` antes de armar el harness.

---

## 2. Pendientes relacionados (ya marcados en el código)

- **`src/data/`** — loader de dataset (keypoints COCO). Bloquea el harness (#1) y
  el `evaluate()` real de todas las familias.
- **`src/models/yolo.py::evaluate`** — extraer pred/gt keypoints del set de
  validación para poblar `oks`/`pck` (hoy quedan en `None`; solo se calcula mAP
  vía Ultralytics y `size_mb`).
- **`src/optimization/pruning.py`** — pruning estructurado real con
  torch_pruning (DepGraph); hoy es L1-unstructured.
- **`src/optimization/quantization.py`** — `prepare_calibration_dataset` y
  `quantize_to_int8` son stubs (Fase 3).
- **`src/optimization/distillation.py`** — `run_distillation_training` es stub
  (Fase 4).
- **MoveNet / BlazePose** — `evaluate()` sobre dataset (hoy MoveNet solo reporta
  `size_mb`, BlazePose nada).
