# pose-ml — Backend de ML para rehabilitación con Pose Estimation

> ⚠️ **Repositorio de entrega académica**
>
> Esta es una copia pública creada para la entrega a la universidad. **No es el
> repositorio original del trabajo**: el original permanece privado. Esta copia no
> incluye ningún dato del dataset UCO ni artefactos de modelo; solo los punteros
> DVC, los scripts que los producen y los metadatos de las corridas de MLflow
> citadas en el informe (ver [Datos y modelos en esta copia](#datos-y-modelos-en-esta-copia)).

## Datos y modelos en esta copia

Los datasets usados por el pipeline **no se distribuyen en este repositorio** y
se comparten por un Drive adjunto al correo de esta entrega:

- **UCO Physical Rehabilitation** (Universidad de Córdoba): dataset de terceros
  que se distribuye solo con fines de investigación y a pedido. Para obtenerlo
  hay que solicitarlo por correo a `inforeha@uco.es`. En el Drive de la entrega
  se incluye la copia procesada que usa este pipeline.
- **Datos propios** (referencias de movimiento y clips del banco): material de
  los autores, también en el Drive.
- **Modelos entrenados** (checkpoints y `.tflite`): en el Drive; en este repo
  quedan los punteros `.dvc` (`models/exports/*.dvc`) que verifican el hash.

Las etiquetas de los datasets se **regeneran** con los scripts versionados acá:

```bash
# UCO crudo -> pool YOLO (cuadros + labels apareados, stride 15/30)
python -m src.data.uco_to_yolo            # en venv-torch, con data/raw/uco/ en su lugar

# Pseudo-etiquetas densas (17 puntos) con RTMDet-m + ViTPose-H
python -m src.data.pseudo_label_vitpose --config configs/pseudo_label_vitpose.yaml

# Mezcla COCO-Pose val2017 del linaje v3
python scripts/make_coco_pose_manifest.py
```

Para reproducir los números del capítulo 7 del informe, la validación cruzada
de la receta del desplegable se corre con `scripts/run_v3_cv.py` y la de la
receta de la fase 5 con `scripts/run_fase5_cv.py`.

### Corridas de MLflow incluidas

`mlruns/` trae solo los **metadatos** (parámetros, métricas y tags — sin
artefactos) de los experimentos que el informe cita por nombre, coherentes con
la conclusión de que el modelo desplegado es **v3** y el de dominio general es
la línea base no desplegada:

| Experimento | Contenido | Referencia del informe |
|---|---|---|
| `fase2-v3-cv` | 5 corridas terminadas, una por partición (GroupKFold k=5) de la receta v3 | sección 5.4, validación de la receta del desplegable |
| `fase5-combined` | corrida madre `11734b62` + 5 corridas anidadas de la receta de la fase 5 | sección 5.4, corrida madre consolidada |
| `fase5-deploy` | comparativa del desplegable en la partición 0: dominio general (stock), v2 y **v3** | sección 5.4, tablas del desplegable |

Para verlas: `mlflow ui --backend-store-uri ./mlruns`.

## Pipeline de 5 fases

| Fase | Técnica | Modelos candidatos |
|------|---------|--------------------|
| 1 | Baseline (Float32) | YOLOv8-pose n/s/x, MoveNet, BlazePose, OpenPose/DensePose (solo referencia) |
| 2 | Pruning estructurado + fine-tuning | YOLOv8-pose, MoveNet, BlazePose |
| 3 | Cuantización standalone (FP32→INT8) | YOLOv8-pose, MoveNet, BlazePose |
| 4 | Knowledge Distillation | Maestro yolov8x-pose → estudiante yolov8n-pose |
| 5 | Combinada (Pruning+Distill+INT8) | Mejor estudiante de Fase 4 |

Todas las fases registran runs comparables en MLflow: `mAP, OKS, PCK, FPS,
latency_ms, size_mb` + hiperparámetros + artefacto exportado.

## Estrategia de entornos (IMPORTANTE)

El stack tiene conflictos conocidos, por eso se usan **venvs separados**
(Python **3.11**). Los dos primeros comparten `requirements-base.txt`:

| Entorno | Archivo | Para qué |
|---------|---------|----------|
| `venv-torch` | `requirements-torch.txt` | YOLOv8-pose: train, distillation y **export TFLite** |
| `venv-tf` | `requirements-tf.txt` | MoveNet, BlazePose y **pruning** (tfmot) |
| `venv-mmpose` | `requirements-mmpose.txt` | ViTPose-H + RTMDet: **auto-etiquetado** (fuera del pipeline de fases) |

### Zonas de atención de versiones

- **tfmot ↔ Keras 2**: `tensorflow-model-optimization` 0.8.x trabaja cómodo con
  Keras 2. Por eso se fija **TF 2.15** (Keras 2 nativo). Si se sube a TF ≥ 2.16
  (Keras 3) hay que instalar `tf-keras` y exportar `TF_USE_LEGACY_KERAS=1`.
- **MediaPipe ↔ protobuf/numpy**: MediaPipe 0.10.x exige `protobuf>=4.25,<5` y
  `numpy<2`. Eso fija `numpy~=1.26` y `protobuf 4.x` en **todo** el stack.
- **Export TFLite de Ultralytics**: corre PyTorch→ONNX→TF→TFLite, así que TF +
  `onnx2tf` viven dentro de `venv-torch`. El par `tensorflow`↔`onnx2tf` es lo
  más frágil: **validarlo con un spike de export antes de avanzar**.
- **mmcv ↔ torch/CUDA**: `mmcv` se distribuye como wheel precompilada contra un
  par `(torch, cuda)` concreto, y arrastra su propio `torch`. Por eso vive en un
  **tercer venv aislado** con `torch 2.1.2+cu121` / `mmcv 2.1.0`: meterlo en
  `venv-torch` rompería el toolchain de export ya validado. Con un torch más
  nuevo, `mim install mmcv` intenta compilar con MSVC (en Windows, ahí se rompe).

### Instalación

```bash
# Entorno YOLO / export
python3.11 -m venv venv-torch
source venv-torch/bin/activate        # Windows: venv-torch\Scripts\activate
pip install -r requirements-torch.txt

# Entorno TF / MediaPipe / pruning (venv separado)
python3.11 -m venv venv-tf
source venv-tf/bin/activate           # Windows: venv-tf\Scripts\activate
pip install -r requirements-tf.txt

# Entorno OpenMMLab / auto-etiquetado (sólo si vas a generar pseudo-labels).
# Requiere GPU: ver requirements-mmpose.txt para el orden exacto de instalación.
make setup-mmpose
```

## Setup inicial (git, DVC→DagsHub, MLflow)

```bash
# --- Git ---
git init
git add .
git commit -m "chore: scaffold pose-ml repo"

# --- DVC apuntando a DagsHub (S3-compatible) ---
# Las credenciales van SIEMPRE en --local (config.local, ignorado por git).
dvc init
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<usuario>/<repo>.s3
dvc remote modify --local origin access_key_id    <DAGSHUB_TOKEN>
dvc remote modify --local origin secret_access_key <DAGSHUB_TOKEN>
# Versionar datos:  dvc add data/raw && git add data/raw.dvc .gitignore && dvc push

# --- MLflow local ---
mlflow server --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000   # Windows: set MLFLOW_TRACKING_URI=...
# Alternativa: MLflow gestionado de DagsHub (apuntar MLFLOW_TRACKING_URI a su URL).
```

## Ejecutar la Fase 1 (baseline)

```bash
# en venv-torch, con el MLflow server arriba
python -m src.training.train_baseline --config configs/baseline_yolov8n.yaml
```

El esqueleto corre end-to-end **incluso sin dataset** (smoke test del pipeline):
crea un run con parámetros, ≥1 métrica y un artefacto. Para entrenar de verdad,
definir `data:` en el YAML de config.

## Auto-etiquetado con ViTPose-H (pseudo-labels)

UCO anota sólo los ~3 joints relevantes al ejercicio y **nunca la cara**: 14 de
los 17 slots COCO salen con `v=0`. Para obtener labels densos se corre un teacher
fuerte sobre las mismas imágenes ya extraídas, en pipeline **top-down**:
RTMDet-m detecta personas → ViTPose-H estima 17 keypoints por recorte.

```bash
# en venv-mmpose (NO en venv-torch)
make teachers    # descarga los checkpoints a models/teachers/

# 1. CALIBRAR los umbrales con datos (obligatorio; ~500 imágenes)
python -m src.data.pseudo_label_vitpose --config configs/pseudo_label_vitpose.yaml --dump-scores

# 2. corrida completa (~2-4 h sobre 74k imágenes en una RTX 3090)
python -m src.data.pseudo_label_vitpose --config configs/pseudo_label_vitpose.yaml
```

### Calibración de umbrales — leer en este orden

Los defaults del YAML son **placeholders**: ViTPose y RTMDet vienen calibrados en
COCO (gente de pie, deporte), y la población objetivo —adultos mayores sentados,
con andador, miembro inferior ocluido— es justo el régimen donde sus scores caen.

1. **`det_thr` primero** (`calibration/det_thr_sweep.csv`). Es el error más caro:
   si el detector no dispara se pierde el **frame entero**, no un keypoint, y es
   irrecuperable sin rehacer la corrida. Si la cola baja de `det_score` está
   poblada, bajar a 0.3–0.4 y dejar que el umbral de keypoints y el flag de
   revisión filtren después.
2. **`vis_thr_low` después** (`calibration/score_stats.csv`), porque depende del
   anterior: cambiar `det_thr` cambia la población de recortes. Es el **único**
   umbral que decide qué aprende el student. Mirar los percentiles por keypoint
   separando los ejercicios de miembro inferior (01–08) del resto.
3. **`vis_thr_high` no se calibra.** Ultralytics binariza la visibilidad (usa
   `v>0` como máscara de la loss), así que `v=1` y `v=2` entrenan idéntico: sólo
   separa `n_v1`/`n_v2` en el manifest para priorizar la revisión humana.

La salida (`data/processed/uco_pseudo/`) espeja la del converter —pool plano con
hardlinks, `folds/` y `pose_dataset_pseudo_fold{k}.yaml`— así que los runners de
fase la consumen cambiando sólo `data:` en el config. El reparto de sujetos por
fold se **verifica** contra el dataset GT: si difiere, aborta, porque comparar
fold *k* pseudo contra fold *k* GT dejaría de ser válido.

Antes de entrenar, revisar `summary.json` (`sin_deteccion` y
`sin_keypoints_validos` deberían ser ~0) y `manifest.csv` (columna
`baja_confianza` para priorizar la revisión humana).

## Estructura

```
src/
├── models/        base.py (PoseModel, ModelMetrics) · yolo.py · registry.py
├── training/      train_baseline.py (Fase 1) · prune · quantize · distill · combined · _kfold
├── evaluation/    metrics.py (OKS/PCK/latencia/size) · pose_eval.py
├── optimization/  pruning.py · quantization.py · distillation.py  (Fases 2/3/5)
├── inference/     keypoint_schema · normalizer · pose_estimator · video_pipeline
├── data/          uco_to_yolo.py (UCO → YOLO) · pseudo_label_vitpose.py (auto-etiquetado)
└── utils/         mlflow_utils.py  (boilerplate de tracking)
configs/           YAMLs por experimento
tests/             test_pseudo_label_format.py  (corre con python plano; no hay pytest aún)
notebooks/         01_eda · 02_benchmarking
```
