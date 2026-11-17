# Datos extraídos del repositorio — paper "pipeline de dataset y entrenamiento"

Fecha de extracción: 2026-09-10. Todos los hechos están citados con la ruta exacta
del archivo (y línea aproximada cuando aplica) de donde salen. Ningún número de este
documento fue estimado o redondeado a un valor plausible: si un dato no estaba en un
archivo del repo, se lo marca como **NO ENCONTRADO** al final de cada bloque en lugar
de inventarlo (ver también `PENDIENTES.md`).

---

## 1. Fuente de datos cruda (UCO Physical Rehabilitation)

- **Sujetos en alcance:** 27. **Ejercicios en alcance:** 01–09 y 13 (flexo-extensión
  de rodilla/pierna, elevaciones de brazo). **Cámaras:** 5 por sujeto/ejercicio.
  Fuente: [AGENTS.md](../../AGENTS.md) (sección "Dataset pipeline"), confirmado por
  la estructura real de `data/raw/uco/`.
- **Resolución de video:** 1280×720 px, frames contiguos `0..N-1` por clip. Fuente:
  [AGENTS.md](../../AGENTS.md), sección "Dataset pipeline".
- **Anotación por cuadro:** proviene de `camN_p2d.txt` (una fila por cuadro, columnas
  `x y` por articulación en un orden fijo por ejercicio), **no** de `dataset_2d.json`
  (ese archivo solo anota el sujeto 0 y sirve para aprender el orden de columnas por
  ejercicio). Fuente: [src/data/uco_to_yolo.py](../../src/data/uco_to_yolo.py),
  función `build_exercise_joint_order` + `parse_p2d_file`.
- **Mapeo ejercicio → articulaciones:** 01–04 pierna izquierda, 05–08 pierna derecha
  (3 keypoints c/u), 09 brazo izquierdo, 13 brazo derecho (3 keypoints c/u). Fuente:
  [AGENTS.md](../../AGENTS.md).
- **UCO nunca anota la cara** (los 5 slots faciales de COCO quedan siempre `v=0`).
  Fuente: [src/data/uco_to_yolo.py](../../src/data/uco_to_yolo.py) (mapeo
  `UCO_TO_COCO`, ~L69–84).
- **NO ENCONTRADO:** duración total de los videos crudos, cantidad de frames antes
  del muestreo por stride, y estadística de frames por sujeto/ejercicio del dataset
  crudo (solo existe el resultado *después* de la conversión). Ver `PENDIENTES.md`.

## 2. Conversión a formato YOLO-pose (`src/data/uco_to_yolo.py`)

- **Muestreo temporal (stride):** paso de 30 cuadros, valor por defecto del flag
  CLI `--stride` (`type=int, default=30`). Fuente:
  [src/data/uco_to_yolo.py](../../src/data/uco_to_yolo.py) (~L621), lógica de
  selección `idx % stride == 0 and idx in frames_by_id` (~L447).
- **Volumen resultante:** 74.107 pares imagen/etiqueta en el pool plano
  `images/all` + `labels/all`, ~5–8 GB en disco a stride 30. Fuente:
  [AGENTS.md](../../AGENTS.md), verificado por conteo cruzado con
  `data/processed/uco_pseudo/summary.json` (74.107 imágenes procesadas).
- **Identificador de cuadro (stem):** `"{subject}_{exercise}_cam{cam}_{frame:06d}"`,
  con función inversa `parse_stem` que valida vía regex
  `^(?P<subject>[^_]+)_(?P<exercise>[^_]+)_cam(?P<cam>[^_]+)_(?P<frame>\d+)$` y
  lanza `ValueError` ante cualquier malformación. Fuente:
  [src/data/uco_to_yolo.py](../../src/data/uco_to_yolo.py) (~L134–177). Round-trip
  verificado sobre los 74.107 stems del pool (AGENTS.md).
- **Caja delimitadora:** envolvente (hull) de los keypoints visibles + margen del
  5 % por lado (`margin = 0.05`), recortada a los bordes del cuadro; si no hay
  keypoints visibles, caja centrada `(0.5,0.5)` con `w=h=1.0`. Fuente:
  [src/data/uco_to_yolo.py](../../src/data/uco_to_yolo.py) (~L300–315).
- **Formato de etiqueta:** `class cx cy w h  px1 py1 v1 … px17 py17 v17`
  (1 + 4 + 17×3 = 56 tokens), todo normalizado a `[0,1]`. Verificado por
  `check_label_line()` en
  [tests/test_pseudo_label_format.py](../../tests/test_pseudo_label_format.py)
  (~L31–68). Invariante `v=0 ⇒ px=py=0` verificado en el mismo test.
- **Códigos de visibilidad:** `v=0` no anotado/fuera de cuadro, `v=1` anotado pero
  no visible (ocluido), `v=2` visible. Fuente:
  [src/inference/keypoint_schema.py](../../src/inference/keypoint_schema.py)
  (~L8–11).
- **Partición (GroupKFold por sujeto):** `assign_subject_folds(subjects, n_folds=5,
  seed=42)`, reparto round-robin tras mezclar sujetos, sin fuga (un sujeto entero
  queda de un solo lado). CLI: `--folds default=5`, `--seed default=42`. Fuente:
  [src/data/uco_to_yolo.py](../../src/data/uco_to_yolo.py) (~L417–449, ~L608–609).
- **Particiones materializadas como listas de rutas** (`folds/fold{k}_{train,val}.txt`,
  rutas absolutas), no como copias de imágenes — el pool de imágenes es único y
  plano. Fuente: [AGENTS.md](../../AGENTS.md), sección "Split — GroupKFold by
  subject".
- **Esquema COCO-17 y esqueleto:** `COCO_KEYPOINT_NAMES` (17 nombres),
  `NUM_COCO_KEYPOINTS=17`, `DEFAULT_VISIBILITY_THRESHOLD=0.3`, `COCO_SKELETON` (19
  conexiones). Fuente:
  [src/inference/keypoint_schema.py](../../src/inference/keypoint_schema.py)
  (~L17–61).

## 3. Pseudo-etiquetado con ViTPose-H (`src/data/pseudo_label_vitpose.py`)

- **Arquitectura del anotador:** *top-down* de dos etapas — detector de personas
  (RTMDet-m) + estimador de pose sobre el recorte (ViTPose-H). Solo se retiene la
  detección de mayor confianza por cuadro (UCO es de un único sujeto por escena),
  `max_persons: 1`. Fuente:
  [configs/pseudo_label_vitpose.yaml](../../configs/pseudo_label_vitpose.yaml).
- **Detector — RTMDet-m:** `det_thr: 0.3`, `nms_thr: 0.3`, `person_cat_id: 0`.
  Checkpoint `models/teachers/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth`,
  MD5 `ddb325cbd9439d60924380e86e474b39`. Fuente:
  [configs/pseudo_label_vitpose.yaml](../../configs/pseudo_label_vitpose.yaml) y
  `data/processed/uco_pseudo/run_metadata.json`.
- **Calibración del umbral de detección** (barrido sobre muestra estratificada de
  500 imágenes): `det_thr=0.2` → 0,2 % de cuadros perdidos; `0.3` → 0,2 % (elegido);
  `0.4` → 0,6 %; `0.5` → 1,0 % (~740 cuadros). Fuente:
  [configs/pseudo_label_vitpose.yaml](../../configs/pseudo_label_vitpose.yaml),
  bloque de calibración.
- **Estimador — ViTPose-H:** `vis_thr_low: 0.3`, `vis_thr_high: 0.6`. Checkpoint
  `models/teachers/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth`,
  MD5 `ad498b7fe0998f48c9958251db15a64d`. Fuente: ídem.
- **Semántica de los dos umbrales de visibilidad:** `score < vis_thr_low` → `v=0`
  (descartado); `vis_thr_low ≤ score < vis_thr_high` → `v=1`; `score ≥ vis_thr_high`
  → `v=2`. Como Ultralytics binariza en `v>0`, **solo `vis_thr_low` afecta lo que
  aprende el modelo**; `vis_thr_high` solo sirve para priorizar la revisión humana.
  Fuente: [src/data/pseudo_label_vitpose.py](../../src/data/pseudo_label_vitpose.py)
  (~L245–267) y [configs/pseudo_label_vitpose.yaml](../../configs/pseudo_label_vitpose.yaml).
- **Calibración del umbral de visibilidad:** percentil 5 de confianza nunca por
  debajo de 0,41 (peor caso: codos/muñecas); miembro inferior entre 0,75 y 0,82 (p5)
  y 0,86–0,93 (p50). Con `vis_thr_low=0.3` se retienen en promedio 16,9 de 17
  keypoints por imagen (0,6 % de descarte). Fuente:
  [configs/pseudo_label_vitpose.yaml](../../configs/pseudo_label_vitpose.yaml).
- **Ejecución real (`run_metadata.json`):** fecha 2026-07-25T15:04:47, duración
  7575,8 s (~2,1 h), device CUDA 0 (RTX 3090), seed 42. Dependencias registradas:
  torch 2.1.2+cu121, numpy 1.26.4, opencv 4.11.0, mmcv 2.1.0, mmdet 3.2.0, mmpose
  1.3.2, mmengine 0.10.7. Fuente: `data/processed/uco_pseudo/run_metadata.json`.
- **Resultados agregados (`summary.json`):** 74.107 imágenes procesadas, 73.955
  personas escritas, 69.880 con los 17 keypoints completos (94,3 %), 152 imágenes
  sin ninguna detección, 2.100 cuadros multi-persona, 2.396 cuadros marcados
  `baja_confianza`, confianza media global por keypoint 0,8897. Fuente:
  `data/processed/uco_pseudo/summary.json`.
- **Manifiesto por imagen (`manifest.csv`):** columnas `image, subject, exercise,
  cam, frame, n_personas_detectadas, n_personas_escritas, det_score,
  mean_kpt_score, n_v2, n_v1, n_v0, baja_confianza`. Fuente:
  `data/processed/uco_pseudo/manifest.csv`.
- **Divergencias deliberadas frente al conversor GT:** la caja sale del detector
  (no del hull de keypoints) y todas las coordenadas se recortan a `[0,1]` (ViTPose
  puede extrapolar fuera del recorte). Fuente: [AGENTS.md](../../AGENTS.md),
  sección "Pseudo-labeling with ViTPose-H".
- **Verificación de divergencia de folds:** `folds.verify_against` recalcula el
  reparto de sujetos por fold del dataset GT y aborta ante cualquier diferencia
  con el del dataset pseudo-etiquetado. Fuente:
  [configs/pseudo_label_vitpose.yaml](../../configs/pseudo_label_vitpose.yaml),
  bloque `folds`.
- **NO ENCONTRADO:** mAP de detección/pose del anotador contra un ground-truth
  externo (el pipeline no mide precisión del anotador contra un benchmark, solo
  reporta las estadísticas de confianza propias). Ver `PENDIENTES.md`.

## 4. Cola de anotación manual (CVAT) — `src/audit/` + `configs/audit_cvat.yaml`

- **Selección del subconjunto a auditar:** TODOS los cuadros con `baja_confianza=1`
  del manifiesto de pseudo-etiquetado (motivos, en orden de precedencia:
  `sin_deteccion`, `sin_kpts_validos`, `multi_persona`, `score_bajo`) + una muestra
  de control de `control_size: 150` cuadros sanos, estratificada por
  `stratify_by: [exercise, subject]` con `seed: 42`. Fuente:
  [configs/audit_cvat.yaml](../../configs/audit_cvat.yaml),
  [src/audit/select_audit_subset.py](../../src/audit/select_audit_subset.py).
- **Segundo eje de estratificación (postura):** ejercicios 01–08 → `sedente`;
  ejercicios 09 y 13 → `brazo`. Ejercicio desconocido aborta con `ValueError`.
  Fuente: [configs/audit_cvat.yaml](../../configs/audit_cvat.yaml), bloque
  `estratos`.
- **Empaquetado para CVAT:** ZIP formato "Ultralytics YOLO Pose 1.0" + archivo de
  especificación de esqueleto `cvat_skeleton_labels.json` con los 17 sub-labels en
  **orden COCO** (el importador de CVAT mapea puntos del `.txt` a sub-labels por
  posición, no por nombre). Fuente:
  [src/audit/pack_for_cvat.py](../../src/audit/pack_for_cvat.py).
- **Servidor CVAT:** versión v2.74.1, autoalojado vía Docker Compose, puertos 8080
  (UI/API) y 8090; credenciales en `cvat_credentials.local` (gitignored). Fuente:
  [docs/cvat_setup.md](../cvat_setup.md).
- **Gate de round-trip (obligatorio antes de auditar):** importa un paquete de 5
  cuadros, exporta sin tocar nada y compara pixel a pixel. Resultado medido el
  2026-09-05: visibilidad idéntica en el 100 % de los casos, ruido de fondo máximo
  de **0,005 px** — muy por debajo del umbral de corrección (`corregido_thr_px:
  3.0`), por lo que cualquier desplazamiento mayor a 3 px es corrección real y no
  artefacto de la herramienta. Fuente:
  [src/audit/verify_roundtrip.py](../../src/audit/verify_roundtrip.py),
  [docs/cvat_setup.md](../cvat_setup.md).
- **Dos trampas de API documentadas** (en comentarios de
  [scripts/cvat_roundtrip_gate.py](../../scripts/cvat_roundtrip_gate.py) y
  [docs/cvat_setup.md](../cvat_setup.md)): (1) la carga de un solo request
  requiere las cabeceras `Upload-Start`/`Upload-Finish` y **no** el parámetro de
  query `filename`; (2) CVAT no conserva la caja del detector en el reexport — se
  reconstruye como hull de keypoints + 5 % de margen, igual que en el conversor GT.
- **Reconciliación (`reconcile_corrections.py`):** produce un reporte de auditoría
  (`audit_report.json`, con métricas `tasa_correccion`, `kpts_movidos`,
  `kpts_agregados`, `kpts_eliminados`, `cambios_v1v2` desglosados por `motivo` y
  por `postura`) y un conjunto dorado (`data/processed/uco_gold/`, con
  `gold_stems.txt` como lista de exclusión del entrenamiento). Fuente:
  [src/audit/reconcile_corrections.py](../../src/audit/reconcile_corrections.py).
- **Estado real de la auditoría: NO EJECUTADA.** No existe
  `data/processed/uco_gold/` ni `audit_report.json` en el repositorio al momento
  de esta extracción. El pipeline está implementado y el gate de round-trip pasó,
  pero ninguna cifra de tasa de corrección, cantidad de cuadros efectivamente
  auditados u horas humanas insumidas existe todavía. Ver `PENDIENTES.md` para el
  comando exacto que la generaría.

## 5. Entrenamiento — hiperparámetros reales por fase (YAML íntegros)

Todas las fases comparten: imagen 640 px, AMP activado, `patience: 15`, `device: 0`.

| Fase | Config | Modelo(s) | Batch | Optimizador / lr0 | Notas |
|---|---|---|---|---|---|
| 1 — Baseline | [configs/baseline_yolov8n.yaml](../../configs/baseline_yolov8n.yaml) | yolov8n-pose | 48 | auto (no fijado) | export tflite float32 |
| 2 — Pruning | [configs/prune_yolov8n.yaml](../../configs/prune_yolov8n.yaml) | yolov8n-pose | 48 | AdamW / 0.01 (fine-tune usa lr0/10) | sparsity 0.3, fine_tune_epochs 20 |
| 3 — Quantize | [configs/quantize_yolov8n.yaml](../../configs/quantize_yolov8n.yaml) | yolov8n-pose | — | — | calibración 100–300 img |
| 4 — Distill | [configs/distill_yolov8n.yaml](../../configs/distill_yolov8n.yaml) | teacher yolov8x-pose / student yolov8n-pose | 24 | no fijado | T=4.0, α=0.5, 100 épocas |
| 5 — Combined (CV) | [configs/combined_yolov8n.yaml](../../configs/combined_yolov8n.yaml) | yolov8x→yolov8n | 24 | AdamW / 0.01 | sparsity 0.3, fine_tune 10 ép, T=4.0, α=0.5, distill 80 ép |
| 5 — Combined (deploy) | [configs/combined_yolov8n_deploy.yaml](../../configs/combined_yolov8n_deploy.yaml) | ídem | 24 | AdamW / 0.01 | `cross_validation:false`, `fold:0`; fine_tune **25** ép; distill **20** ép |

- **Por qué `optimizer` se fija explícito:** en modo `optimizer:"auto"` (default de
  Ultralytics), `build_optimizer` **sobrescribe** el `lr0` recibido con uno derivado
  del tamaño del dataset — así se perdía silenciosamente la reducción deliberada a
  `lr0/10` del fine-tune post-poda. Por eso `prune_yolov8n.yaml` y
  `combined_yolov8n.yaml` declaran `optimizer: AdamW` de forma explícita. Fuente:
  [AGENTS.md](../../AGENTS.md), sección "Training phase runners".
- **v3 deploy (lineage corregido):** dataset mixto UCO-pseudo + COCO-pose val2017
  (oversampling ×8, ≈24 % de la mezcla), augmentación reforzada: `scale: 0.9,
  degrees: 10.0, translate: 0.2, mosaic: 1.0`. Fuente:
  [configs/prune_yolov8n_deploy_v3.yaml](../../configs/prune_yolov8n_deploy_v3.yaml).
- **14 claves de augmentación forwardeadas** desde el YAML a Ultralytics
  (`hsv_h/s/v, degrees, translate, scale, shear, perspective, flipud, fliplr,
  mosaic, mixup, copy_paste, close_mosaic, multi_scale`); una clave desconocida
  aborta con `PipelineError`. Fuente:
  [src/models/yolo.py](../../src/models/yolo.py), `augmentation_overrides()`.
- **Infraestructura de cómputo (RTX 3090, 24 GB):** ~233 img/s a batch 24 → ~4,1
  min/época sobre 57.805 cuadros de entrenamiento (fold 0). Evaluación: `.pt` en
  GPU 12,8 ms/imagen; `.tflite` INT8 en CPU 196 ms/imagen (batch fijo en 1) → una
  pasada de evaluación INT8 sobre las 16.150 imágenes de validación cuesta ~53 min.
  Fuente: [AGENTS.md](../../AGENTS.md), "Measured cost on the RTX 3090".

## 6. Compresión para despliegue (poda → recuperación → destilación → INT8)

- **Poda estructurada real (no dispersión no estructurada):** `torch_pruning`
  (`GroupNormPruner` + `GroupMagnitudeImportance`) sobre el grafo de dependencias;
  cirugía `C2f → C2f_v2` previa porque el trazador no interpreta el `split` de
  tensores de C2f; cabezas Detect/Pose excluidas de la poda. Fuente:
  [src/optimization/pruning.py](../../src/optimization/pruning.py).
- **Resultado medido de podar a `sparsity=0.3`** (split de validación fold-0
  completo, 16.150 imágenes): params 3,30 M → 1,91 M (−42 %), OKS 0,846 → **0,000**
  (deja de detectar persona), latencia TFLite 46,7 → 31,8 ms (−32 %). Fuente:
  [AGENTS.md](../../AGENTS.md), "Pruning at sparsity: 0.3 zeroes the model".
- **Recuperación (fine-tune post-poda):** `lr0` reducido a un décimo del original;
  20 épocas (fase 2 aislada) recuperan OKS a 0,935 (1,90 M params) — el estudiante
  supera al maestro COCO `yolov8x-pose` (OKS 0,911, muestra de 600 cuadros, mismo
  split). En la config de despliegue, `fine_tune_epochs: 25` (no 10 como en CV) por
  el colapso total medido tras podar. Fuente: [AGENTS.md](../../AGENTS.md).
- **Destilación (attention transfer):** por nivel de cuello (P3/P4/P5),
  `A(F) = normalize(mean_c(F²))`, pérdida MSE entre mapas de atención
  maestro/estudiante; combinada con la pérdida de tarea:
  `L = α·L_task + (1−α)·T²·L_AT`, con `T=4.0`, `α=0.5` (idéntico en fases 4 y 5).
  Fuente: [src/optimization/distillation.py](../../src/optimization/distillation.py).
- **Por qué destilación y no fine-tuning supervisado para recuperar la poda:** la
  destilación por attention-transfer no requiere etiquetas ground-truth (usa solo
  imágenes + maestro), mientras que UCO anota apenas 3 de 17 keypoints por
  ejercicio — entrenar con ese ground-truth parcial enseña activamente al modelo a
  declarar ausentes los 14 keypoints no anotados (no es solo falta de señal, es
  supresión activa vía la pérdida de confianza/visibilidad). Fuente:
  [docs/requerimientos-de-datos-por-tecnica.md](../requerimientos-de-datos-por-tecnica.md).
- **Cuantización — trampa documentada:** el `*_int8.tflite` que exporta el pipeline
  es *dynamic-range* real (pesos INT8, activaciones y E/S en float32), **no**
  full-integer, aunque el pipeline calibre con hasta 300 imágenes representativas
  (mínimo 100). Las imágenes de calibración no influyen en los bytes finales del
  archivo entregado. Fuente:
  [src/optimization/quantization.py](../../src/optimization/quantization.py),
  [AGENTS.md](../../AGENTS.md), "Two TFLite export traps".
- **Contrato del artefacto final:** E/S float32 (exactamente lo que consume la app
  Android), pesos INT8. Cadena de exportación PyTorch → ONNX → TensorFlow
  SavedModel → TFLite.

## 7. Bug documentado de destilación (Fase 5, hallazgo negativo A)

- **Degradación medida:** una corrida completa de despliegue llevó al estudiante de
  OKS 0,935 → **0,626** (PCK 0,997 → 0,753) en 20 épocas de destilación, con
  coordenadas de keypoints prediciendo infinito. Fuente: [AGENTS.md](../../AGENTS.md),
  sección "BROKEN — the distillation stage diverges".
- **Causa raíz 1 — LR 10× demasiado alto:** el fine-tune post-poda usa `lr0/10 =
  0.001` (AdamW), pero la destilación reenvía `lr0=0.01` sin ajustar —
  reinicia un modelo ya convergido a la tasa completa. La pérdida de tarea sube
  monótonamente desde el primer paso (0,648 → 0,864 en dos épocas).
- **Causa raíz 2 — término de attention-transfer numéricamente inerte:**
  `_attention_transfer_loss` divide por H·W (6400 en P3), aplastando el término 3–4
  órdenes de magnitud (medido: `at_loss=1.17e-3`, contribución final `9.4e-3` contra
  una pérdida de tarea de ~2.0, 0,47 % del total). La forma canónica de Zagoruyko y
  Komodakis usa la **norma**, no la media de cuadrados.
- **Causa raíz 3 — el factor `T²` no aplica a mapas de atención:** viene de Hinton
  et al. (compensa el escalado 1/T² del softmax de logits), y los mapas de
  atención no tienen softmax ni logits.
- **Las causas 2 y 3 se cancelaban entre sí** (por eso no se notaba antes): `T²`
  infla un término que la normalización ya había aplastado. **Corregir solo la 2
  invertiría la falla**: con la norma, `at_loss≈1.01`, término final `≈8.1` contra
  tarea `≈2.0` — la destilación dominaría 4× sobre la tarea. Cualquier arreglo debe
  tocar LR, normalización y peso `T²` juntos, y volver a calibrar `α`. **Estado:
  sin arreglar** — no usar destilación de fases 4/5 para resultados reales todavía.

## 8. Colapso fuera de dominio del checkpoint desplegado (hallazgo negativo B)

- **Síntoma:** `student_pruned_ft_deploy_v2_int8.tflite` medía OKS 0,931 en el
  split de validación in-domain (16.149 cuadros, fold 0) pero producía un
  esqueleto visiblemente aplastado horizontalmente en la app Android. Fuente:
  [AGENTS.md](../../AGENTS.md), "RESOLVED — the deployed model's keypoints
  collapsed out of domain".
- **Diagnóstico (`scripts/diagnose_kpt_ratio.py`):** descarta cuantización (INT8 ≈
  float32 ≈ `.pt`), descarta framing/escala (cuadros UCO completos y recortados
  igual que la app son sanos), y confirma que es **falla de generalización de
  apariencia** del fine-tune post-poda: sobre 79 imágenes reales de COCO
  train2017 (held-out, ninguno de los dos estudiantes las vio), normalizando la
  extensión en x de los keypoints por el **ancho de la caja ground-truth**:
  v2 = 0,701 (ratio predicho/GT), 40/76 imágenes colapsadas, error medio de
  keypoint = 0,314–0,358 de la diagonal GT (el esqueleto cae fuera del cuerpo, no
  solo angosto).
- **Corrección — lineage v3:** poda 0.3 → fine-tune de 25 épocas sobre dataset
  **mixto** (`uco_pseudo` fold-0 + COCO-pose val2017 ×8 ≈ 24 % de la mezcla) con
  augmentación reforzada → INT8. Resultado: ratio 0,913, solo 1/79 colapsadas,
  error 0,049; techo de referencia (modelo stock COCO) = 0,977 / 0 colapsadas /
  0,022 de error. OKS in-domain sube además de 0,931 a 0,941.
- **Nota metodológica:** normalizar por la caja **predicha** (en vez de la GT)
  subestima el colapso, porque un modelo roto también predice una caja mala (v2
  sobreestimaba el ancho de caja OOD ~2,5×).

## 9. Reproducibilidad y versionado

- **MLflow:** un experimento por fase (`fase{n}-...`), corridas nombradas por
  modelo, tags `cv=groupkfold|single-fold`. Config completa aplanada
  (`_flatten_params`, un solo nivel) logueada en cada corrida. Fuente:
  [AGENTS.md](../../AGENTS.md), convenciones de MLflow.
- **DVC:** remoto `origin` → `https://dagshub.com/fernandeznicolas1801/Tesis-ModelosCV.dvc`
  (S3-compatible vía DagsHub). Fuente: `.dvc/config`. Único objeto versionado
  documentado explícitamente: `data/raw/uco_sample.dvc` (133 archivos, ~496 MB).
- **Fail-fast:** `src/utils/strict.py` — dataset/checkpoint/calibración/export
  faltante aborta con `PipelineError`; único escape es `smoke_test: true` en el
  YAML, que degrada a warning y loguea `pipeline_smoke_test=1.0`.
- **Determinismo:** semillas fijas en conversión (`seed=42`) y pseudo-etiquetado
  (`seed=42`), checkpoints del anotador identificados por hash MD5 en
  `run_metadata.json`, selección de fold por nombre de archivo (no por índice
  ordenado — evita el trampa `fold10` < `fold2` lexicográfico con ≥10 folds).
  Fuente: [tests/test_kfold_single.py](../../tests/test_kfold_single.py).
- **No hay `dvc.yaml`** (pipeline DVC declarativo): la orquestación es vía MLflow +
  módulos Python invocados directamente (`python -m src.training.<fase> --config
  <yaml>`). Fuente: `README.md`.

## 10. Licencia / EULA del dataset UCO — estado real (importante)

- El propio borrador de tesis interno
  ([docs/redaccion-secciones-4.4-y-5.1.md](../redaccion-secciones-4.4-y-5.1.md),
  sección 5.1.2.2) marca los términos exactos de licencia como
  **"CITA PENDIENTE / VERIFICAR"** — no hay en el repositorio ningún documento de
  EULA, licencia o términos de uso del dataset UCO Physical Rehabilitation.
- Lo único verificable en el repo son los **resguardos operativos** adoptados
  independientemente de esos términos: almacenamiento local + remoto privado de
  versionado (sin redistribución pública de imágenes), artefactos publicados que
  no permiten reconstruir imágenes/identidades, y uso limitado a entrenamiento/
  evaluación del estimador. Fuente:
  [docs/redaccion-secciones-4.4-y-5.1.md](../redaccion-secciones-4.4-y-5.1.md).
- **Conclusión para el paper:** el eje "EULA restrictiva impide subir a terceros"
  no puede afirmarse como hecho verificado; se enuncia como restricción de diseño
  asumida y se deja `[PENDIENTE]` la verificación de los términos exactos (ver
  `PENDIENTES.md`).

## 11. Métricas de evaluación

- **OKS** con sigmas estándar de COCO (17 valores), enmascarado por `v>0`. Fuente:
  [src/evaluation/metrics.py](../../src/evaluation/metrics.py), `compute_oks`.
- **PCK@0.2** relativo a un tamaño de referencia por instancia. Fuente: ídem,
  `compute_pck`.
- **Latencia:** `measure_latency`, dtype leído de `input_details` del intérprete
  TFLite (crítico: modelos INT8 deben recibir entrada int8). Fuente: ídem.
- **Evaluador unificado (`pose_eval.py`):** agrupa pierna/brazo por el **stem**
  (no por visibilidad, que colapsaría a "todo pierna" con etiquetas densas);
  exige `task="pose"` al cargar un `.tflite` en `YOLO(...)` — sin eso, Ultralytics
  no reconoce la tarea y devuelve cajas sin keypoints, arruinando silenciosamente
  toda métrica INT8 hasta que se corrigió (impacto real medido de INT8 tras la
  corrección: solo −0,005 OKS). Fuente:
  [src/evaluation/pose_eval.py](../../src/evaluation/pose_eval.py),
  [AGENTS.md](../../AGENTS.md).

---

## Resumen de lo NO ENCONTRADO (ver detalle y comando en `PENDIENTES.md`)

1. Conteo exacto de imágenes train/val por cada uno de los 5 folds (existen los
   `.txt`, no un resumen numérico ya calculado).
2. Estadísticas del UCO crudo antes del muestreo (duración de video, frames
   totales, distribución por sujeto/ejercicio).
3. Términos exactos de la licencia/EULA del dataset UCO.
4. Cualquier resultado cuantitativo de la auditoría CVAT (tasa de corrección,
   cantidad de cuadros auditados, horas humanas insumidas) — la auditoría no se
   corrió todavía.
5. mAP/mAP50 de detección o pose del anotador contra un benchmark externo (el
   pipeline solo reporta estadísticas de confianza propias, no exactitud).
6. Costo de anotación manual (tiempo por cuadro) — no instrumentado en el repo.
