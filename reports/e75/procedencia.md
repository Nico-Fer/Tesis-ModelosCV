# E75 — Procedencia de los datos (A1)

Generado el 2026-10-16 sobre la estación de trabajo (RTX 3090), repo `Tesis-ModelosCV`
en el commit `d1dde1a` más los scripts `scripts/e75_*.py` y `scripts/run_v3_cv.py`
(sin commitear al momento de escribir esto). Cada volumen se contó sobre los manifiestos
que consumen los runners, no sobre la documentación de origen.

## Tabla de procedencia

| Conjunto | Origen | Naturaleza | Anotación | Etapa | Volumen exacto | Licencia / condición | Evidencia |
|---|---|---|---|---|---|---|---|
| UCO, pool de cuadros | UCO Physical Rehabilitation (Aguilar-Ortega et al., 2023), 27 sujetos, ejercicios 01–09 y 13, 5 cámaras 1280×720 | **Real** (video) | ver las dos filas siguientes | entrenamiento y validación, GroupKFold por sujeto k=5 | 74.107 cuadros, **paso de 15** (stems `…_000000`, `_000015`, `_000030`); 12 GB de JPEG | investigación, acceso por solicitud por correo; sin texto de licencia publicado | `data/processed/uco_yolo/images/all`, `data/processed/uco_yolo/folds/` |
| Partición 0 (pseudo) | idem | real | pseudo | entrenamiento / validación | 57.805 / 16.150 cuadros (73.955 = cuadros con persona) | idem | `data/processed/uco_pseudo/folds/fold0_{train,val}.txt` |
| Anotación original de UCO | OptiTrack de 6 cámaras IR, marcadores en hombro-codo-muñeca o cadera-rodilla-tobillo; 3D proyectado a cada cámara RGB; **completado a mano en 2D** donde no se recuperó la articulación (paper, secciones «2. Materials» y «Data Pre-Processing») | **Instrumental** | 3 articulaciones por cuadro (12 distintas en el subconjunto, slots COCO 5–16), 221.865 puntos sobre los 73.955 cuadros con pseudo-etiqueta | referencia de exactitud (no entrena) | 74.107 cuadros | idem UCO | `data/processed/uco_yolo/labels/all`; `camN_p2d.txt` |
| Pseudo-etiquetas | RTMDet-m + ViTPose-H (OpenMMLab) sobre las imágenes reales de UCO | **Derivado** (anotación de modelo sobre imagen real) | 17 puntos; 69.880 con los 17 | entrenamiento; referencia de OKS/PCK | 74.107 procesados, 73.955 con persona, 152 sin detección | propia | `data/processed/uco_pseudo/{manifest.csv,run_metadata.json,summary.json}` |
| COCO-Pose val2017 (mezcla v3) | COCO 2017, split de validación con personas anotadas (lista `coco-pose/val2017.txt` de Ultralytics) | **Real** (fotografías) | manual, de COCO, 17 puntos | **sólo entrenamiento** del linaje v3 (el val sigue siendo 100 % UCO) | 2.346 imágenes únicas × 8 = 18.768 entradas; 24,5 % de la mezcla en la partición 0 (23,2–24,9 % según la partición) | anotaciones CC BY 4.0; imágenes bajo los términos de Flickr | `data/processed/coco_pose/coco_val2017_pose.txt`, `scripts/make_coco_pose_manifest.py` |
| Compuerta OOD (julio) | COCO train2017, una persona grande, frontal, de cuerpo entero | real | manual, de COCO | prueba (aceptación del desplegable) | 79 imágenes; **lista de IDs no conservada** | COCO | `AGENTS.md`, informe Tabla de la compuerta |
| Compuerta OOD (reconstruida) | COCO train2017 y val2017 con criterios explícitos (`scripts/build_ood_gt_set.py`) | real | manual, de COCO | prueba | train2017: 80 de 578 candidatas (semilla 0); val2017: las 37 candidatas | COCO | `data/ood_gate/ids_{train,val}2017.txt` (versionado) |
| Clips propios del banco | grabaciones de los autores con teléfono, frontal y vertical | real | sin puntos; repeticiones del generador de referencias | prueba en dispositivo; 4 son fuente de las referencias | 5 clips, 98,8 s | material propio, no se redistribuye | informe, Tabla de clips del banco |
| Clips UCO del banco (Test Lab) | UCO, sujetos 4, 11 y 14 (validación de la partición 0) | real | anotación original (instrumental) | prueba: precisión en dispositivo | 4 clips × 24 s (600 cuadros c/u) | UCO | app, `androidTest/assets/benchmark` |
| Calibración INT8 | **cuadros de UCO**: los primeros 300 en orden alfabético del val de la partición 0 (todos del sujeto 11, ejercicios 01 y 02) | real | no usa anotación | cuantización; en la variante de rango dinámico **no altera el `_int8.tflite`** que se entrega | 300 cuadros | UCO | `runs/quantize_deploy_v3.log` l. 16–18 (`collecting INT8 calibration images from data\processed\pose_dataset_pseudo_fold0.yaml` … `300 images`) |
| Trayectorias de referencia del comparador | ángulos extraídos de 4 clips propios con el mismo `.tflite` de la app | **derivado** de video real; `configs/references.yaml`: «La referencia NUNCA es sintética» | no corresponde | referencia del comparador | 4 archivos | propia | `src/references/`, `configs/references.yaml` |

### Sobre la calibración INT8

`calibration_image_sample_data_20x128x128x3_float32.npy` (raíz del repo) **no** es el set
de calibración: Ultralytics 8.3.253 lo predescarga siempre como parche de un bug de
onnx2tf (`ultralytics/utils/export/tensorflow.py` l. 85–88) y, cuando hay imágenes propias,
pasa `custom_input_op_name_np_data_path` con `tmp_tflite_int8_calibration_images.npy`
(los cuadros de UCO). Además, el archivo `*_int8.tflite` que se entrega es el renombre de
`*_dynamic_range_quant.tflite` (pesos INT8, activaciones y E/S float32), que no usa la
calibración: ésta sólo da forma a `*_integer_quant.tflite` / `*_full_integer_quant.tflite`,
que el pipeline no devuelve.

### Por qué v2 figura con 76 y no con 79 en la compuerta de julio

En el código de julio (`git show HEAD:scripts/diagnose_kpt_ratio.py`, `cmd_gt_set`), el
denominador de la razón, del error y de «colapsadas» es `len(pred_rel)`: imágenes con GT de
≥ 4 puntos **y** una detección con ≥ 4 puntos de confianza > 0,5. En 3 de las 79, v2 no dio
un esqueleto utilizable. El `n_det` de julio no se conservó. Sobre el set reconstruido, el
mismo criterio deja a v2 con 73 de 80 evaluables (7 «sin esqueleto»).

### Solapamiento con el preentrenamiento de `yolov8n-pose.pt`

`coco-pose.yaml` de Ultralytics entrena sobre `train2017.txt` (56.599 imágenes) y valida
sobre `val2017.txt` (2.346). Verificado contra `coco2017labels-pose.zip`
(github.com/ultralytics/assets, release v0.0.0):

- las **80/80** imágenes de la compuerta reconstruida de train2017 están en el split de
  entrenamiento del modelo de partida;
- las **37/37** de val2017 están en el split de validación (ninguna en entrenamiento);
- las 2.346 de la mezcla del v3 son exactamente el split de validación (ninguna en
  entrenamiento), así que no se solapan con la compuerta de train2017.

Las 79 originales se tomaron de train2017 con el mismo requisito de anotación; su lista no
se conservó, pero toda imagen de train2017 con puntos anotados está en esa lista de 56.599.
**Consecuencia**: para el stock (techo de la tabla) la compuerta train2017 no es externa;
v2/v3 parten de ese modelo pero se reentrenaron después de la poda (exposición indirecta).
val2017 es limpio para stock, v2 y la CV de la Fase 5; no para v3 (lo usó al entrenar).

## Aumentaciones (Ultralytics 8.3.253, `runs/pose/*/args.yaml`)

| Parámetro | v2 (`pruned_finetune6`) | v3 (`pruned_finetune8`) | CV Fase 5, ajuste fino / destilación (`runs/fase5_cv/fold0/…/args.yaml`) |
|---|---|---|---|
| scale | 0,5 | **0,9** | 0,5 / 0,5 |
| degrees | 0 | **10** | 0 / 0 |
| translate | 0,1 | **0,2** | 0,1 / 0,1 |
| fliplr (con `flip_idx`) | 0,5 | 0,5 | 0,5 / 0,5 |
| flipud | 0 | 0 | 0 / 0 |
| hsv_h / hsv_s / hsv_v | 0,015 / 0,7 / 0,4 | idem | idem |
| mosaic (**combina 4 imágenes**) | 1,0 | 1,0 | 1,0 / 1,0 |
| close_mosaic | 10 (sin mosaico en las épocas 16–25) | 10 | 10: el ajuste fino de 10 épocas **no** usa mosaico; la destilación sí, salvo las últimas 10 |
| mixup / copy_paste / cutmix | 0 / 0 / 0 (no combinan) | idem | idem |
| shear / perspective | 0 / 0 | idem | idem |
| auto_augment / erasing | randaugment / 0,4 — **inactivos en pose**: Ultralytics sólo los usa en `classify_augmentations` | idem | idem |
| Albumentations | no instalado en `venv-torch` → no se aplica | idem | idem |

**Validación**: `ultralytics/data/build.py` construye el dataset con `augment=mode == "train"`;
en validación sólo hay letterbox. La única aumentación que combina varias imágenes es el
mosaico.

## Pruebas de humo

`configs/smoke_yolov8n.yaml`: `smoke_test: true`, sin `data:`, experimento MLflow
`smoke-test`. Corre el esqueleto sin datos reales y loguea `pipeline_smoke_test = 1.0`,
que invalida sus métricas (`AGENTS.md`, sección de `src/utils/strict.py`). Ninguna corrida
así marcada entra al informe.

## Latencia de escritorio: entrada sintética

`src/evaluation/metrics.py::measure_latency` alimenta al intérprete TFLite con
`np.random.rand(*shape)` (o `randint` para entradas enteras), 10 pasadas de calentamiento y
50 medidas. Es la **única entrada sintética** del trabajo: mide tiempo de cómputo, que en
esta red no depende del contenido (sin control de flujo dependiente del dato; la NMS va
fuera del `.tflite`), y no entra en ninguna métrica de exactitud. Las latencias en teléfono
(Test Lab y uso real) se midieron sobre cuadros de video reales.

## Modelo que empaqueta hoy la app

`Tesis-AppMobile` (`f8aaaae`), `PoseModelSpec.ACTIVE = STOCK_YOLOV8N` →
`yolov8n-pose_int8.tflite`, MD5 `ee27dcf9f0ad0c25fcf37868e097ad68`, idéntico a
`models/yolov8n-pose_saved_model/yolov8n-pose_int8.tflite` (control stock). La app también
empaqueta `student_pruned_ft_deploy_v2_int8.tflite` (`babaf180…`) y
`student_pruned_ft_deploy_v3_int8.tflite` (`63a694b7…`), idénticos a los de `models/`, pero
no los carga.
