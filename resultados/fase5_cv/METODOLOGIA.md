# Metodología — validación cruzada de la Fase 5

Qué se midió, cómo y con qué limitaciones. Los resultados están en
[RESULTADOS.md](RESULTADOS.md).

## 1. Qué pregunta responde

¿Cuánta precisión se pierde (o se gana) al llevar YOLOv8n-pose a un modelo apto para el
teléfono? El pipeline aplica, en orden, poda estructurada, fine-tune, distillation y
cuantización INT8. La validación cruzada mide cada etapa sobre sujetos que el modelo
nunca vio y estima cuánto varía el resultado entre grupos de sujetos.

## 2. Datos

| | |
|---|---|
| Dataset | UCO (Universidad de Córdoba), ejercicios de rehabilitación |
| Sujetos | 27 |
| Ejercicios | 01–08 (pierna: flexión de rodilla, extensión de pierna) y 09, 13 (brazo: elevación) |
| Cámaras | 5 por toma, 1280×720 |
| Frames | 73.955 con persona (el muestreo toma 1 de cada 15 frames del video) |
| Etiquetas | **pseudo-labels de ViTPose-H**: el detector RTMDet-m encuentra la persona y ViTPose-H estima 17 keypoints COCO |

**Por qué pseudo-labels.** UCO anota sólo las 3 articulaciones del ejercicio de cada clip
(14 de los 17 keypoints COCO quedan siempre vacíos). Las pseudo-labels dan los 17
keypoints en todos los frames. La contracara: la "verdad" contra la que se evalúa es
ViTPose-H, no una anotación humana (ver §7).

### Split: GroupKFold por sujeto, k = 5

Cada sujeto queda entero en un solo fold. Ningún frame de un sujeto de validación se usa
en el entrenamiento de ese fold, y eso incluye otras cámaras y otros ejercicios del mismo
sujeto. Así se mide la generalización a **personas nuevas**, no a frames nuevos de las
mismas personas.

| Fold | Sujetos de validación | Frames train | Frames val | Frames evaluados* |
|---|---|---|---|---|
| 0 | 4, 11, 14, 19, 20, 23 | 57.805 | 16.150 | 16.149 |
| 1 | 2, 3, 10, 13, 15, 25 | 56.671 | 17.284 | 17.277 |
| 2 | 1, 16, 17, 18, 24 | 62.041 | 11.914 | 11.914 |
| 3 | 6, 8, 9, 21, 22 | 59.317 | 14.638 | 14.614 |
| 4 | 0, 5, 7, 12, 26 | 59.986 | 13.969 | 13.968 |

\* Se excluyen los frames donde el detector base no encuentra a nadie, porque no hay
contra qué normalizar (§4). Son 33 de 73.955 en total.

## 3. Pipeline por fold (5 etapas)

Cada fold repite el pipeline completo desde cero, partiendo del YOLOv8n-pose
pre-entrenado en COCO. Config: [`config/combined_yolov8n.yaml`](config/combined_yolov8n.yaml).

| Etapa | Qué se hace | Qué mide |
|---|---|---|
| `base` | Nada: el `yolov8n-pose.pt` COCO original (3,30 M params) | El punto de partida |
| `pruned` | Poda estructurada del 30 % de los canales (torch-pruning, DepGraph, importancia por magnitud L2 de grupo; el head queda intacto). Sin entrenamiento | El **daño** de la poda |
| `pruned_ft` | Fine-tune del modelo podado: 10 epochs, AdamW, lr 0,001 | La **recuperación** |
| `distilled` | Distillation desde yolov8x-pose (COCO): hasta 80 epochs, lr 0,001, early stopping con patience 15 | El aporte de la etapa de KD |
| `int8` | Export TFLite cuantizado de Ultralytics del modelo `distilled` | El costo de cuantizar el **entregable** |

**Detalles que hay que saber para leer los números:**

- **Distillation.** Es *attention transfer* (Zagoruyko & Komodakis, 2017) sobre los
  feature maps del neck (P3, P4, P5). La loss es
  `L = α·L_task + (1−α)·(β/2)·B·L_AT`, con α = 0,5 y β = 1000. `L_task` es la loss
  supervisada de Ultralytics contra las pseudo-labels. **El término de imitación pesa
  sólo ~2–3 % de la loss** después de los primeros pasos. La etapa es, en la práctica,
  sobre todo entrenamiento supervisado.
- **Augmentation.** El fine-tune (10 epochs) corrió sin mosaic, porque Ultralytics lo
  apaga en los últimos 10 epochs (`close_mosaic: 10`) y acá son todos. La distillation
  usa mosaic hasta 10 epochs antes de su final.
- **Selección de checkpoint.** Al final de cada entrenamiento se queda el `best.pt`: el
  epoch de mayor *fitness* de Ultralytics (mAP50-95 de caja + mAP50-95 de pose) sobre el
  split de validación del fold.
- **INT8 = dynamic-range.** El `*_int8.tflite` que entrega Ultralytics 8.3.253 tiene
  pesos INT8 con activaciones y entrada/salida en float32. No es full-integer: la
  calibración con ≤300 imágenes no cambia el archivo entregado. La entrada y salida float
  es el contrato que espera la app Android.

## 4. Evaluación

Todas las etapas se evalúan **igual** y sobre el split de validación **completo** del
fold, con el evaluador unificado (`src/evaluation/pose_eval.py`).

1. **Una predicción por frame:** la detección de mayor confianza (umbral 0,25). Si el
   modelo no detecta a nadie, los 17 keypoints cuentan como error (predicción en cero).
2. **Normalizador fijo:** el bbox de persona sale del detector `base`, se calcula
   **una sola vez** por frame y se reusa en las 5 etapas. Así una etapa no puede "mejorar"
   por predecir un bbox distinto, y las etapas son comparables entre sí. El bbox de UCO
   no sirve, porque rodea sólo las 3 articulaciones anotadas.
3. **GT:** la primera persona de la pseudo-label del frame (UCO tiene una por frame).
   Sólo entran los keypoints con visibilidad `v > 0`.

### Métricas

| Métrica | Definición en este pipeline | Cómo leerla |
|---|---|---|
| **OKS** | Por frame: promedio sobre keypoints visibles de `exp(−d² / (2·s²·(2σ_k)²))`, con los σ de COCO y `s²` = **área del bbox de persona**. Después, promedio sobre frames | 1 = coincidencia perfecta. **No es el AP de COCO** y no se compara con valores publicados: es un OKS medio, y al usar el área del bbox (mayor que el área de segmentación que usa COCO) resulta más permisivo |
| **PCK@0.2** | Fracción de keypoints visibles a menos de 0,2 × diagonal del bbox de persona | Métrica primaria del pipeline. Con este radio (grande) está **casi saturada** (0,98–0,998): sirve para detectar fallos groseros, no para ordenar modelos buenos |
| PCK pierna / brazo | El mismo PCK@0.2 restringido a los clips de ejercicios de pierna (01–08) o de brazo (09, 13) | Ambos grupos se evalúan sobre los 17 keypoints, no sólo sobre la articulación del ejercicio |
| Params | Parámetros del `.pt` | — |
| Tamaño | Bytes del archivo (`.pt` en `base` y `distilled`, `.tflite` en `int8`) | — |
| Latencia | Media de inferencias TFLite (1×640×640×3) en la **CPU x86 de escritorio** | Sirve para comparar etapas fp32 entre sí. **No representa un teléfono**: el INT8 dynamic-range corre más lento que el fp32 en x86 |

## 5. Estadística

- **Media ± σ** sobre los 5 folds. σ es el desvío **poblacional**, igual que el `_std`
  de MLflow. El muestral está en `tablas/resumen_por_etapa.csv`.
- **Diferencias pareadas por fold.** Cada fold compara dos etapas sobre los mismos
  frames, con el mismo normalizador. Por eso la diferencia se calcula dentro de cada
  fold y después se promedia. Eso elimina la variación de dificultad entre grupos de
  sujetos, que es la mayor parte del σ de cada etapa. El fold 0 es el más bajo en todas
  las etapas, y el desvío de las diferencias (0,0028 en la KD, 0,0004 en el INT8) es
  mucho menor que el de los niveles (~0,007). Se reporta el IC 95 % con *t* de Student
  (4 grados de libertad) y el p-valor del *t* pareado.
- **n = 5.** Con 5 folds el IC es ancho y los supuestos del *t* son difíciles de
  verificar. La columna `folds_mismo_signo` de `tablas/deltas_pareados.csv` da una lectura
  sin supuestos: 5/5 folds con el mismo signo es la evidencia más fuerte posible con k = 5.

## 6. Desvíos respecto de una corrida ideal

- **8 interrupciones** (1 accidental, 7 pedidas para apagar la PC), todas reanudadas
  desde checkpoint. Pesos, EMA, optimizer, scaler, epoch y schedule de LR se restauran
  exactos. **No se restauran** el estado aleatorio de la augmentation y el shuffle, ni
  el contador del early stopping, que vuelve a contar desde el epoch reanudado. El efecto
  esperado es del orden de cambiar la semilla. El registro completo está en
  [bitacora_ejecucion.md](bitacora_ejecucion.md).
- **Epochs efectivos de la distillation:** 50 / 80 / 80 / 80 / 80. Sólo el fold 0 cortó
  por early stopping, y su `best.pt` es el del epoch 33, el mismo que habría dado una
  corrida sin pausas.
- **`workers: 2`** en lugar de 8: sólo afecta la velocidad de carga de datos. Con 8 se
  agotaba el archivo de paginación de Windows.

## 7. Amenazas a la validez

1. **La referencia es ViTPose-H, no humana.** Un OKS de 0,96 dice que el modelo coincide
   con ViTPose-H. Si ViTPose-H se equivoca de forma sistemática (por ejemplo, en
   miembros ocluidos por el andador), el modelo aprende y reproduce ese error sin
   penalización.
2. **Sesgo de selección de modelo.** El `best.pt` y el early stopping usan el mismo split
   de validación sobre el que después se reporta. Eso infla un poco `pruned_ft`,
   `distilled` e `int8`, y no afecta a `base` ni a `pruned`. Debería ser chico, porque
   las curvas de validación son casi planas al final (ver `figuras/curvas_entrenamiento.png`),
   pero no está medido.
3. **`base` no es un baseline justo de compresión.** Es un modelo COCO que nunca vio UCO.
   El salto `base → pruned_ft` mezcla recuperación de la poda con adaptación de dominio.
4. **Dominio cerrado.** El OKS de la CV se mide sólo en UCO: cámaras fijas, una
   persona, laboratorio. Un modelo entrenado sólo en UCO ya colapsó una vez fuera de
   dominio (v2). **Los 5 modelos de la CV se pasaron por la compuerta OOD y no la
   aprueban:** detectan a la persona en ~31 % de las imágenes externas con el umbral de
   la app, y el error de punto es 0,42 de la diagonal contra un criterio de ≤ 0,05. Ver
   [gate_ood/GATE_OOD.md](gate_ood/GATE_OOD.md).
5. **Latencia no representativa** del dispositivo destino (§4).

## 8. Glosario

| Término | Significado |
|---|---|
| Fold | Una de las 5 particiones de sujetos; cada fold entrena con 4/5 de los sujetos y evalúa con el 1/5 restante |
| FT | Fine-tune (re-entrenamiento) del modelo podado |
| KD | *Knowledge distillation*: entrenar al modelo chico (estudiante) imitando a uno grande (maestro) |
| Attention transfer | Variante de KD que hace coincidir los mapas de atención espacial de estudiante y maestro |
| Dynamic-range INT8 | Cuantización de los pesos a 8 bits; los cálculos y la E/S siguen en float32 |
| `best.pt` | Checkpoint del epoch con mejor métrica de validación |
| Early stopping | Cortar el entrenamiento si la métrica de validación no mejora en N epochs (acá N = 15) |
| IC 95 % | Intervalo de confianza del 95 % para la media de las diferencias por fold |
