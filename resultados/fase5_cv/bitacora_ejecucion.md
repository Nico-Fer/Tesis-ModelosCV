# Bitácora — Validación cruzada completa de la Fase 5 (yolov8n-pose, 5 folds × 5 etapas)

Registro vivo de la corrida que produce los números finales de la Fase 5 para el
informe: hallazgos, decisiones y checkpoints. La tabla de avance de más abajo la
regenera el propio pipeline en cada checkpoint (`src/utils/checkpoints.py`); el resto
del documento se escribe a mano.

## Resultados finales (CV completa, 5 folds — terminada 2026-10-14 23:49)

La CV terminó sin errores (`rc=0`). Run padre MLflow **`11734b62e1504b15a146dacd944f96a5`**
(experimento `fase5-combined`, FINISHED) con los 5 runs de fold FINISHED y sin runs
huérfanos. Cada fold guardó su `best_int8.tflite` (2,39 MB) como artefacto. El run
padre tiene los agregados por etapa: `{métrica}_{etapa}_mean` / `_std` / `_n_folds`.
Durante la corrida hubo 8 pausas (1 accidental, 7 pedidas); se retomó siempre desde el
checkpoint.

**Tabla 1 — media ± σ por etapa** (σ poblacional, la misma que el `_std` de MLflow;
n = 5 en todas las celdas):

| Etapa | OKS | PCK@0.2 | PCK pierna | PCK brazo | Params | Tamaño | Latencia TFLite* |
|---|---|---|---|---|---|---|---|
| base (COCO) | 0,8552 ± 0,0074 | 0,9810 ± 0,0019 | 0,9769 ± 0,0025 | 0,9967 ± 0,0008 | 3.295.470 | 6,83 MB | 44,3 ± 3,2 ms (fp32) |
| pruned | 0,0000 ± 0,0000 | 0,0000 ± 0,0000 | 0,0000 | 0,0000 | 1.907.230 | — | 31,5 ± 2,5 ms (fp32) |
| pruned_ft | 0,9569 ± 0,0076 | 0,9951 ± 0,0028 | 0,9942 ± 0,0036 | 0,9982 ± 0,0005 | 1.907.230 | — | — |
| distilled | 0,9676 ± 0,0063 | 0,9979 ± 0,0011 | 0,9978 ± 0,0015 | 0,9983 ± 0,0005 | 1.907.230 | 4,07 MB | 30,9 ± 0,4 ms (fp32) |
| **int8** | **0,9640 ± 0,0062** | **0,9980 ± 0,0011** | 0,9978 ± 0,0015 | 0,9983 ± 0,0005 | — | **2,39 MB** | 187,0 ± 2,4 ms* |

\* Latencia medida en la CPU x86 de escritorio. El `_int8.tflite` es dynamic-range
y en x86 corre más lento que el fp32, así que esa celda no representa un teléfono:
la latencia INT8 hay que medirla on-device. Entre las etapas fp32 la comparación sí
vale, aunque la latencia de escritorio varía entre corridas (σ de `base` = 3,2 ms).

**Tabla 2 — diferencias pareadas por fold** (OKS; media, IC 95 % con t de Student,
4 g.l.):

| Transición | Por fold (F0…F4) | Media | IC 95 % |
|---|---|---|---|
| base → pruned_ft | +0,097 / +0,100 / +0,094 / +0,104 / +0,114 | **+0,1017** | [+0,092, +0,112] |
| pruned_ft → distilled | +0,013 / +0,011 / +0,013 / +0,011 / +0,006 | **+0,0107** | [+0,007, +0,014] |
| distilled → int8 | −0,0036 / −0,0033 / −0,0035 / −0,0035 / −0,0043 | **−0,0036** | [−0,0041, −0,0032] |
| base → int8 (pipeline completo) | +0,107 / +0,107 / +0,103 / +0,112 / +0,116 | **+0,1088** | [+0,103, +0,115] |

En PCK@0.2: pruned_ft → distilled +0,0029, IC [−0,0001, +0,0058] (no excluye 0);
distilled → int8 +0,0000.

**Compresión del entregable (int8 vs base):** params −42 % (3,30 M → 1,91 M), tamaño
−65 % (6,83 → 2,39 MB). Latencia fp32 de escritorio −30 % (44,3 → 30,9 ms) entre el
`base` y el modelo podado+distilado.

**Epochs efectivos de la distillation por fold:** 50 / 80 / 80 / 80 / 80. En el fold 0
cortó el early stopping; en los demás la métrica de validación siguió mejorando
hasta el final.

### Cómo leer estos números en el informe

1. **El entregable** (yolov8n podado 30 % + FT + KD + INT8) logra **OKS 0,964 ± 0,006**
   y **PCK@0.2 0,998 ± 0,001**, con 1,91 M de parámetros y 2,39 MB. El `base` COCO sin
   tocar logra 0,855 sobre los mismos frames.
2. **Casi toda la ganancia viene del fine-tune sobre UCO** (+0,10 OKS). Ese salto mezcla
   tres cosas: la recuperación del daño de la poda (que deja OKS 0), la adaptación de
   dominio (el `base` es un modelo COCO y el `pruned_ft` es el primero entrenado en
   UCO) y el entrenamiento en sí. Para separarlas hace falta el baseline de la Fase 1
   entrenado en UCO sin podar.
3. **La etapa de distillation suma +0,011 OKS de forma consistente** (el IC 95 % excluye
   el 0), pero no se puede atribuir a la KD en sí. En la misma etapa se suman la
   recuperación incompleta del FT de 10 epochs, 50–80 epochs más de entrenamiento con
   otra augmentation (mosaic) y un término de imitación que pesa sólo ~2–3 % de la
   loss. En PCK la mejora no es significativa. El control que falta es la misma
   etapa con α = 1 (sin imitación).
4. **El INT8 cuesta −0,0036 OKS**, con una variación mínima entre folds, y no cambia el
   PCK. Es dynamic-range (pesos INT8, activaciones float32), no full-integer.
5. **Advertencias que hay que declarar:**
   - el "GT" son pseudo-labels ViTPose-H, no anotación humana;
   - el `best.pt` del FT y de la KD se elige sobre el mismo `val` en el que se reporta:
     es un sesgo optimista, chico dadas las curvas planas, en `pruned_ft`,
     `distilled` e `int8`;
   - hubo 8 reanudaciones: pesos, optimizer y EMA se restauran exactos, no así el RNG
     de augmentation ni el contador del early stopping;
   - la latencia es de una CPU de escritorio.

## Qué se corre

`src.training.combined` con `configs/combined_yolov8n.yaml`: **GroupKFold k=5 por
sujeto** sobre `uco_pseudo` (labels densas ViTPose-H), un run MLflow padre en el
experimento **`fase5-combined`** y un run anidado por fold. Cada fold parte del
`models/yolov8n-pose.pt` COCO y produce su propio `.tflite` INT8; las 5 etapas se
evalúan sobre el split val **completo** del fold con el evaluador unificado
(`pose_eval`, mismo bbox normalizador en todas las etapas).

| Etapa | Qué mide | Modelo evaluado |
|---|---|---|
| `base` | punto de partida (COCO, sin tocar) | `yolov8n-pose.pt` |
| `pruned` | **daño** de la poda estructurada (sparsity 0.3) | podado, sin entrenar |
| `pruned_ft` | **recuperación** tras el fine-tune post-poda | podado + 10 epochs |
| `distilled` | aporte de la distillation (maestro yolov8x-pose) | podado + FT + 80 epochs KD |
| `int8` | costo de cuantizar el entregable | `best_int8.tflite` |

**Receta (la de la config de CV, sin cambios de modelo):** estudiante yolov8n-pose,
maestro yolov8x-pose, imgsz 640, batch 24, AdamW. Fine-tune post-poda 10 epochs a
`lr0/10 = 0.001`. Distillation 80 epochs, `L = 0.5·L_task + 0.5·(β/2)·B·L_AT` con
β = 1000, LR 0.001, `warmup_bias_lr 0`, **validación por epoch y early stopping con
patience 15**. INT8: la exportación de Ultralytics (calibración acotada a ≤300 imágenes).

| Fold | Train | Val |
|---|---|---|
| 0 | 57.805 | 16.150 |
| 1 | 56.671 | 17.284 |
| 2 | 62.041 | 11.914 |
| 3 | 59.317 | 14.638 |
| 4 | 59.986 | 13.969 |

**Costo estimado** (tiempos medidos en esta RTX 3090 en las corridas del 30-09): fine-tune
~5,4 min/epoch, distillation ~11,4 min/epoch + ~1,3 min de validación por epoch,
evaluación `.pt` en GPU ~3,5 min por etapa, evaluación INT8 en CPU ~53 min. Por fold
≈ 1 h (FT) + hasta ~17 h (KD; menos si corta el early stopping) + ~1,5 h de
evaluaciones/exports ≈ **~19 h por fold, ~4 días en total**.

## Cómo lanzar y cómo REANUDAR tras un corte de luz

Desde la raíz del repo, en cualquier terminal:

```
venv-torch\Scripts\python.exe scripts\run_fase5_cv.py
```

Ese mismo comando **arranca o reanuda**: levanta MLflow si hace falta, muestra en qué
checkpoint quedó la corrida y relanza. No hay que borrar nada ni editar la config. Si
la config cambia (salvo `workers`), aborta a propósito: mezclar folds entrenados con
recetas distintas en una misma media ± σ invalidaría el resultado sin dejar rastro.

Qué se conserva en un corte, de grueso a fino (todo bajo `runs/fase5_cv/`):

1. **Folds terminados**: no se repiten; sus métricas entran igual a la media ± σ del
   mismo run padre MLflow (se reabre por `run_id`, no se crea otro).
2. **Etapas terminadas** de un fold a medias: se saltean; la siguiente arranca del
   artefacto en disco (`student_pruned.pt`, `student_pruned_ft.pt`, `best.pt` de la
   distillation). El contexto de evaluación (GT + bbox normalizador) también se
   persiste (`eval_ctx.pkl`), así el bbox no se recalcula y las 5 etapas siguen siendo
   comparables.
3. **Entrenamiento en curso** (fine-tune o distillation): se reanuda desde el último
   epoch guardado (`weights/last.pt`) con optimizer, EMA, scaler y schedule de LR. Se
   pierde como mucho el epoch en curso (~12 min en la distillation).

Lo que **no** se conserva: una evaluación a medias (la INT8 son ~53 min, se repite
entera) y un export a medias. Si el corte cae justo mientras se escribe un `last.pt`,
el pipeline aborta con un mensaje que indica qué directorio borrar para reiniciar sólo
esa etapa.

## Avance (checkpoints)

<!-- CHECKPOINTS:BEGIN (generado por src/utils/checkpoints.py, no editar) -->
_Actualizado: 2026-10-14T23:49:01 — `runs/fase5_cv`_

Cada celda: ✅ etapa terminada y logueada (con su OKS), ⏳ entrenando (último epoch guardado en `last.pt`), · pendiente.

| Fold | Run MLflow | base | pruned | pruned_ft | distilled | int8 | Fin del fold |
|---|---|---|---|---|---|---|---|
| 0 | `ee2f122c` | ✅ 0.8459 | ✅ 0.0000 | ✅ 0.9431 | ✅ 0.9561 | ✅ 0.9525 | 2026-10-09T12:44:28 |
| 1 | `c06817a6` | ✅ 0.8567 | ✅ 0.0000 | ✅ 0.9564 | ✅ 0.9672 | ✅ 0.9640 | 2026-10-11T00:51:31 |
| 2 | `f12d16d3` | ✅ 0.8681 | ✅ 0.0000 | ✅ 0.9617 | ✅ 0.9745 | ✅ 0.9710 | 2026-10-12T09:50:33 |
| 3 | `cec8614d` | ✅ 0.8536 | ✅ 0.0000 | ✅ 0.9576 | ✅ 0.9686 | ✅ 0.9652 | 2026-10-13T20:10:26 |
| 4 | `9c0f3ea0` | ✅ 0.8514 | ✅ 0.0000 | ✅ 0.9656 | ✅ 0.9717 | ✅ 0.9674 | 2026-10-14T23:49:01 |

**Media ± σ por etapa sobre los folds terminados** (n = folds con esa métrica):

| Etapa | OKS | PCK@0.2 | PCK pierna | PCK brazo | Params | Latencia TFLite (ms) | Tamaño (MB) |
|---|---|---|---|---|---|---|---|
| base | 0.8552 ± 0.0074 (n=5) | 0.9810 ± 0.0019 (n=5) | 0.9769 ± 0.0025 (n=5) | 0.9967 ± 0.0008 (n=5) | 3295470 ± 0 (n=5) | 44.3 ± 3.2 (n=5) | 6.83 ± 0.00 (n=5) |
| pruned | 0.0000 ± 0.0000 (n=5) | 0.0000 ± 0.0000 (n=5) | 0.0000 ± 0.0000 (n=5) | 0.0000 ± 0.0000 (n=5) | 1907230 ± 0 (n=5) | 31.5 ± 2.5 (n=5) | — |
| pruned_ft | 0.9569 ± 0.0076 (n=5) | 0.9951 ± 0.0028 (n=5) | 0.9942 ± 0.0036 (n=5) | 0.9982 ± 0.0005 (n=5) | 1907230 ± 0 (n=5) | — | — |
| distilled | 0.9676 ± 0.0063 (n=5) | 0.9979 ± 0.0011 (n=5) | 0.9978 ± 0.0015 (n=5) | 0.9983 ± 0.0005 (n=5) | 1907230 ± 0 (n=5) | 30.9 ± 0.4 (n=5) | 4.07 ± 0.00 (n=5) |
| int8 | 0.9640 ± 0.0062 (n=5) | 0.9980 ± 0.0011 (n=5) | 0.9978 ± 0.0015 (n=5) | 0.9983 ± 0.0005 (n=5) | — | 187.0 ± 2.4 (n=5) | 2.39 ± 0.00 (n=5) |
<!-- CHECKPOINTS:END -->

## Hallazgos

### Antes de lanzar (2026-10-08)

1. **El pipeline no se podía reanudar.** `_kfold.run_phase` no guardaba estado: un
   corte a mitad de la CV perdía el fold en curso y cerraba el run padre, sin forma de
   completar la media ± σ sobre los 5 folds. Con ~19 h por fold esto hacía la corrida
   inviable con cortes de luz. Se agregaron checkpoints en tres niveles (fold, etapa,
   epoch), opt-in con `checkpoint_dir` en la config (`src/utils/checkpoints.py`).
2. **La etapa `base` borraba el INT8 de control en cada fold.** Ultralytics exporta
   junto al `.pt` de origen y arranca con un `rmtree` de `*_saved_model/`, así que medir
   la latencia de `models/yolov8n-pose.pt` borraba `models/yolov8n-pose_saved_model/`
   (el `yolov8n-pose_int8.tflite` que usan la app y `diagnose_kpt_ratio.py`). Estaba
   documentado como "hacer backup antes de correr"; ahora `_tflite_latency_from_pt`
   exporta una copia en un directorio temporal y el original no se toca. MD5 del
   control antes de lanzar: `ee27dcf9f0ad0c25fcf37868e097ad68` (el correcto).
3. **`workers: 8` en la config de CV volteaba la corrida en Windows** (17 procesos de
   dataloader agotaron el archivo de paginación, WinError 1455, ya visto el 30-09). Se
   bajó a 2, igual que en el config de deploy. Sólo afecta la velocidad de carga.
4. **La receta de esta CV NO es la del modelo desplegado.** `combined_yolov8n_deploy.yaml`
   decía ser "byte-idéntico salvo la CV", pero difiere en tres claves: fine-tune 25 vs
   10 epochs, distillation 20 vs 80 epochs, `val: false` vs validación por epoch. Para
   el informe: estos números describen la receta de `combined_yolov8n.yaml` y no se
   pueden presentar como la media ± σ del artefacto de `fase5-deploy` (run `42c1c36d`).
5. **El run padre sólo agregaba la etapa final.** `_log_aggregate` promedia las
   `ModelMetrics` que devuelve cada fold (las de `int8`). Las métricas por etapa
   (`oks_pruned_ft`, `latency_ms_base`, …), que son la tabla comparativa del informe,
   quedaban sólo en los runs anidados. Ahora el padre también recibe
   `{métrica}_mean` / `_std` / `_n_folds` de cada métrica de cada etapa, y la tabla de
   arriba muestra la media ± σ parcial a medida que terminan los folds.
6. **Reanudar no es bit-idéntico a correr de corrido**, y hay que declararlo si el
   informe dice que hubo interrupciones. Pesos, EMA, optimizer, scaler, epoch y LR se
   restauran exactos (verificado: diferencia máxima 0,0 entre el modelo reanudado y el
   EMA de `last.pt`). Lo que no se restaura: el estado del RNG de augmentation y
   shuffle (Ultralytics re-siembra al arrancar) y el contador del early stopping
   (`patience` vuelve a contar desde el epoch reanudado). El efecto esperado es del
   orden de la variabilidad entre semillas, no un sesgo.
7. **La primera versión de la reanudación fallaba con validación por epoch, y la prueba
   con cortes reales lo detectó.** El validador de Ultralytics cachea la loss
   (`criterion`) en el módulo EMA, y ese módulo se guarda en `last.pt`. Ultralytics
   carga los checkpoints en CPU, así que el módulo reanudado traía una `criterion` con
   tensores en CPU mientras el modelo pasaba a CUDA. Reventaba en el primer batch
   (`Expected all tensors to be on the same device`). La reanudación estándar de
   Ultralytics no lo sufre porque construye un modelo nuevo; nuestros trainers
   devuelven el módulo podado tal cual. Corregido: los dos trainers descartan la
   `criterion` cacheada en `get_model`. Eso además evita arrastrar los
   hiperparámetros de la corrida anterior, que la `criterion` guarda.
8. **Reanudar con el `last.pt` estándar de Ultralytics hacía diverger el fine-tune a
   NaN.** Es una trampa de Ultralytics que afecta cualquier reanudación con AdamW, no
   sólo este pipeline, y sin la prueba de cortes reales habría arruinado un fold
   entero en silencio: los runs siguientes heredan el modelo roto. `save_model` guarda
   el estado del optimizer convertido a fp16, y el segundo momento de AdamW
   (`exp_avg_sq` ≈ g², del orden de 1e-9) cae bajo el mínimo de fp16 y queda en 0. Al
   reanudar, el denominador de Adam colapsa y los pasos se multiplican. Con el
   `warmup_bias_lr` 0,1 del fine-tune, diverge en ~5 batches. Experimento A/B con el
   mismo seed y el mismo epoch 1:

   | `last.pt` | `exp_avg_sq` en 0 | box loss ep 1 → 2 → 3 |
   |---|---|---|
   | como lo guarda Ultralytics (fp16) | 887.112 / 1.907.214 (46,5 %) | 3,39 → NaN → NaN |
   | momentos en fp32 | 0 | 3,39 → 2,58 → 2,30 |

   Un fold sin cortes da 3,33 → 2,43 → 2,14, así que con fp32 la curva reanudada es
   sana. Corregido con `Fp32OptimizerCheckpointMixin` en los dos trainers custom
   (`src/utils/checkpoints.py`). Cuesta ~8 MB más por `last.pt`.

### Contexto metodológico para el informe (ya conocido, relevante para leer estos números)

- **El "GT" de la evaluación son las pseudo-labels ViTPose-H**, no anotación humana: el
  OKS mide el acuerdo con el maestro de etiquetado sobre UCO.
- **`base` es un modelo COCO y `pruned_ft` es el primero entrenado en UCO**, así que el
  salto `base → pruned_ft` mezcla daño de poda, recuperación y adaptación de dominio.
- **El `_int8.tflite` es dynamic-range** (pesos INT8, activaciones e I/O float32): la
  calibración no influye en los bytes entregados. La latencia INT8 medida en x86 no
  representa un teléfono (sale más lenta que fp32 en CPU de escritorio).
- **Pruning a sparsity 0.3 lleva el OKS a ~0** (verificado, no es un bug): la etapa
  `pruned` mide un modelo que dejó de detectar personas.
- **El checkpoint de cada etapa entrenada se elige sobre el mismo split que después
  se reporta.** El fine-tune y la distillation validan por epoch sobre el `val` del
  fold, y Ultralytics se queda con el `best.pt` de mayor fitness (y corta por early
  stopping) según ese mismo `val`. Es la misma partición sobre la que `pose_eval`
  calcula OKS/PCK, así que hay un sesgo optimista por selección de modelo en
  `pruned_ft` y `distilled`. `int8` hereda el de `distilled`; `base` y `pruned` no lo
  tienen. Con curvas tan planas como las observadas (en el fold 0, mAP50-95 entre
  0,9040 y 0,9045 durante 8 epochs) debería ser chico, pero hay que declararlo. La
  alternativa limpia sería un split de validación interno separado del fold.

## Registro de eventos

- **2026-10-08** — Implementados y probados los checkpoints reanudables (ver
  "Verificación de la reanudación" abajo).
- **2026-10-08 12:03** — **Lanzada la CV** con `scripts\run_fase5_cv.cmd`, abierto vía
  `explorer.exe` para que no dependa de la sesión de VSCode/Claude Code (corre en su
  propia ventana de consola). Experimento MLflow `fase5-combined` (nuevo, sin runs
  previos), run padre **`11734b62e1504b15a146dacd944f96a5`**; fold 0 =
  `ee2f122c2c014b25bc4d5a29baa1e83c`. Huella de la config `3aba772da4400779`. Código:
  rama `RunCrossValidation` sobre `db95489` + los cambios de esta sesión, sin commitear
  (no cambiar de rama mientras corre: el proceso relanzado tomaría otro código).
- **2026-10-08 12:07 — fold 0, `base`**: OKS 0,8459, PCK 0,9813 (pierna 0,9771, brazo
  0,9952) sobre 16.149 frames (1 de 16.150 sin bbox de persona base, excluido). Coincide
  con el 0,846 medido antes sobre este split: el evaluador es consistente. Latencia
  TFLite fp32 50,6 ms, contra 43,9 ms en el run de deploy del 30-09: la latencia en
  CPU de escritorio varía entre corridas, así que conviene compararla sólo entre etapas
  del mismo fold y no como número absoluto.
- **2026-10-08 12:09 — fold 0, `pruned`**: OKS ≈ 0, PCK 0 (colapso esperado de la poda
  a sparsity 0,3), 1.907.230 params, 36,6 ms.
- **2026-10-08 12:16 — fold 0, fine-tune epoch 1/10 guardado**: 5,9 min/epoch. Losses
  finitas (box 0,889, pose 3,30) y mAP50-95 de pose de Ultralytics 0,687 ya en el
  epoch 1. Verificado sobre el `last.pt` real: optimizer en fp32 con 0 momentos en
  cero, EMA con la arquitectura podada (1.907.230 params). La reanudación queda
  operativa desde este punto.
- **2026-10-08 13:02 — fold 0, fine-tune 10/10 terminado** (52 min, ~5,2 min/epoch).
  Curva sana y monótona: box loss 0,889 → 0,291, pose loss 3,30 → 0,551; mAP50-95 de
  pose de Ultralytics 0,687 → 0,869 sin meseta todavía. Como referencia, el fine-tune de
  25 epochs del run de deploy llegó a 0,897, así que con 10 epochs la recuperación queda
  incompleta y la distillation va a seguir recuperando. Es exactamente la mezcla de
  efectos (recuperación + KD) que hay que tener presente al leer el delta
  `pruned_ft → distilled` en el informe.
- **2026-10-08 ~13:04 — INTERRUPCIÓN #1** durante la evaluación `pruned_ft` del fold 0
  (el fine-tune ya había terminado a las 13:02 y `student_pruned_ft.pt` estaba
  guardado a las 13:03:31). Murieron juntos la ventana de consola (`cmd.exe`), el
  launcher, el entrenamiento y el servidor MLflow, sin traceback ni línea de cierre del
  launcher. Tampoco hay eventos de crash en el visor de eventos de Windows ni falta de
  memoria (18 GB libres). Lo más probable es que se haya cerrado la ventana de consola.
  Estado al reanudar: el fine-tune figura terminado (`last.pt` sin optimizer), así que
  no se reentrena; sólo se repite la evaluación de `pruned_ft` (~4 min).
- **2026-10-08 13:10 — reanudada** con el mismo comando. Funcionó como se diseñó: reabrió
  la misma CV y los mismos runs MLflow (sigue habiendo un único padre `11734b62` y un
  único fold 0 `ee2f122c`, sin duplicados) y reusó el contexto de evaluación. Además
  salteó `base`, `pruned` y el entrenamiento del fine-tune. Se perdieron ~2 min de
  cómputo.
- **2026-10-08 13:12 — fold 0, `pruned_ft`**: OKS **0,9431**, PCK 0,9904 (pierna
  0,9881, brazo 0,9979), 1.907.230 params (sin pérdida de BatchNorm). Con sólo 10 epochs
  de fine-tune, el podado ya supera al `base` COCO (0,846) y al maestro yolov8x-pose
  medido antes sobre UCO (0,911); el fine-tune de 25 epochs del deploy había dado
  0,9535. Recordar que este salto incluye la adaptación de dominio a UCO (el `base` es
  COCO). Sigue la distillation: 80 epochs, ~17 h.
- **2026-10-08 13:27 — fold 0, distillation epoch 1/80**: 13,2 min/epoch con la
  validación incluida (80 epochs ≈ 17,6 h si no corta el early stopping). Losses
  finitas. Dos observaciones para leer después la etapa `distilled`:
  - **El término de imitación pesa poco después del arranque.** Su peso en la loss
    cae de 11,6 % (step 0) a ~3,5 % (step 2.500), porque `L_AT` baja de 1,75e-3 a
    2,3e-4: el estudiante alinea rápido sus mapas de atención con los del maestro.
    Desde ahí ~96 % del gradiente es la task loss, así que la etapa se comporta
    mayormente como un re-entrenamiento supervisado con un pequeño término de
    imitación.
  - **El fine-tune y la distillation no usan la misma augmentation.** El fine-tune
    (10 epochs, `close_mosaic: 10`) corrió **sin mosaic** desde el epoch 0: el log
    dice "Closing dataloader mosaic" al arrancar. La distillation (80 epochs) usa
    mosaic hasta el epoch 70. Tras el primer epoch de KD, el mAP50-95 de pose de
    Ultralytics en val bajó de 0,869 (fin del FT) a 0,827, consistente con el cambio
    de augmentation y el reinicio del warmup. El delta `pruned_ft → distilled` del
    informe mezcla entonces KD, más epochs y otra augmentation.
- **2026-10-08 15:23 — fold 0, distillation epoch 10/80**: la caída inicial se recuperó.
  El mAP50-95 de pose en val fue 0,827 → 0,815 (ep 2) → 0,889 (ep 10), ya por encima
  del 0,869 del final del fine-tune, y sigue subiendo. Pose loss de train 1,87 → 0,90,
  sin NaN; ~12,9 min/epoch. La imitación sigue bajando (2–3 % de la loss).
- **2026-10-08 17:31 — fold 0, distillation epoch 20/80**: mAP50-95 de pose en val
  0,889 → **0,901**, mejora monótona pero aplanándose (~+0,001/epoch en los últimos 5).
  El run de deploy (25 FT + 20 KD) terminó en 0,897 con esta misma métrica. Como la
  métrica sigue mejorando de a poco, el early stopping (patience 15) no va a cortar
  pronto: hay que contar con los 80 epochs completos.
- **2026-10-08 20:28 — PAUSA #2 (planificada, a pedido)**: detenida justo después de
  guardar el epoch 34/80 de la distillation del fold 0. Se mató el árbol de la ventana
  y el servidor MLflow 6 s después del checkpoint. Verificado: `last.pt` es el epoch 34
  (índice 33), con optimizer en fp32. El mAP50-95 de pose en val iba en 0,9045. No se
  pierde nada: al reanudar sigue en el epoch 35. Restan 46 epochs (~10 h) + la
  evaluación de `distilled` e `int8` del fold 0, y después los folds 1–4.
- **2026-10-09 08:07 — reanudada** (mismo comando). Retomó en "epoch 35 to 80" con el
  mismo run padre y fold 0, y reusó el contexto de evaluación. El primer paso de KD
  tras la reanudación está en línea con el último antes de la pausa: task term 31,5 vs
  36,7; imitación 2,5 % vs 2,1 %. No hay salto ni NaN.
- **2026-10-09 09:24 — fold 0, distillation epoch 40/80: meseta, se viene el early
  stopping.** La fitness de Ultralytics 8.3.253 (mAP50-95 de box + mAP50-95 de pose,
  `Metric.fitness` con pesos `[0, 0, 0, 1]`) tocó su máximo en el **epoch 33** (1,86411;
  mAP50-95 de pose 0,9044) y desde ahí baja levemente. Con `patience: 15`, una corrida de corrido
  cortaría tras el epoch 48. **Efecto de la pausa:** `EarlyStopping` no se guarda en
  `last.pt` (Ultralytics lo recrea al reanudar), así que su contador arrancó de nuevo
  en el epoch 35 y el corte llegará recién tras el ~50. No cambia el modelo evaluado:
  la etapa `distilled` usa `best.pt`, y el mejor fitness sí se restaura desde el
  checkpoint, así que `best.pt` sigue siendo el del epoch 33 salvo que otro lo supere.
  Sólo cuesta ~2 epochs (~26 min) de más. Nueva estimación para cerrar el fold 0: KD
  hasta ~11:30, evaluaciones ~1 h, **fold 0 completo ~12:45** (no ~19:00).
- **2026-10-09 11:26 — fold 0, distillation cortada por early stopping tras el epoch
  50/80**, como se había previsto: 50 epochs en vez de 80, y 16 epochs desde la
  reanudación en 3,28 h. Ultralytics informa "best results observed at epoch 35"
  porque su contador se reinició en la reanudación. Verificado que el `best.pt`
  evaluado es el del **epoch 33**: la fitness guardada en el checkpoint (1,8641)
  coincide con la de ese epoch. Una corrida sin pausas habría cortado tras el epoch 48
  con el mismo `best.pt`: la pausa costó 2 epochs y no cambió el modelo. Sigue la
  evaluación de `distilled` y la etapa `int8`.
- **2026-10-09 11:29 — fold 0, `distilled`**: OKS **0,9561**, PCK 0,9958 (pierna
  0,9951, brazo 0,9983), 1.907.230 params, 4,07 MB, latencia TFLite fp32 31,6 ms (36,6
  en `pruned`, en la misma máquina pero en otro momento). Contra `pruned_ft` (0,9431)
  es **+0,013 OKS**. Con esta receta la etapa sí suma, a diferencia del deploy (FT 25
  + KD 20), donde fue 0,9535 → 0,9533. El valor final del fold 0 (0,9561) queda
  incluso apenas por encima del `distilled` del deploy. Pero, según los hallazgos de
  más arriba, este +0,013 no se puede atribuir a la distillation: el FT de 10 epochs
  dejó la recuperación incompleta, la KD aporta ~2–3 % del gradiente y cambia la
  augmentation (mosaic). Lo más probable es que sea sobre todo entrenamiento
  supervisado adicional. Para aislar el efecto de la KD haría falta un control con
  α = 1 (mismo entrenamiento, sin término de imitación).
- **2026-10-09 12:44 — FOLD 0 TERMINADO** (`int8`: OKS **0,9525**, PCK 0,9958, 2,39 MB).
  Resumen del fold 0 (16.149 frames de val):

  | Etapa | OKS | PCK@0.2 | Params | Tamaño | Latencia TFLite |
  |---|---|---|---|---|---|
  | base | 0,8459 | 0,9813 | 3.295.470 | 6,83 MB | 50,6 ms (fp32) |
  | pruned | ≈ 0 | 0 | 1.907.230 | — | 36,6 ms (fp32) |
  | pruned_ft | 0,9431 | 0,9904 | 1.907.230 | — | — |
  | distilled | 0,9561 | 0,9958 | 1.907.230 | 4,07 MB | 31,6 ms (fp32) |
  | int8 | 0,9525 | 0,9958 | — | 2,39 MB | 191 ms (x86, no representativa) |

  - **Costo del INT8: −0,0036 OKS** con PCK idéntico, y −65 % de tamaño respecto del
    `base` (−41 % respecto del `distilled`). Es del mismo orden que en el deploy (−0,004).
  - Tiempo de cómputo efectivo del fold: ~12 h 55 min (12:03–13:04, 13:10–20:28 y
    08:07–12:44). El early stopping cortó la distillation en el epoch 50, y el export
    INT8 tardó 18 min y la evaluación INT8 ~56 min.
  - El MD5 del INT8 de control sigue intacto (`ee27dcf9…`).
  - **Fold 1 arrancó a las 12:44:28.** Es el fold de val más grande (17.284 frames),
    así que sus evaluaciones son ~7 % más largas. Si los folds 1–4 cortan por early
    stopping de forma parecida (~13 h c/u), la CV termina en ~52 h de cómputo: el
    **domingo 11 por la tarde** sin pausas.
- **2026-10-09 12:46 — PAUSA #3 (a pedido, descanso de la PC).** Detenida ~2 min
  después de arrancar el fold 1, mientras armaba su contexto de evaluación (todavía
  sin nada guardado: se pierden sólo esos ~2 min). Se cortó el árbol de la ventana
  y el servidor MLflow. Estado: fold 0 **terminado**; fold 1 con run MLflow
  `c06817a6` ya creado, que se va a reabrir al reanudar. Para seguir: doble clic en
  `scripts\run_fase5_cv.cmd`. Arranca el fold 1 desde el principio de su primera
  etapa y saltea el fold 0.
- **2026-10-09 15:24 — reanudada.** Salteó el fold 0 ("ya terminado") y arrancó el
  fold 1 reabriendo su run `c06817a6`. MLflow sigue con exactamente 3 runs: el padre
  `11734b62` (RUNNING), el fold 0 (FINISHED) y el fold 1. La huella de config y el MD5
  del control están intactos.
- **2026-10-09 16:27 — fold 1 hasta `pruned_ft`** (17.284 frames de val). `base` OKS
  0,8567 / PCK 0,9786; `pruned` ≈ 0; `pruned_ft` **0,9564** / PCK 0,9940, tras 10
  epochs de FT (~5,6 min/epoch, mAP50-95 de pose de Ultralytics 0,905). Este fold
  recupera bastante más que el fold 0 (0,9431) con el mismo entrenamiento: la
  variabilidad entre folds (sujetos distintos) de `pruned_ft` es del orden de 0,01
  OKS, mayor que el costo del INT8. Sigue la distillation.
- **2026-10-09 18:22 — PAUSA #4 (a pedido)**: fold 1 detenido 7 s después de guardar
  el epoch 9/80 de la distillation. Verificado: `last.pt` = epoch 9, optimizer en fp32;
  se cortó el árbol de la ventana y el servidor MLflow. Curva del fold 1 hasta acá:
  mAP50-95 de pose 0,844 (ep 1) → 0,920 (ep 9), ya por encima del 0,905 de su FT, con
  el mismo patrón de caída y recuperación que el fold 0. Al reanudar sigue en el
  epoch 10. Como en el fold 0, el contador del early stopping arrancará de nuevo
  (ver la entrada de las 09:24).
- **2026-10-10 08:41 — reanudada.** Salteó el fold 0, reusó el contexto de evaluación
  del fold 1 y retomó la distillation "from epoch 10 to 80". El primer paso de KD es
  sano (task term 29,7; imitación 2,9 %), con la GPU al 87 %.
- **2026-10-10 17:09 — fold 1, distillation epoch 50/80: este fold no se estanca.** A
  diferencia del fold 0, que tocó su máximo en el epoch 33 y cortó en el 50, el fold 1
  sigue marcando máximos de a poco: fitness 1,8856 (ep 20) → 1,8873 (ep 28) → 1,8903
  (ep 39) → **1,8915 (ep 50)**, con mAP50-95 de pose 0,9351. Es coherente con el LR
  que decae linealmente hasta el epoch 80. Lo más probable es que corra los 80 epochs
  (~6 h más) y que el fold 1 termine cerca de la medianoche. Para el informe: la
  cantidad de epochs efectivos de la distillation varía entre folds (early stopping),
  así que conviene reportarla por fold junto con las métricas.
- **2026-10-10 23:33 — fold 1, `distilled`**: corrió los **80 epochs completos** sin
  que cortara el early stopping. Mejor fitness en el epoch 79-80; mAP50-95 de pose de
  Ultralytics 0,9396; desde el epoch 71 sin mosaic (`close_mosaic`), lo que se ve como
  una caída abrupta de la loss de train. Resultado: OKS **0,9672**, PCK 0,9984 (pierna
  0,9984, brazo 0,9983) sobre 17.277 frames (7 de 17.284 sin bbox base), 1.907.230
  params, 4,07 MB, 30,5 ms fp32. Contra su `pruned_ft` (0,9564): **+0,0108 OKS**, el
  mismo orden que el fold 0 (+0,0130), con 80 epochs acá y 50 allá. Sigue la etapa
  `int8`.
- **2026-10-11 00:51 — FOLD 1 TERMINADO** (`int8`: OKS **0,9640**, PCK 0,9985, 2,39 MB).
  Costo del INT8: **−0,0033 OKS** (fold 0: −0,0036), con PCK prácticamente idéntico.
  Cómputo efectivo del fold: ~19 h (la distillation corrió los 80 epochs). El MD5 del
  control sigue intacto.

  **Parcial con 2 de 5 folds** (OKS, media ± σ poblacional):

  | Etapa | Fold 0 | Fold 1 | Media ± σ |
  |---|---|---|---|
  | base | 0,8459 | 0,8567 | 0,8513 ± 0,0054 |
  | pruned | 0,0000 | 0,0000 | 0,0000 ± 0,0000 |
  | pruned_ft | 0,9431 | 0,9564 | 0,9498 ± 0,0067 |
  | distilled | 0,9561 | 0,9672 | 0,9617 ± 0,0055 |
  | int8 | 0,9525 | 0,9640 | 0,9582 ± 0,0057 |

  El orden de las etapas se repite idéntico en los dos folds (pruned ≪ base <
  pruned_ft < int8 < distilled), y el desvío entre folds (~0,006) es mayor que el costo
  del INT8 (~0,0035), pero menor que el aporte de la etapa de distillation (~0,012).
- **Fold 2 arrancó a las 00:51:32**: train 62.041 (el más grande, epochs ~7 % más
  largos), val 11.914 (el más chico, evaluaciones más cortas). Quedan los folds 2–4: entre ~13 h
  (si corta el early stopping como en el fold 0) y ~19 h (80 epochs, como en el fold 1)
  cada uno, es decir 39–57 h de cómputo.
- **2026-10-11 01:56 — fold 2 hasta `pruned_ft`** (11.914 frames de val): `base` OKS
  0,8681 / PCK 0,9841; `pruned` ≈ 0; `pruned_ft` **0,9617** / PCK 0,9951 (10 epochs de
  FT a ~6,3 min/epoch; mAP50-95 de pose de Ultralytics 0,77 en el ep 1, el arranque más
  alto de los tres folds). Es el `pruned_ft` más alto hasta ahora (fold 0: 0,9431; fold
  1: 0,9564). Sigue la distillation.
- **2026-10-11 19:34 — PAUSA #5 (a pedido)**: fold 2 detenido 6 s después de guardar el
  epoch **79/80** de la distillation. `last.pt` = epoch 79, optimizer en fp32; se cortó
  el árbol de la ventana y el servidor MLflow. Igual que el fold 1, el fold 2 no se
  estancó: el mAP50-95 de pose subió sin parar, de 0,852 (ep 1) a 0,951 (ep 73–79),
  así que va a completar los 80 epochs. Al reanudar: 1 epoch (~13 min), evaluación de
  `distilled`, export y evaluación INT8 (~1 h); después, folds 3 y 4.
- **2026-10-12 08:36 — reanudada.** Salteó los folds 0 y 1, reusó el contexto de
  evaluación del fold 2 y retomó "from epoch 80 to 80". El primer paso de KD es sano
  (task term 25,6; imitación 2,9 %), con la GPU al 95 %.
- **2026-10-12 08:53 — fold 2, `distilled`**: también corrió los 80 epochs completos,
  igual que el fold 1. OKS **0,9745**, PCK 0,9989 (pierna 0,9992, brazo 0,9976) sobre
  11.914 frames (todos con bbox base), 1.907.230 params, 4,07 MB, 30,9 ms fp32. Contra
  su `pruned_ft` (0,9617): **+0,0128 OKS**. El delta de la etapa es consistente en los
  tres folds (+0,0130 / +0,0108 / +0,0128), aunque los epochs efectivos variaron
  (50 / 80 / 80). Sigue la etapa `int8`.
- **2026-10-12 09:50 — FOLD 2 TERMINADO** (`int8`: OKS **0,9710**, PCK 0,9989, 2,39 MB).
  Costo del INT8: **−0,0035 OKS** (folds 0/1: −0,0036 / −0,0033). El MD5 del control
  sigue intacto. Arrancó el fold 3.

  **Parcial con 3 de 5 folds** (OKS, media ± σ poblacional):

  | Etapa | Fold 0 | Fold 1 | Fold 2 | Media ± σ |
  |---|---|---|---|---|
  | base | 0,8459 | 0,8567 | 0,8681 | 0,8569 ± 0,0091 |
  | pruned | 0,0000 | 0,0000 | 0,0000 | 0,0000 ± 0,0000 |
  | pruned_ft | 0,9431 | 0,9564 | 0,9617 | 0,9538 ± 0,0078 |
  | distilled | 0,9561 | 0,9672 | 0,9745 | 0,9660 ± 0,0075 |
  | int8 | 0,9525 | 0,9640 | 0,9710 | 0,9625 ± 0,0076 |

  En los tres folds el orden es el mismo y las diferencias entre etapas tienen un
  tamaño parecido. Que el ranking de folds se repita en todas las etapas
  (fold 0 < fold 1 < fold 2) indica que gran parte de la σ viene de la dificultad de
  los sujetos de cada split y no del entrenamiento: un análisis pareado por fold de
  las diferencias entre etapas va a ser bastante más preciso que comparar las medias
  sueltas.
- **2026-10-12 10:54 — fold 3 hasta `pruned_ft`** (14.638 frames de val): `base` OKS
  0,8536 / PCK 0,9814; `pruned` ≈ 0; `pruned_ft` **0,9576** / PCK 0,9980 (10 epochs de
  FT a ~5,4 min/epoch; mAP50-95 de pose de Ultralytics 0,724 en el ep 1). En el ranking
  de folds queda en el medio (0,943 / 0,956 / 0,962 / **0,958**). Sigue la distillation.
- **2026-10-12 18:24 — PAUSA #6 (a pedido)**: fold 3 detenido 6 s después de guardar
  el epoch **35/80** de la distillation. `last.pt` = epoch 35, optimizer en fp32; se
  cortó el árbol de la ventana y el servidor MLflow. La curva sigue el patrón de los
  folds 1 y 2: mAP50-95 de pose 0,868 (ep 1) → 0,940 (ep 32–35), con el mejor
  fitness en el epoch 35, todavía en subida. Al reanudar sigue en el epoch 36; el
  contador del early stopping arranca de nuevo.
- **2026-10-13 07:54 — reanudada.** Salteó los folds 0–2, reusó el contexto de
  evaluación del fold 3 y retomó "from epoch 36 to 80". El primer paso de KD es sano
  (task term 25,0; imitación 2,9 %), con la GPU al 100 %.
- **2026-10-13 10:37 — PAUSA #7 (a pedido, inmediata)**: detenida a mitad del epoch 48
  del fold 3, sin esperar el checkpoint. Se pierde sólo el avance de ese epoch, porque
  el último guardado es el epoch **47** (verificado: `last.pt` = epoch 47, optimizer en
  fp32). Se cortaron el árbol de la ventana y MLflow. La curva sigue subiendo de a
  poco: mejor fitness en el epoch 47 (mAP50-95 de pose 0,9409), así que la meseta que
  parecía haber en el epoch 40 no se confirmó. Al reanudar sigue en el epoch 48.
- **2026-10-13 11:46 — reanudada.** Retomó la distillation del fold 3 "from epoch 48 to
  80". El primer paso de KD es sano (task term 24,4; imitación 2,9 %), con la GPU al
  96 %.
- **2026-10-13 19:01 — fold 3, `distilled`**: 80 epochs completos, como los folds 1 y
  2 (mejor fitness al final; mAP50-95 de pose de Ultralytics 0,9421 en el epoch 79).
  OKS **0,9686**, PCK 0,9987 (pierna 0,9989, brazo 0,9981) sobre 14.614 frames (24 de
  14.638 sin bbox base), 1.907.230 params, 4,07 MB, 31,2 ms fp32. Contra su
  `pruned_ft` (0,9576): **+0,0110 OKS**. El delta de la etapa sigue estable en los
  cuatro folds: +0,0130 / +0,0108 / +0,0128 / +0,0110. Sigue la etapa `int8`.
- **2026-10-13 20:10 — FOLD 3 TERMINADO** (`int8`: OKS **0,9652**, PCK 0,9987, 2,39 MB).
  Costo del INT8: **−0,0035 OKS**, igual que en los demás folds (−0,0036 / −0,0033 /
  −0,0035). El MD5 del control sigue intacto. **Arrancó el fold 4 (el último) a las
  20:10:28.**

  **Parcial con 4 de 5 folds** (OKS, media ± σ poblacional):

  | Etapa | F0 | F1 | F2 | F3 | Media ± σ |
  |---|---|---|---|---|---|
  | base | 0,8459 | 0,8567 | 0,8681 | 0,8536 | 0,8561 ± 0,0080 |
  | pruned | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 ± 0,0000 |
  | pruned_ft | 0,9431 | 0,9564 | 0,9617 | 0,9576 | 0,9547 ± 0,0070 |
  | distilled | 0,9561 | 0,9672 | 0,9745 | 0,9686 | 0,9666 ± 0,0066 |
  | int8 | 0,9525 | 0,9640 | 0,9710 | 0,9652 | 0,9631 ± 0,0067 |

  Los deltas pareados por fold son muy estables: `pruned_ft → distilled` entre +0,011
  y +0,013, `distilled → int8` entre −0,0033 y −0,0036.
- **2026-10-13 21:14 — fold 4 hasta `pruned_ft`** (13.969 frames de val): `base` OKS
  0,8514 / PCK 0,9794; `pruned` ≈ 0; `pruned_ft` **0,9656** / PCK 0,9979 (10 epochs de
  FT a ~5,6 min/epoch). Es el `pruned_ft` más alto de los cinco folds (0,943 / 0,956 /
  0,962 / 0,958 / **0,966**). Sigue la distillation, la última del pipeline.
- **2026-10-13 23:54 — PAUSA #8 (a pedido)**: fold 4 detenido 7 s después de guardar
  el epoch **12/80** de la distillation (`last.pt` = epoch 12, optimizer en fp32); se
  cortaron el árbol de la ventana y MLflow. Curva hasta acá: mAP50-95 de pose 0,863
  (ep 1) → 0,927 (ep 12), con el mismo patrón que los otros folds. Al reanudar sigue
  en el epoch 13. Faltan la distillation del fold 4 (~68 epochs, ~15 h si corre
  completa), su `int8` (~1,2 h) y el cierre de la CV (agregados en el run padre).
- **2026-10-14 07:53 — reanudada.** Salteó los folds 0–3, reusó el contexto de
  evaluación del fold 4 y retomó "from epoch 13 to 80". El primer paso de KD es sano
  (task term 27,8; imitación 2,9 %), con la GPU al 87 %.
- **2026-10-14 22:42 — fold 4, `distilled`**: 80 epochs completos, como los folds 1–3
  (mejor fitness al final; mAP50-95 de pose de Ultralytics 0,9404). OKS **0,9717**, PCK
  0,9978 (pierna 0,9975, brazo 0,9991) sobre 13.968 frames (1 de 13.969 sin bbox base),
  1.907.230 params, 4,07 MB, 30,4 ms fp32. Contra su `pruned_ft` (0,9656): **+0,0061
  OKS**, el delta más chico de los cinco (resto: +0,011 a +0,013). Es coherente con que
  este fold ya partía del `pruned_ft` más alto: queda menos margen para mejorar. Sigue
  la etapa `int8`, la última de la CV.
- **2026-10-14 23:49 — FOLD 4 TERMINADO y CV COMPLETA** (`int8`: OKS **0,9674**, PCK
  0,9978; costo del INT8 −0,0043, el mayor de los cinco pero del mismo orden). El
  launcher cerró con `rc=0`. Verificado: run padre `11734b62` FINISHED con los
  agregados por etapa, 5 folds FINISHED, 6 runs en el experimento (ninguno huérfano),
  un `best_int8.tflite` por fold en MLflow, y MD5 del INT8 de control intacto
  (`ee27dcf9…`). Resultados y lectura para el informe en la sección "Resultados
  finales", al principio del documento.
- **2026-10-16 — compuerta fuera de dominio: los 5 modelos FALLAN.** El set original de
  79 imágenes se había perdido, así que se reconstruyó con criterios explícitos
  (`scripts/build_ood_gt_set.py`, IDs versionados en `data/ood_gate/`): 80 imágenes de
  COCO train2017 y 37 de val2017. Las referencias medidas de nuevo dan los mismos
  veredictos que en julio: v2 0,759 · 33/73 · error 0,393; v3 0,945 · 0/80 · 0,039;
  stock 0,984 · 0/80 · 0,017. **CV INT8 (5 folds):** persona detectada con conf ≥ 0,25
  en 13–33 de 80 imágenes (v3 78/80); error de punto 0,42 ± 0,09 (criterio ≤ 0,05); 38 %
  de las evaluables colapsadas. El `best.pt` fp32 también falla (no es la cuantización), y
  val2017 da lo mismo. El deploy de la Fase 5 (`42c1c36d`) también falla: razón 1,30,
  error 0,43. Detalle y fragmento LaTeX en `resultados/fase5_cv/gate_ood/`.

## Verificación de la reanudación (antes de lanzar)

- `tests/test_checkpoints.py` (7 tests): huella de config, estado por fold y etapa,
  reescritura de la tabla sólo entre marcadores, detección nuevo / a mitad / terminado /
  corrupto del `last.pt`.
- Chequeo enfocado de los trainers custom (poda y KD), con validación por epoch como
  en la config real: entrenar 3 epochs, abortar tras el epoch 1 y reanudar. En los
  dos: arranca en el epoch 2, conserva la arquitectura podada (1.907.230 params), los
  pesos son idénticos al EMA de `last.pt` (diferencia máxima 0,0) y el `results.csv`
  queda continuo `[1, 2, 3]`. En la KD, los hooks de attention transfer siguen activos
  después de reanudar (aparece el log `KD step 0` posterior al `Resuming training`).
- Prueba end-to-end del pipeline real (`src.training.combined`, misma config salvo
  tamaños): 2 folds mini, `taskkill /F /T` en mitad del fine-tune del fold 0 y en mitad
  de la distillation del fold 1, relanzando el mismo comando. Las dos primeras
  versiones destaparon los hallazgos 7 y 8. Con los dos fixes, la tercera pasa
  completa:
  - 1 run padre + 2 anidados en MLflow (reabiertos por `run_id`, sin runs huérfanos),
    los 3 en `FINISHED`; las 5 etapas logueadas en cada fold, y en el padre los
    agregados por etapa (`oks_pruned_ft_mean`, …) con `n_folds = 2`;
  - `results.csv` continuos `[1, 2, 3]` en los dos entrenamientos reanudados, sin
    NaN/Inf en ningún entrenamiento;
  - el fold con el fine-tune reanudado recupera igual que el otro (OKS `pruned_ft`
    0,19 vs 0,21; `distilled` 0,49 vs 0,53, con 3 epochs sobre 400 imágenes);
  - params 3.295.470 → 1.907.230 idénticos en `pruned`, `pruned_ft` y `distilled`
    (sin pérdida de BatchNorm);
  - MD5 del INT8 de control intacto y `models/` sin cambios;
  - la tabla de avance se regenera sin tocar el texto escrito a mano.
