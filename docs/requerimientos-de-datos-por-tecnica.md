# Requerimientos de datos por técnica de optimización

> **Propósito.** Este documento justifica una decisión de diseño central del
> pipeline: **por qué la recuperación de precisión tras la poda se hace con
> *knowledge distillation* y no con *fine-tuning* supervisado**. La razón de
> fondo es que no todas las técnicas de optimización necesitan un dataset
> etiquetado, y las que sí lo necesitan chocan con una limitación del dataset
> UCO (anotación parcial de 3 keypoints por ejercicio). Distillation permite
> recuperar precisión **usando solo imágenes**, evitando esa limitación.

---

## 1. Contexto

El objetivo de la tesis es un modelo de *pose estimation* **optimizado para
correr en celulares Android de gama media-baja** (offline, export TFLite INT8).
Para llegar a ese modelo se aplican tres técnicas de optimización en secuencia:

1. **Poda estructurada** (structured pruning) — elimina canales/filtros para
   reducir parámetros, FLOPs, tamaño y latencia.
2. **Knowledge distillation** — un modelo maestro (yolov8x-pose) supervisa al
   estudiante podado (yolov8n-pose) para recuperar la precisión perdida.
3. **Cuantización INT8** (post-training) — convierte pesos y activaciones de
   FP32 a INT8 para el deployment en el dispositivo.

La pregunta que este documento responde: **¿por qué distillation después de la
poda, en lugar de un fine-tuning clásico con el dataset?**

---

## 2. Tabla resumen: ¿qué técnica necesita ground-truth?

No todas las técnicas requieren un dataset **etiquetado** (con keypoints
ground-truth). Algunas necesitan solo **imágenes**, otras necesitan la señal de
un **modelo maestro**, y solo algunas necesitan las **etiquetas reales**.

| Técnica | ¿Necesita labels (ground-truth)? | Qué necesita en su lugar |
|---|---|---|
| **Cuantización INT8** | **No** | Un puñado de imágenes de *calibración* (sin etiquetas). Los frames de UCO alcanzan tal cual. |
| **Poda estructurada** | **No** (la poda en sí) | Nada: opera sobre los pesos del modelo. Recuperar la precisión perdida **sí** requiere una señal de entrenamiento (ver siguiente fila). |
| **Distillation (attention-transfer)** | **No** (el término de distillation) | Solo **imágenes** + el **modelo maestro**. La señal la da el maestro, no el ground-truth. |
| **Fine-tune con ground-truth** | **Sí** | Un dataset con keypoints etiquetados. Acá es donde entraría el etiquetado manual. |
| **Evaluación (medir la caída de precisión)** | **Sí, imprescindible** | Ground-truth para comparar. Sin él no se puede reportar cuánto se pierde al optimizar. |

---

## 3. Detalle técnico por técnica

### 3.1. Cuantización INT8 — no necesita labels

La cuantización *post-training* (PTQ) convierte los pesos de FP32 a INT8 y, en el
modo *full-integer*, también las activaciones. Para cuantizar las activaciones el
conversor necesita observar el **rango de valores** que toman en la práctica, y
para eso corre el modelo sobre un conjunto de **imágenes representativas**
(*representative / calibration dataset*).

Punto clave: esas imágenes se usan solo para **medir estadísticas de
activación**, no para calcular gradientes ni comparar contra un ground-truth. Por
eso **no hacen falta etiquetas**: cualquier conjunto de imágenes del dominio de
deployment sirve. Los frames de UCO son perfectamente válidos como set de
calibración tal como están.

> En el pipeline: `src/optimization/quantization.py` construye el
> `representative_dataset_fn` a partir de un directorio de imágenes; la export de
> Ultralytics (`export(int8=True, data=...)`) calibra internamente con el dataset.

### 3.2. Poda estructurada — no necesita labels, pero degrada la precisión

La poda elimina canales completos siguiendo el grafo de dependencias de la red
(DepGraph). Es una operación puramente sobre los **pesos**: no requiere datos ni
etiquetas para ejecutarse.

El problema es su **consecuencia**: al remover canales se pierde capacidad
representacional y la precisión **cae**. Para recuperarla hace falta un paso de
re-entrenamiento con alguna **señal de aprendizaje**. Esa señal puede venir de:

- **el ground-truth** (fine-tuning supervisado clásico), o
- **un modelo maestro** (knowledge distillation).

La elección entre estas dos opciones es la decisión central de este documento
(ver sección 4).

> En el pipeline: `src/optimization/pruning.py` hace la poda estructurada real con
> `torch_pruning`; la recuperación se delega a la etapa de distillation en la
> Fase 5 (`src/training/combined.py`).

### 3.3. Distillation (attention-transfer) — no necesita labels

La *knowledge distillation* transfiere el "conocimiento" de un modelo grande
(maestro, yolov8x-pose) a uno chico (estudiante, yolov8n-pose podado). La variante
implementada es **attention transfer** (Zagoruyko & Komodakis, 2017):

- Para cada nivel del *neck* (P3/P4/P5) se calcula el **mapa de atención
  espacial** `A(F) = normalize(mean_c F²)` — es decir, en qué zonas de la imagen
  se concentra la activación.
- La señal de distillation es el **MSE entre las atenciones del maestro y del
  estudiante**. El estudiante aprende a "mirar donde mira el maestro".

Punto clave: esta señal se calcula comparando **feature maps entre dos modelos**,
sobre las mismas imágenes. **No interviene el ground-truth en ningún punto** — el
maestro cumple el rol que normalmente cumplirían las etiquetas. Por eso la
distillation, en su componente de attention-transfer, **necesita solo imágenes**.

> **Matiz importante sobre la loss combinada.** La formulación completa es:
>
> ```
> L = α · L_task + (1 − α) · T² · L_AT
> ```
>
> - `L_AT` (attention transfer) → **no usa labels**.
> - `L_task` (la task loss normal de YOLO-pose) → **sí usa labels**.
>
> Con `α > 0` el término `L_task` reintroduce la dependencia del ground-truth
> **y**, sobre UCO, reintroduce el problema de la anotación parcial (los 14
> keypoints en `v=0` entrenan a la red a declararlos "ausentes"; ver sección 5).
>
> **Conclusión de configuración:** para una recuperación **sin etiquetas** y sin
> el problema de UCO, la señal de recuperación debe apoyarse en el término de
> distillation (`α` bajo o `α = 0`), donde el maestro es la única fuente de
> supervisión. Así la Fase 5 (prune → distill) recupera precisión usando **solo
> imágenes**.

> En el pipeline: `src/optimization/distillation.py` implementa el
> `DistillPoseTrainer` con la loss combinada; `α` (`distillation.alpha`) y `T`
> (`distillation.temperature`) se configuran en el YAML de la fase.

### 3.4. Fine-tune con ground-truth — sí necesita labels

El fine-tuning supervisado clásico ajusta los pesos minimizando el error contra
las **etiquetas reales**. Requiere, por definición, un dataset con keypoints
ground-truth. Es la alternativa a la distillation para recuperar precisión tras
la poda, y la que **requeriría etiquetar UCO** (los 17 keypoints, no solo los 3
que provee la anotación original).

### 3.5. Evaluación — sí necesita labels (imprescindible)

Independientemente de cómo se recupere la precisión, para **reportar** el efecto
de cada optimización (cuánto baja mAP / OKS / PCK al podar, al cuantizar, etc.)
hace falta comparar las predicciones contra un **ground-truth**. Sin etiquetas no
hay forma de cuantificar la degradación, que es justamente el resultado central
de la tesis.

**Cobertura actual:** UCO provee ground-truth **real y validado** para los 3
keypoints relevantes de cada ejercicio (cadera/rodilla/tobillo en flexión de
rodilla y extensión de pierna; hombro/codo/muñeca en levantamiento de brazos).
Eso alcanza para evaluar el *scope* de la tesis con OKS/PCK, que ignoran
automáticamente los keypoints en `v=0`.

---

## 4. Justificación de la decisión: distillation después de la poda

Poniendo junto lo anterior, la secuencia **poda → distillation → cuantización**
se sostiene sobre esta lógica:

1. La **poda** reduce el modelo (bien para Android) pero **degrada la precisión**.
2. Recuperar esa precisión necesita una señal de entrenamiento. Hay dos caminos:
   - **Fine-tune supervisado** → requiere un dataset etiquetado que **no
     tenemos** de forma completa (UCO solo anota 3 keypoints por ejercicio; usar
     esos labels parciales para entrenar **daña** el modelo, ver sección 5).
   - **Distillation** → recupera precisión usando **el maestro + imágenes**, sin
     depender de etiquetas ground-truth.
3. Por lo tanto, **distillation es el camino viable** para recuperar la precisión
   post-poda con los datos disponibles hoy. El maestro (yolov8x-pose,
   pre-entrenado en COCO) aporta el conocimiento de pose completo que el
   estudiante podado necesita re-aprender.
4. Finalmente, la **cuantización INT8** (que tampoco necesita labels) produce el
   artefacto TFLite para el dispositivo.

En una frase: **distillation se elige sobre fine-tuning porque permite recuperar
la precisión perdida en la poda sin necesitar un dataset etiquetado, que es
justamente lo que no tenemos de forma completa.**

---

## 5. Por qué no se usa el ground-truth parcial de UCO para entrenar

UCO es un dataset de rehabilitación: cada clip anota **solo los 3 joints del
ejercicio** (nunca la cara ni el resto del cuerpo). En el formato YOLO-pose eso
significa que 14 de los 17 keypoints quedan en `visibility = 0`.

Usar esos labels para **fine-tunear** el modelo de pose completo es
contraproducente por cómo funciona la loss de YOLOv8-pose, que tiene **dos
componentes**:

- **Loss de ubicación** (dónde está el keypoint): está *enmascarada* por
  visibilidad → los keypoints en `v=0` se excluyen. No corrompe directamente,
  pero al no recibir señal, esos joints sufren *olvido* (forgetting).
- **Loss de visibilidad/confianza** (¿el keypoint está presente?): se entrena
  contra la máscara de visibilidad. Con 14 keypoints en `v=0`, el target es
  "ausente" → **el modelo aprende activamente a declarar que la cara, el torso y
  los miembros no anotados NO existen**, aunque estén visibles. Es supresión
  activa, no solo olvido.

Por eso el ground-truth parcial de UCO sirve para **evaluar** (OKS/PCK ignoran
los `v=0`) pero **no para entrenar** el modelo completo sin antes completar la
anotación de los 17 keypoints.

---

## 6. Trabajo futuro: comparar distillation vs. fine-tuning supervisado

La decisión de este documento (recuperar vía distillation) es la adecuada **con
los datos disponibles hoy**. A futuro, si el tiempo lo permite, se puede:

1. **Etiquetar UCO de forma completa** (los 17 keypoints) mediante *model-assisted
   labeling*: pre-anotar con el maestro yolov8x-pose y **verificar/corregir
   manualmente** cada frame. Con verificación humana genuina, el resultado es un
   ground-truth válido (el humano es la autoridad final, no el modelo). Conviene
   **subsamplear** primero — los frames de UCO son muy redundantes (~55 frames
   casi idénticos por clip), así que un subconjunto representativo da lo mismo con
   una fracción del esfuerzo.
2. **Corregir el bounding box** al mismo tiempo: hoy la caja rodea solo los 3
   keypoints (queda del tamaño de un miembro); con detección de persona del
   modelo pre-entrenado se puede reemplazar por la caja de la persona completa,
   sin tocar los keypoints reales.
3. **Comparar los dos métodos de recuperación** — distillation (sin labels) vs.
   fine-tuning supervisado (con el dataset etiquetado) — y reportar cuál rinde
   mejor en el *scope* de la tesis. Esa comparación es en sí misma un aporte.

Hasta entonces, el pipeline queda como: **poda → distillation → cuantización
INT8**, con evaluación sobre los 3 joints reales de UCO.

---

### Referencias

- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a
  Neural Network.*
- Zagoruyko, S., & Komodakis, N. (2017). *Paying More Attention to Attention:
  Improving the Performance of Convolutional Neural Networks via Attention
  Transfer.*
