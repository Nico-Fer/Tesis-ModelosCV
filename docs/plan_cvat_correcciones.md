# Plan: cerrar el loop de corrección humana (CVAT → entrenamiento)

Estado: **F1–F5 y F7 implementados; F6 con configs listas, pendiente de la
anotación humana** (2026-09-28). Rama: `IntegracionCVAT`. Decisiones tomadas con
los valores recomendados (D1 train+eval por fold, D2 excluir multi sin
discrepancia con muestra de 50, D3 100 px), todas configurables en
`configs/audit_cvat.yaml`.

## Estado de implementación (2026-09-28)

| Fase | Estado | Evidencia |
|---|---|---|
| F1 selección | hecho | 998 frames: 459 `discrepancia_gt`, 187 `score_bajo`, 152 `sin_deteccion`, 150 `control`, 50 `multi_persona_muestra` (1976 multi excluidos). Postura: 183 brazo (antes 27), 815 sedente. El flag recalculado coincide con el manifest en los 74.107 frames. 52 s. |
| F2 montaje | hecho | Gate por el camino nuevo OK: visibilidad idéntica, ruido 0.005 px, 3 jobs para 5 frames, orden mezclado respetado. |
| F3 export | hecho | Verificado en el proyecto del gate: solo exporta jobs `completed`; un frame auditado con todas las personas borradas **no aparece** en el export (confirma el bug 2). |
| F4 reconcile | hecho | Sobre el export del gate: 3/998 auditados, 34 kpts sin correcciones, 1 frame vaciado. |
| F5 apply | hecho | Sobre datos reales (scratch): el frame vaciado sale del val del fold 4 y del train de los folds 0–3; verificación de membresía OK contra los manifests reales. |
| F6 experimento | configs | `configs/prune_yolov8n_deploy_v4.yaml`, `configs/quantize_deploy_v4.yaml`, `diagnose_kpt_ratio.py --oks-full`. Se corre después de anotar. |
| F7 docs | hecho | `docs/cvat_setup.md`, `AGENTS.md`, este plan, `PENDIENTES.md`. DVC: pendiente hasta que exista el gold. |

**Correcciones al plan original:**

- Los 152 frames `sin_deteccion` **no** se entrenan como fondo: el
  pseudo-labeling corrió con `skip_empty: true`, así que no están en ningún
  manifest (fold0 train+val = 73.955 = frames con persona). Anotarlos **suma**
  152 frames difíciles al entrenamiento.
- F5 no regenera los folds con `write_fold_manifests` + seed: hereda el reparto
  sujeto → fold directamente de los manifests de `uco_pseudo`, así que "mismo
  reparto" es un hecho y no una suposición. La membresía sigue `skip_empty`: el
  gold solo puede sumar frames (anotados de cero) o quitarlos (vaciados).
  Cualquier otro cambio aborta.
- Un frame auditado **sin correcciones** conserva el pseudo-label byte a byte.
  El gold reconstruye el bbox como hull + 5 %, y reemplazarlo cambiaría la
  convención de bbox en frames que el humano confirmó como correctos.
  `provenance.csv` distingue `fuente` (¿lo verificó un humano?) de `label_de`
  (¿de dónde salen los bytes?).
- F2 usa un único proyecto nuevo con una sola task; no hay task de "triage"
  aparte, porque los `multi_persona` quedan cubiertos por el GT (D2).

## Objetivo

Que las correcciones hechas a mano en CVAT **vuelvan al entrenamiento**, que la
auditoría mire los frames que realmente están mal (no solo los de baja
confianza) y que la métrica de la tesis sea correcta aunque la auditoría se
exporte a medias.

## Punto de partida (medido el 2026-09-28)

**Lo que ya funciona:** `select_audit_subset` → `pack_for_cvat` → gate de
round-trip (pasado el 2026-09-05, ruido 0.005 px) → proyecto CVAT
`auditoria-pseudo-labels` (id 4) montado → `reconcile_corrections`.

**Estado de CVAT:** proyecto 4 con 1 task (id 2) de 2546 frames en **un solo
job**, estado `new`, sin asignar. No se anotó nada todavía.

**Selección actual (2546 frames):** 2100 `multi_persona`, 152 `sin_deteccion`,
144 `score_bajo`, 150 `control`. Por postura: 2519 `sedente`, 27 `brazo`.

**Problemas encontrados:**

1. **Las correcciones no vuelven al entrenamiento.** `reconcile` escribe
   `uco_gold/` como held-out "que nunca se mezcla al train", pero ningún código
   lee `gold_stems.txt`. Resultado: los frames corregidos siguen en los folds de
   train con el pseudo-label **viejo**, y el held-out tampoco se cumple.
2. **"Auditado" se infiere por presencia en el export, y eso es incorrecto.**
   CVAT exporta todos los frames con pre-anotación, se hayan tocado o no. El
   export sin editar del gate lo demuestra: contiene los 5 frames. Con un export
   parcial, los frames no revisados cuentan como auditados con 0 correcciones y
   `verificado_por_humano=1`.
3. **El flag `baja_confianza` no detecta los errores reales.** Comparé el
   pseudo-label contra el GT de UCO (las 3 articulaciones del ejercicio, en
   `uco_yolo/labels/all`, a 1280×720) sobre los 74.107 frames:

   | | Frames | Marcados `baja_confianza` |
   |---|---|---|
   | error máx > 50 px | 2604 | 18 (1 %) |
   | error máx > 100 px | 339 | 3 |
   | articulación visible en GT que el pseudo dejó en `v=0` | 120 | 78 |

   La distribución del error es p50 22 px, p90 35, p95 43 y p99 78. El error
   de ~22 px es, casi seguro, diferencia de convención entre la articulación de
   UCO y el keypoint COCO, no un error. **Lo que vale la pena revisar a mano son
   los outliers.**
4. **Los frames `multi_persona` están bien donde importa.** Contra el GT tienen
   la misma distribución de error que el control: p50 24 vs 21 px, p99 48 vs
   47 px; solo 3 superan 100 px y 50 tienen `kpt_perdido`. Como `max_persons: 1`
   escribe un solo esqueleto, la única duda era si ese esqueleto es el paciente,
   y el GT ya lo confirma en ~97 % de los casos. `score_bajo` tampoco muestra
   errores en las articulaciones del ejercicio (máximo 45 px), pero 28 de esos
   frames tienen `kpt_perdido`.
5. **Un solo job de 2546 frames** impide exportar solo lo terminado, priorizar y
   medir el tiempo de anotación.

## Decisiones antes de implementar

| # | Decisión | Recomendado | Alternativa |
|---|---|---|---|
| D1 | Uso de las correcciones | **Train y eval, separados por fold.** Con GroupKFold por sujeto, cada frame corregido está en val en un fold y en train en los otros 4: el held-out es por **sujeto**. | Solo eval: los manifests de train excluyen `gold_stems.txt` (tampoco está implementado hoy). |
| D2 | `multi_persona` sin discrepancia con el GT | **Excluirlos** de la auditoría y dejar una muestra de 50 como estrato propio, para documentar la exclusión en la tesis. | Auditar los 2100. |
| D3 | Umbral de discrepancia con el GT | **100 px** (por encima del p99 = 78 px): ~339 frames, más los de `kpt_perdido`. | 50 px: 2604 frames, demasiados para revisar a mano. |

Con los valores recomendados, la selección queda en **~850 frames** en lugar de
2546. La cifra exacta sale al regenerarla.

## Fases

Las fases 1 a 5 se implementan **antes de empezar a anotar**: las fases 1 y 2
cambian qué se anota y la 3 cambia cómo se sabe qué está auditado. La fase 5 se
puede probar con un export sintético. La fase 6 va después de la anotación.

```
F1 selección ─► F2 montaje CVAT + gate ─► F3 export/backup ─► F4 reconcile
                                                                   │
                        [anotación humana en CVAT] ◄───────────────┘
                                                                   ▼
                                     F5 apply_corrections ─► F6 experimento
F7 (docs/DVC) acompaña a cada fase.
```

### F1 — Selección con discrepancia contra el GT de UCO

**Archivos:** `src/audit/select_audit_subset.py`, `src/audit/common.py`,
`configs/audit_cvat.yaml`, `tests/test_audit_subset.py`.

- Config nueva:
  ```yaml
  gt:
    labels_dir: data/processed/uco_yolo/labels/all
    img_size: [1280, 720]        # se verifica contra una imagen al arrancar; si no coincide, aborta
    discrepancia_thr_px: 100     # D3
  seleccion:
    excluir_multi_sin_discrepancia: true   # D2
    muestra_multi_persona: 50
  ```
- Para cada frame, comparar la primera persona del GT (UCO tiene un solo
  sujeto) contra la primera del pseudo-label (`max_persons: 1`), **solo** en las
  articulaciones con `v>0` en el GT:
  - `err_gt_px` = distancia máxima en px entre articulaciones visibles en ambos;
  - `kpt_perdido` = alguna articulación visible en el GT quedó en `v=0` en el
    pseudo-label;
  - `discrepancia_gt` = `err_gt_px > thr` **o** `kpt_perdido`.
- Nueva precedencia de `motivo`: `sin_deteccion` > `sin_kpts_validos` >
  `discrepancia_gt` > `score_bajo` > `multi_persona` > `control`.
  `multi_persona` pasa al final porque es el motivo menos informativo; si un
  frame multi-persona tiene discrepancia, probablemente se eligió a la persona
  equivocada, y así entra por `discrepancia_gt`.
- `multi_persona` sin discrepancia: se excluye y se toma una muestra
  estratificada de `muestra_multi_persona` como motivo `multi_persona_muestra`.
- El pool de control pasa a ser `baja_confianza=0` **y** `discrepancia_gt=0`.
- `SELECTION_FIELDS` suma `err_gt_px`, `kpt_perdido` y `discrepancia_gt`, para
  poder cruzar motivos en el reporte. `read_selection` sigue validando las
  columnas de forma estricta, así que hay que regenerar `selection.csv` (no se
  anotó nada, no se pierde trabajo).
- Rendimiento: leer 148k labels tardó unos minutos en la medición. Loguear el
  progreso.

**Tests:** precedencia con `discrepancia_gt`; detección de `kpt_perdido`;
control sin frames con discrepancia; `err_gt_px` escalado por `img_size`;
`img_size` que no coincide aborta.

**Aceptación:** la selección regenerada coincide con las cifras medidas arriba:
~339 frames con > 100 px, 120 con `kpt_perdido`, y solo ~3 `multi_persona` con
> 100 px.

### F2 — Montaje en CVAT por jobs, con control mezclado

**Archivos:** `src/audit/pack_for_cvat.py`, un nuevo `scripts/cvat_audit.py`
(subcomandos `montar`, `exportar`, `backup`) y un nuevo `scripts/cvat_client.py`
con los helpers que hoy están en `cvat_roundtrip_gate.py` (`leer_credenciales`,
`esperar_request`, `crear_proyecto`, `exportar_dataset`), que el gate también
pasa a importar.

- **Un proyecto nuevo** (`auditoria-v2`). **No se toca el proyecto 4.** Borrarlo
  queda para el final, con confirmación explícita, y solo después de verificar
  que no tiene ediciones: exportarlo y correr `verify_roundtrip --original` con
  los labels del pack.
- **Una sola task** con toda la selección, en **orden mezclado y determinista**
  (`seed`) para que el anotador no sepa qué frames son de control. Si los
  control estuvieran en una task aparte, la tasa de corrección "marcados vs
  control" tendría el sesgo de un anotador que revisa con menos cuidado lo que
  sabe que está sano.
- `segment_size: 100` (config `cvat.segment_size`) ⇒ ~9 jobs.
  `image_quality: 95` (config), porque el default de CVAT (70) degrada la vista
  para anotar.
- Camino por API (a verificar en v2.74.1):
  1. `POST /api/tasks` con `{name, project_id, segment_size}`;
  2. `POST /api/tasks/{id}/data` con un ZIP de las imágenes,
     `sorting_method: predefined` y `image_quality`, con los headers
     `Upload-Start` + `Upload-Finish` que ya se usan en el gate;
  3. subir las pre-anotaciones a nivel task:
     `POST /api/tasks/{id}/annotations?format=Ultralytics YOLO Pose 1.0` con el
     ZIP de labels (`data.yaml` + `train.txt` + `labels/`).
- **Persistir el mapa frame → job** en `data/processed/uco_audit/cvat_frames.csv`
  (`stem, task_id, job_id, frame`), armado con `GET /api/tasks/{id}/data/meta` y
  el `start_frame`/`stop_frame` de cada job. Verificar que el orden coincide con
  el mezclado; si CVAT no respeta `predefined` con ZIP, usar
  `sorting_method: random` y quedarse con el orden que devuelve `data/meta`.
- **Fallback manual:** en la UI, Create task → Advanced → Segment size 100 →
  subir imágenes → Actions → Upload annotations (Ultralytics YOLO Pose 1.0).
  Después, `cvat_audit.py montar --solo-mapa` genera `cvat_frames.csv` de todos
  modos.
- **Repetir el gate por el camino nuevo**, porque importar a nivel task y
  exportar por job son rutas que el gate del 05/09 no cubrió: pack de 5 frames,
  `segment_size: 2` (3 jobs), exportar cada job sin editar y correr
  `verify_roundtrip`.

**Aceptación:** gate OK por el camino nuevo; `cvat_frames.csv` cubre la
selección completa sin duplicados; el número de jobs es `ceil(n / 100)`.

### F3 — Export solo de jobs completados + backup

**Archivos:** `scripts/cvat_audit.py`.

- `cvat_audit.py exportar --project <id>`:
  - lista los jobs y se queda con los `state == completed` (config
    `cvat.estado_auditado`). Esa es la **señal de auditado**: el anotador marca
    el job como completed al terminar de revisar todos sus frames, y un frame no
    editado dentro de un job completed significa "revisado y correcto";
  - exporta cada job con `POST /api/jobs/{id}/dataset/export` (formato YOLO
    Pose, `save_images=false`) y junta todo en
    `data/processed/uco_audit/cvat_export/labels/<stem>.txt`;
  - escribe `auditados.txt` a partir de `cvat_frames.csv` de los jobs completed,
    **nunca** a partir de qué archivos aparecen en el export. Un frame auditado
    en el que el anotador borró todas las personas no aparece en el export;
  - escribe `export_meta.json` con fecha, versión de CVAT e ids y estado de los
    jobs.
- `cvat_audit.py backup --project <id>`: guarda el backup del proyecto
  (`/api/projects/{id}/backup/export`, a verificar) en
  `data/processed/uco_audit/backups/`. Las horas de anotación son el artefacto
  más caro del pipeline, y un `docker compose down -v` las borra.
- Opcional, no bloquea: ver si v2.74.1 expone el tiempo de trabajo por job
  (events/analytics). Si lo expone, guardarlo en `export_meta.json` para el
  pendiente "horas por frame" del paper.

**Aceptación:** con 1 de 3 jobs en completed en el proyecto del gate,
`auditados.txt` lista solo los frames de ese job.

### F4 — Reconciliación basada en `auditados.txt`

**Archivos:** `src/audit/reconcile_corrections.py`,
`tests/test_audit_reconcile.py`.

- El conjunto auditado sale de `<export>/auditados.txt`, que es obligatorio: si
  falta, `PipelineError`/`ValueError`, porque sin esa lista no hay forma de
  saber qué se revisó.
- Frame auditado sin label en el export ⇒ `corregidas = []`. Es una decisión
  válida del anotador y se cuenta como `personas_eliminadas`; en UCO siempre hay
  un paciente, así que se loguea un warning con el total.
- Frame presente en el export pero **no** en `auditados.txt` ⇒ `sin_auditar`.
- Un stem de `auditados.txt` que no está en la selección aborta: indica un
  proyecto o export equivocado.
- **Reconstruir `gold_root` desde cero** en cada corrida (borrar
  `labels/all` e `images/all`). Hoy solo sobreescribe por stem, y si un job se
  reabre quedan archivos viejos.
- Actualizar el docstring sobre el held-out según D1.

**Tests:** estar en el export no implica estar auditado; frame auditado sin
label ⇒ gold vacío; limpieza de archivos viejos; `auditados.txt` faltante ⇒
error.

### F5 — `apply_corrections`: dataset corregido y folds

**Archivos:** nuevo `src/audit/apply_corrections.py`, `configs/audit_cvat.yaml`,
nuevo `tests/test_audit_apply.py`.

```yaml
aplicar:
  out_root: data/processed/uco_pseudo_corrected
  out_yaml: data/processed/pose_dataset_pseudo_corrected.yaml
  usar_en_train: true          # D1; con false, los manifests de train excluyen gold_stems
```

- Parte de cero cada vez; es un artefacto derivado.
- Hardlinkea las 74.107 imágenes de `uco_pseudo/images/all` con `link_image`.
- Copia los labels de `uco_pseudo/labels/all` (con `copyfile`, byte a byte) y
  **reemplaza** los stems del gold que tuvieron correcciones por su label gold.
  Los auditados sin correcciones conservan el pseudo (ver "Correcciones al plan
  original"). **Nunca** edita `uco_pseudo` en el lugar: es el baseline de la
  comparación y tiene su `.cache` de Ultralytics.
- Escribe `provenance.csv` (`stem, fuente ∈ {pseudo, humano}, label_de ∈
  {pseudo, gold}, motivo`).
- Folds *(implementado distinto a lo propuesto)*: el reparto sujeto → fold se
  hereda de los `fold{k}_val.txt` de `uco_pseudo`; entra a los manifests todo
  frame cuyo label tenga una persona (`skip_empty`). Después, cada fold se
  verifica: solo los stems del gold pueden cambiar de membresía respecto de
  `uco_pseudo`.
- Genera también `pose_dataset_pseudo_corrected_coco_fold0.yaml`, el mix v3 con
  las rutas de `uco_pseudo_corrected` y el mismo manifest COCO.
- Genera los manifests de eval gold: `folds/fold{k}_val_gold.txt` (gold ∩ val
  del fold k) y `pose_dataset_pseudo_corrected_gold_fold{k}.yaml` (train = train
  del fold, val = el subset gold). `pose_eval` lee solo `val`; `train` está
  porque Ultralytics exige la clave.
- Nota de caché: los dos mix (pseudo+COCO y corregido+COCO) comparten el
  `.cache` bajo `coco_pose/`, porque se nombra según el primer directorio de
  labels ordenado. Ultralytics valida por hash (rutas + tamaños), así que solo
  se regenera; no hay riesgo de usar labels equivocados.

**Tests:** 74.107 imágenes y labels; los labels difieren del pseudo **solo** en
stems del gold; todo pasa `check_label_line`; los sujetos de val por fold son
idénticos a los de `uco_pseudo`; `fold{k}_val_gold` ⊂ `fold{k}_val`; con
`usar_en_train: false`, ningún stem del gold aparece en un manifest de train.

### F6 — Experimento: ¿la corrección humana mejora el modelo?

Va después de la anotación.

- `configs/prune_yolov8n_deploy_v4.yaml` = v3 con
  `data: data/processed/pose_dataset_pseudo_corrected_coco.yaml`, experimento y
  `save_checkpoint` propios; `configs/quantize_deploy_v4.yaml` en paralelo.
  Single-fold (fold 0), como v3.
- Evaluar **v3 y v4** sobre:
  1. el val de fold 0 con labels corregidos (`pose_dataset_pseudo_corrected_fold0.yaml`);
  2. el subset gold del val de fold 0
     (`pose_dataset_pseudo_corrected_gold_fold0.yaml`), con desglose por motivo
     desde `gold_manifest.csv`. El subset está sesgado hacia frames difíciles; el
     estrato `control` es la única muestra aleatoria;
  3. el gate OOD `diagnose_kpt_ratio.py --gt-set`: v4 no puede empeorar a v3
     (span ≥ 0.9, ≤ 1/79 colapsadas).
- Para evaluar un checkpoint sobre otro YAML sin reentrenar: **hecho**,
  `diagnose_kpt_ratio.py --oks-full --model <m> --data <yaml>` puntúa todo el
  val del YAML.
- Expectativa: los frames auditados son ~1,3 % del pool (998 / 74.107), así que
  en el OKS global el efecto probablemente sea chico. Donde hay que mirar es el
  subset gold y los 152 frames sin detección, que hoy **no están** en el
  entrenamiento (`skip_empty`) y entran una vez anotados. Si el resultado es
  nulo, se reporta igual.

### F7 — Docs y versionado (acompaña cada fase)

- `docs/cvat_setup.md`: flujo nuevo (task por jobs, `completed` como señal de
  auditado, `cvat_audit.py exportar/backup`, por qué el control va mezclado) y
  la guía de anotación actualizada.
- `AGENTS.md`: actualizar la sección de auditoría con la semántica del gold
  según D1, `apply_corrections` y `uco_pseudo_corrected`.
- `docs/paper_pipeline/PENDIENTES.md`: actualizar el ítem de la auditoría y
  sumar las cifras de discrepancia con el GT, que son un resultado en sí.
- DVC: versionar `uco_gold/`, `uco_pseudo_corrected/labels` + `folds/` y
  `uco_audit/*.csv`. **No** las imágenes hardlinkeadas: DVC las duplicaría en su
  caché (GBs) y se regeneran con `apply_corrections`.

## Riesgos y cosas a verificar en CVAT v2.74.1

| Qué | Si falla |
|---|---|
| `POST /api/tasks/{id}/data` con ZIP + `segment_size` en una sola request | Crear la task en la UI (fallback de F2) |
| `sorting_method: predefined` respetado con un ZIP | `random`, y usar el orden de `data/meta` |
| Importar Ultralytics YOLO Pose a nivel task (match por nombre) | Lo detecta el gate de F2; no anotar hasta que pase |
| `POST /api/jobs/{id}/dataset/export` | Exportar la task entera y filtrar por `cvat_frames.csv` (sigue siendo correcto porque la señal es `auditados.txt`) |
| Endpoint de backup del proyecto | Backup desde la UI (Project → Backup) |

## Fuera de alcance (opcional para la tesis)

- Acuerdo intra-anotador: re-anotar ~50 frames a las 2 semanas para medir el
  ruido humano frente a `corregido_thr_px`.
- Auditar las articulaciones que no son del ejercicio (cara, el otro miembro):
  el GT de UCO no las cubre, así que solo quedan cubiertas por `score_bajo` y el
  control.
- Arreglar la destilación (sección BROKEN de `AGENTS.md`): independiente de
  este plan.
