# Pendientes del paper "pipeline de dataset y entrenamiento"

Agrupados por sección del `paper_pipeline.tex`. Cada ítem indica qué falta, y el
comando o archivo que lo resolvería. Ninguno de estos huecos se completó con un
valor inventado o estimado (ver regla de honestidad numérica del pedido original).

## 1. Introducción

- `[PENDIENTE: confirmar el número de referencia bibliográfica final de la cita
  anonimizada al trabajo complementario de comparación de movimiento (DTW), una
  vez fijado el orden definitivo de citas del documento]`.

## 2. Trabajos relacionados

- `[VERIFICAR CITA: dataset UCO Physical Rehabilitation — la atribución
  "Aguilar-Ortega et al., 2023, Sensors, vol. 23, n.º 21" aparece solo como
  sugerencia sin confirmar en docs/redaccion-secciones-4.4-y-5.1.md. Verificar
  autores, año, volumen y DOI contra la publicación/sitio de distribución oficial
  del dataset antes de citarla.]`
- `[VERIFICAR CITA: afirmación sobre "top-down pose estimation como el enfoque
  estándar de mayor exactitud" — la fuente sugerida ("Chen et al., 2023") no está
  identificada de forma concreta en ningún archivo del repo. Buscar un survey o
  paper real que respalde la afirmación, o reformular la oración sin cita.]`
- Las siguientes SÍ son reutilizables tal cual (bibliografía real del DTW_PAPER,
  ya usada por el mismo equipo): YOLO-Pose (Maji et al. 2022), Ultralytics Pose
  Estimation docs, cuantización INT8 (Jacob et al. 2018), fiabilidad de pose
  móvil (Hellstén et al. 2025), captura markerless por smartphone (Cunha et al.
  2025). No requieren marcador.
- El resto de las referencias nuevas propuestas (PyTorch, MLflow, DVC, OpenCV,
  DepGraph, COCO, Pseudo-Label, RTMDet, ViTPose, Attention Transfer, Distillation,
  Pruning, validación cruzada por sujeto) son papers reales y bien conocidos, pero
  no se verificaron contra una fuente externa en esta sesión (no hubo búsqueda
  web). `[PENDIENTE: pasada final de verificación bibliográfica (volumen, páginas,
  DOI) de las 12 entradas de "alta confianza" en referencias.bib antes de envío]`.

## 3. Requisitos del conjunto de datos

- `[PENDIENTE: verificar términos exactos de licencia/EULA del dataset UCO
  Physical Rehabilitation — revisar el sitio de distribución oficial y/o el
  artículo del dataset, y transcribir el tipo de licencia (p. ej. uso académico
  no comercial) y cualquier requisito de atribución. Hasta entonces, el paper
  enuncia la restricción de subir datos a terceros como una decisión de diseño
  asumida, no como un hecho de licencia verificado — así lo marca ya el propio
  borrador de tesis interno en docs/redaccion-secciones-4.4-y-5.1.md §5.1.2.2.]`

## 4. Pipeline de anotación

- `[PENDIENTE: estadísticas del dataset UCO crudo antes del muestreo por stride
  (duración total de video, cantidad de frames por sujeto/ejercicio/cámara antes
  de aplicar stride=30) — no existe un archivo de metadatos que las registre;
  habría que derivarlas leyendo los .mp4/camN_p2d.txt directamente.]`
- `[PENDIENTE: conteo exacto de imágenes train/val por cada uno de los 5 folds
  (GT y pseudo) — los manifiestos folds/fold{k}_{train,val}.txt existen pero no
  hay un resumen numérico ya calculado. Comando: contar líneas de cada .txt en
  data/processed/uco_yolo/folds/ y data/processed/uco_pseudo/folds/.]`
- `[PENDIENTE: resultado cuantitativo completo de la auditoría CVAT — la
  anotación humana no se hizo todavía. No existen data/processed/uco_gold/ ni
  audit_report.json. El pipeline está implementado y verificado (2026-09-28):
  selección de 998 frames, gate de round-trip por el camino real (ruido
  0.005 px), export por jobs completados, reconcile y apply probados sobre datos
  reales. Flujo completo en docs/cvat_setup.md; pasos que faltan: anotar en
  CVAT → scripts/cvat_audit.py exportar → src.audit.reconcile_corrections →
  src.audit.apply_corrections.]`
- **Resultado ya disponible (sin anotación humana) para la sección de
  anotación:** contra el GT de UCO (3 articulaciones por ejercicio, 74.107
  frames), el error máximo del pseudo-label por frame es p50 22 px, p90 35,
  p95 43, p99 78; 2604 frames superan 50 px y 339 superan 100 px, y el flag
  `baja_confianza` del teacher marca solo 18 y 3 de ellos respectivamente. 120
  frames pierden a `v=0` una articulación visible en el GT. Es evidencia
  medida de que la confianza del teacher no detecta sus errores en las
  articulaciones clínicamente relevantes. Reproducible con
  `python -m src.audit.select_audit_subset --config configs/audit_cvat.yaml`
  (columnas `err_gt_px` / `kpt_perdido` de `selection.csv`). Nota: el error de
  ~22 px típico es, casi seguro, diferencia de convención articulación UCO vs
  keypoint COCO, no error; conviene decirlo al reportar.
- `[PENDIENTE: costo de anotación (horas humanas por cuadro auditado) — depende
  del resultado de la auditoría real, no instrumentado en el repo.]`
- `[PENDIENTE: mAP o exactitud del anotador (RTMDet-m + ViTPose-H) contra un
  benchmark externo con ground-truth real — el pipeline solo reporta estadísticas
  de confianza propias (mean_kpt_score, tasa de detección), no una métrica de
  exactitud validada externamente.]`

## 5. Entrenamiento y compresión para despliegue

- Sin pendientes de datos: todos los hiperparámetros citados en esta sección
  provienen directamente de los YAML reales en `configs/` y de los módulos en
  `src/optimization/` y `src/models/yolo.py` (ver `datos_extraidos.md`, bloques 5
  y 6).

## 6. Resultados

- `[PENDIENTE: ver "auditoría CVAT" arriba — la subsección de costo/resultado de
  auditoría depende de la misma corrida no ejecutada.]`
- `[PENDIENTE: conteo exacto de imágenes por fold para la Tabla 1 (composición
  del corpus por etapa) — mismo comando que el ítem de folds arriba.]`
- El diagnóstico de iteración negativa (bug de destilación Fase 5 + colapso OOD
  v2→v3) SÍ tiene todos los números medidos y trazables — no requiere marcador,
  ver `datos_extraidos.md` bloques 7 y 8.

## 7. Limitaciones

- Sin pendientes nuevos — se redacta a partir de los hallazgos ya documentados
  (destilación rota, auditoría no corrida, EULA no verificada, validación cruzada
  del modelo final no ejecutada por costo de cómputo).

## 8. Conclusiones y trabajo futuro

- Sin pendientes — se listan explícitamente como trabajo futuro: (a) validación
  cruzada agrupada por sujeto sobre el modelo final desplegado (pendiente por
  costo de cómputo: 5× el costo de una corrida single-fold, ver
  datos_extraidos.md bloque 5); (b) finalización de la campaña de auditoría CVAT
  en curso.

## Resumen de comandos que resolverían pendientes (referencia rápida)

| Pendiente | Comando / archivo |
|---|---|
| Estadísticas UCO crudo | Derivar de `data/raw/uco/**/camN_p2d.txt` + metadata de video |
| Conteo train/val por fold | Contar líneas de `data/processed/{uco_yolo,uco_pseudo}/folds/fold{k}_{train,val}.txt` |
| Auditoría CVAT completa | Anotar en CVAT → `scripts/cvat_audit.py exportar` → `reconcile_corrections` → `apply_corrections` (flujo completo en `docs/cvat_setup.md`) |
| Costo de anotación | Depende de la auditoría real (arriba) |
| mAP del anotador | No implementado — requeriría un benchmark externo con GT real |
| Licencia UCO | Sitio/artículo oficial de distribución del dataset UCO Physical Rehabilitation |
| Verificación bibliográfica final | Confirmar cada entrada de `referencias.bib` contra su fuente primaria antes de enviar |
