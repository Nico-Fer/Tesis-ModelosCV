# Fase 5 — Validación cruzada completa (YOLOv8n-pose, 5 folds × 5 etapas)

Esta carpeta reúne los resultados finales de la validación cruzada de la Fase 5 del
pipeline de optimización (poda → fine-tune → distillation → INT8). La corrida fue
del 8 al 14 de octubre de 2026 y se hizo con GroupKFold por sujeto (k = 5) sobre el
dataset UCO con pseudo-labels ViTPose-H. Contiene todo lo necesario para citar e
interpretar los números en el informe.

## Resultado en una línea

El modelo entregable (YOLOv8n-pose podado 30 % + fine-tune + distillation + INT8) logra
**OKS 0,964 ± 0,006 y PCK@0.2 0,998 ± 0,001 sobre sujetos no vistos**, con
**1,91 M de parámetros (−42 %) y 2,39 MB (−65 %)**. El INT8 cuesta sólo
−0,0036 OKS.

| Etapa | OKS (media ± σ, 5 folds) | PCK@0.2 | Params | Tamaño |
|---|---|---|---|---|
| base (COCO, sin adaptar) | 0,855 ± 0,007 | 0,981 | 3,30 M | 6,83 MB |
| pruned (sin recuperar) | 0,000 | 0,000 | 1,91 M | — |
| pruned_ft | 0,957 ± 0,008 | 0,995 | 1,91 M | — |
| distilled | 0,968 ± 0,006 | 0,998 | 1,91 M | 4,07 MB |
| **int8 (entregable)** | **0,964 ± 0,006** | **0,998** | 1,91 M | **2,39 MB** |

**Ninguno de estos modelos es desplegable.** Fuera de UCO fallan la compuerta de
generalización igual que v2: detectan a la persona en ~31 % de las imágenes externas y
el esqueleto cae fuera del cuerpo (error 0,42 de la diagonal, criterio ≤ 0,05). Ver
[gate_ood/GATE_OOD.md](gate_ood/GATE_OOD.md). Estos números caracterizan la cadena de
optimización dentro del dominio; el modelo de la app sigue siendo v3.

Antes de citar estos números conviene leer las advertencias de
[RESULTADOS.md §6](RESULTADOS.md#6-advertencias-para-citar-estos-números). Las tres
principales:

- la referencia son pseudo-labels de ViTPose-H, no anotación humana;
- el `base` es un modelo COCO, no un baseline entrenado en UCO;
- la mejora de la etapa de distillation no es atribuible a la distillation en sí.

## Qué leer y en qué orden

| Documento | Para qué |
|---|---|
| [RESULTADOS.md](RESULTADOS.md) | Tablas, figuras, qué se puede afirmar y qué no, advertencias |
| [METODOLOGIA.md](METODOLOGIA.md) | Datos, split, receta de cada etapa, cómo se calcula cada métrica, estadística, amenazas a la validez, glosario |
| [gate_ood/GATE_OOD.md](gate_ood/GATE_OOD.md) | Compuerta fuera de dominio de los 5 modelos contra v2, v3 y el stock; set de imágenes reconstruido y su validación. Incluye `fragmento_informe.tex` para la sección `artefacto_generalizacion` |
| [bitacora_ejecucion.md](bitacora_ejecucion.md) | Registro cronológico de la corrida: hallazgos, pausas y reanudaciones, verificaciones (copia de `docs/fase5_cv_bitacora.md`) |

## Contenido de la carpeta

```
fase5_cv/
├── README.md                    este índice
├── RESULTADOS.md                resultados e interpretación
├── METODOLOGIA.md               protocolo y definiciones
├── bitacora_ejecucion.md        registro de la corrida
├── tablas/
│   ├── metricas_por_fold.csv    fold × etapa: oks, pck, pck_leg, pck_arm, params, latency_ms, size_mb, n_frames
│   ├── resumen_por_etapa.csv    etapa × métrica: media, σ poblacional y muestral, mín, máx, n
│   ├── deltas_pareados.csv      diferencias entre etapas por fold, media, IC 95 %, t, p, folds con el mismo signo
│   ├── entrenamiento_por_fold.csv  tamaños de split, sujetos de val, epochs, mejor epoch, horas, run ID
│   └── mlflow_runs.json         métricas, params y tags del run padre y de los 5 runs de fold
├── curvas/
│   ├── fold{k}_finetune_post_poda.csv  results.csv de Ultralytics del fine-tune (métricas por epoch)
│   └── fold{k}_distillation.csv        results.csv de Ultralytics de la distillation
├── figuras/
│   ├── oks_por_etapa.png        OKS por etapa: media ± σ y cada fold
│   ├── deltas_pareados.png      aporte de cada etapa con IC 95 %
│   ├── curvas_entrenamiento.png mAP de validación por epoch, por fold
│   └── tamano_vs_oks.png        tamaño del archivo vs. OKS
├── gate_ood/
│   ├── GATE_OOD.md              compuerta fuera de dominio: veredicto, método, resultados
│   ├── fragmento_informe.tex    párrafo + tabla para la sección artefacto_generalizacion
│   ├── resumen.csv              set × modelo: razón, colapsadas, error, detección con conf 0,25
│   ├── por_imagen.csv           set × modelo × imagen
│   ├── gate_{train,val}2017.json  salida completa de gate_gt_set
│   ├── compuerta_ood.png        detección y error de punto por modelo
│   ├── razon_por_modelo.png     razón de extensión pred/GT por imagen
│   └── overlays/                esqueleto predicho vs GT, imágenes de ejemplo
├── modelos/
│   └── fold{k}_int8.tflite      el .tflite INT8 de cada fold (no versionado en git)
└── config/
    ├── combined_yolov8n.yaml    la config exacta usada
    └── procedencia.json         run IDs MLflow, huella de la config, commit, versiones, MD5 de los modelos
```

Las tablas están en CSV con punto decimal, para leerlas con pandas o Excel. Los
documentos y figuras usan coma decimal, como el informe.

## Procedencia

| | |
|---|---|
| Experimento MLflow | `fase5-combined` (id `853839127993800373`) |
| Run padre | `11734b62e1504b15a146dacd944f96a5`, con los agregados `{métrica}_{etapa}_mean/_std/_n_folds` |
| Runs de fold | F0 `ee2f122c…` · F1 `c06817a6…` · F2 `f12d16d3…` · F3 `cec8614d…` · F4 `9c0f3ea0…` (IDs completos en `config/procedencia.json`) |
| Config | `configs/combined_yolov8n.yaml`, huella `3aba772da4400779` |
| Código | rama `RunCrossValidation`, commit `d1dde1a` |
| Entorno | Python 3.11.9, torch 2.12.1+cu126, Ultralytics 8.3.253, RTX 3090 |
| Artefactos originales | `runs/fase5_cv/fold{k}/`: `student_pruned.pt`, `student_pruned_ft.pt`, `distillation/distill_student/weights/best.pt` y el `best_int8.tflite` |

Para regenerar todo menos los `.md` y el `.tex` escritos a mano (desde la raíz del repo,
con venv-torch):

```
venv-torch\Scripts\python.exe scripts\export_fase5_cv_results.py
python scripts\build_ood_gt_set.py --annotations-dir <dir con person_keypoints_*.json>
venv-torch\Scripts\python.exe scripts\run_ood_gate_fase5.py
```
