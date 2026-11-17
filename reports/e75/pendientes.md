# E75 — Pendientes (A3)

Estado al 2026-10-16. Cada ítem con su alcance: qué falta, qué consecuencia tiene sobre lo
que el informe puede afirmar y qué lo resolvería.

| # | Pendiente | Alcance | Qué lo resuelve |
|---|---|---|---|
| 1 | **Comparador sin validar** con grabaciones etiquetadas por un profesional | No se sabe si distingue una ejecución correcta de una incorrecta; los umbrales de `tab:umbrales_comparador` no están calibrados | Campaña de grabaciones etiquetadas por clase |
| 2 | **Modelo propio sin integrar en la app** | `PoseModelSpec.ACTIVE = STOCK_YOLOV8N`: todas las mediciones en dispositivo describen al modelo de dominio general | Apuntar `ACTIVE` a `STUDENT_V3`, regenerar las referencias (mismo estimador) y repetir |
| 3 | **Mediciones del modelo propio en el dispositivo** (RNF-03, RNF-04) | Su latencia sólo se conoce en la CPU de escritorio (dynamic range, ~185 ms, no representativa de un teléfono) | Repetir las dos campañas (Test Lab y uso real) con el v3 |
| 4 | **Batería y memoria** | No se midieron con ningún modelo; RNF-13 sólo tiene criterio de tamaño (≤ 4 MB, se cumple) | Registro de batería/memoria en sesión larga |
| 5 | **Línea base sin podar** | El salto base → recuperado mezcla poda, recuperación y adaptación de dominio | Fase 1 entrenada sobre UCO con la misma receta |
| 6 | **Comparación con otras familias** (MoveNet, BlazePose) | El objetivo 1 pedía elegir por comparación; no se hizo con este protocolo | Evaluar con `pose_eval` y la compuerta |
| 7 | **Prueba con usuarios acotada** | 4 adultos mayores + 1 kinesiólogo; nadie necesitó la emergencia | Más participantes, reclutados de forma independiente |
| 8 | **Auditoría CVAT** | 998 cuadros montados (tarea 4, 10 jobs) el 28/09; 0 auditados exportados; sin `audit_report.json`. Modos de falla del anotador sin caracterizar | Revisar los 10 jobs → `cvat_audit.py exportar` → `reconcile_corrections` |
| 9 | ~~CV de la receta v3~~ **Resuelto el 2026-10-17 16:49** | 5/5 particiones (`fase2-v3-cv`): OKS INT8 0,952 ± 0,006; PCK@0,05 instrumental 0,941 ± 0,013; compuerta error 0,033 ± 0,001, 4/400 colapsadas, detección 395/400. La partición 3 se relanzó dos veces desde cero (OOM de GPU y corte de luz); su resultado es de una sola ejecución completa | — |
| 10 | **Baja de cuenta contra la base real** | El borrado en cascada (tolerancias, referencias) lo aplica PostgreSQL; los tests con repositorios simulados no lo observan (`BajaCuentaTests.cs`) | Prueba de integración contra PostgreSQL |
| 11 | **Canal ARCO no operativo** | Declarado en la pantalla de privacidad; el plazo del art. 14 de la Ley 25.326 no es verificable | Poner en operación el canal |

Además, fuera del pedido original pero encontrado en esta pasada:

- El informe decía paso de muestreo de 30 cuadros y 5–8 GB; los manifiestos muestran paso
  de **15** y el pool ocupa **12 GB**. Corregido en el informe.
- La calibración INT8 son los primeros 300 cuadros alfabéticos (un solo sujeto), no una
  muestra «representativa». Corregido en el informe.
- La compuerta de train2017 está en el split de preentrenamiento del modelo de partida.
  Declarado en el informe.
