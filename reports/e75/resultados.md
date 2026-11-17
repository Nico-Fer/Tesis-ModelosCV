# E75 — Resultados

Generado 2026-10-17T16:49:14 por `scripts/e75_build_report.py` (commit d1dde1a18b1e1e779b8ffcff381b1973e67edaf5+cambios).
Distancias como fracción del alto del cuadro (720 px); dx, dy en px (pseudo − original).

## 1. Pseudo-etiquetas contra la anotación original de UCO (pool completo)

| Articulación | n | PCK@0,05 | PCK@0,10 | Error mediano | dx (px) | dy (px) |
|---|---|---|---|---|---|---|
| left_shoulder | 7676 | 0,982 | 1,000 | 0,024 | -3.5 | +7.3 |
| right_shoulder | 7736 | 0,986 | 1,000 | 0,027 | -2.0 | +11.0 |
| left_elbow | 7676 | 0,924 | 0,996 | 0,022 | -5.8 | +3.9 |
| right_elbow | 7736 | 0,905 | 0,996 | 0,023 | +4.8 | +5.5 |
| left_wrist | 7676 | 0,773 | 0,951 | 0,023 | -1.6 | +0.2 |
| right_wrist | 7736 | 0,760 | 0,957 | 0,028 | +0.2 | +0.2 |
| left_hip | 29481 | 0,989 | 1,000 | 0,027 | -15.4 | +3.6 |
| right_hip | 29062 | 0,981 | 0,999 | 0,028 | +14.8 | +4.0 |
| left_knee | 29481 | 0,996 | 0,999 | 0,014 | +5.4 | +1.8 |
| right_knee | 29062 | 0,984 | 0,998 | 0,017 | -8.7 | +0.4 |
| left_ankle | 29481 | 0,975 | 0,994 | 0,013 | +1.6 | -1.2 |
| right_ankle | 29062 | 0,967 | 0,994 | 0,017 | -5.0 | -1.7 |
| TOTAL | 221865 | 0,962 | 0,994 | 0,020 | -1.5 | +1.3 |

Cuadros comparados: 73955 (sin pseudo-etiqueta: 152). Error máximo por cuadro: {'p50': 22.20081788729134, 'p95': 43.227567312324915, 'p99': 77.77059538218988, 'mayor_a_100px': 339}.

## 2. Auditoría CVAT

```json
{
  "seleccionados": 998,
  "por_estrato": {
    "discrepancia_gt": 459,
    "score_bajo": 187,
    "sin_deteccion": 152,
    "control": 150,
    "multi_persona_muestra": 50
  },
  "montados_en_cvat": 998,
  "jobs": 10,
  "tarea_cvat": [
    "4"
  ],
  "auditados_exportados": 0,
  "audit_report": [],
  "nota": "Sólo existe el auditados.txt del gate de round-trip (5 cuadros de prueba)."
}
```

## 3. Exactitud contra la referencia instrumental, partición 0 (16.149 cuadros)

| Modelo | MD5 | OKS vs pseudo | PCK@0,05 | PCK@0,10 | Error | inf. 0,05 | sup. 0,05 |
|---|---|---|---|---|---|---|---|
| app_stock_int8 | ee27dcf9 | 0,8441 | 0,756 | 0,939 | 0,032 | 0,736 | 0,823 |
| base_pt |  | 0,8459 | 0,755 | 0,939 | 0,032 | 0,733 | 0,829 |
| v2_fp32 | bf0a2fda | 0,9354 | 0,940 | 0,996 | 0,022 | 0,965 | 0,854 |
| v2_int8 | babaf180 | 0,9310 | 0,937 | 0,996 | 0,022 | 0,963 | 0,851 |
| v3_fp32 | 01360a5a | 0,9436 | 0,928 | 0,996 | 0,023 | 0,951 | 0,854 |
| v3_int8 | 63a694b7 | 0,9409 | 0,925 | 0,996 | 0,023 | 0,948 | 0,848 |
| pseudo |  | 1,0000 | 0,964 | 0,997 | 0,020 | 0,981 | 0,907 |

## 4. Validación cruzada — Receta de la Fase 5 (INT8)

| Medida | Media ± σ |
|---|---|
| oks_vs_pseudo | 0,9640 ± 0,0062 |
| pck02_vs_pseudo | 0,9980 ± 0,0011 |
| total_pck0.05 | 0,9512 ± 0,0093 |
| total_pck0.10 | 0,9956 ± 0,0013 |
| total_error_mediano | 0,0213 ± 0,0008 |
| pierna_pck0.05 | 0,9767 ± 0,0133 |
| brazo_pck0.05 | 0,8531 ± 0,0197 |
| compuerta_error_kpt | 0,4177 ± 0,0936 |
| compuerta_razon | 0,8222 ± 0,0396 |
| compuerta_colapsadas | 108/283 |
| compuerta_det025 | 122/400 |

## 4. Validación cruzada — Receta v3 (INT8)

| Medida | Media ± σ |
|---|---|
| oks_vs_pseudo | 0,9519 ± 0,0057 |
| pck02_vs_pseudo | 0,9980 ± 0,0011 |
| total_pck0.05 | 0,9408 ± 0,0125 |
| total_pck0.10 | 0,9951 ± 0,0014 |
| total_error_mediano | 0,0226 ± 0,0012 |
| pierna_pck0.05 | 0,9664 ± 0,0179 |
| brazo_pck0.05 | 0,8417 ± 0,0206 |
| compuerta_error_kpt | 0,0329 ± 0,0008 |
| compuerta_razon | 0,9415 ± 0,0076 |
| compuerta_colapsadas | 4/400 |
| compuerta_det025 | 395/400 |
