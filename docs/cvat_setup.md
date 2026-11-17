# CVAT self-hosted para la auditoría de pseudo-labels

CVAT corre **aislado en Docker**: no toca `venv-torch` / `venv-tf` / `venv-mmpose`
ni el toolchain de export. Se usa para auditar manualmente el subconjunto de
pseudo-labels de ViTPose que tiene más probabilidad de estar mal, y para
devolver las correcciones al entrenamiento (`uco_gold/` → `uco_pseudo_corrected/`).

> **¿Vas a anotar?** La guía paso a paso para levantar CVAT y usar el editor es
> [guia_auditoria_cvat.md](guia_auditoria_cvat.md). Este documento es la
> referencia técnica.

## Instalación

Requisito: Docker Desktop (WSL2) corriendo.

```bash
# Clone DENTRO del workspace, gitignoreado (/cvat/), pineado a release estable
git clone --branch v2.74.1 --depth 1 https://github.com/cvat-ai/cvat.git cvat

cd cvat
docker compose up -d          # primera vez descarga ~5 GB de imágenes
```

**Puertos publicados: 8080 (UI + API) y 8090.** Verificado sin colisión con el
repo — lo único que escucha aquí es MLflow en 5000. UI: <http://localhost:8080>.

Apagar / reencender (los datos persisten en volúmenes Docker):

```bash
docker compose down           # detiene; NO borra anotaciones
docker compose up -d
# docker compose down -v      # ⚠ SOLO si querés borrar TODO (DB incluida)
```

## Superuser

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Variante no interactiva (la usada en el setup inicial): pasar
`DJANGO_SUPERUSER_PASSWORD` como env y `--noinput --username admin --email
admin@localhost`. Las credenciales viven en `cvat_credentials.local`
(formato `KEY=VALUE`: `CVAT_URL` / `CVAT_USER` / `CVAT_PASS`), cubierto por el
patrón `*.local` del `.gitignore` — **nunca se commitea**.

## Verificación post-instalación

```powershell
# server arriba
Invoke-RestMethod http://localhost:8080/api/server/about   # → version 2.74.1

# string EXACTO del formato en la versión desplegada (no asumirlo: cambia entre versiones)
# GET /api/server/annotation/formats (autenticado) debe listar
#   "Ultralytics YOLO Pose 1.0"  como importer Y exporter
```

## Proyecto con skeleton COCO-17

El proyecto CVAT necesita el label `person` de tipo **skeleton** con los 17
keypoints COCO **en orden canónico** (el importer mapea los puntos del `.txt` a
los sublabels **por posición**) y las 19 aristas de `COCO_SKELETON`.
`scripts/cvat_audit.py montar` lo crea por API desde `cvat_skeleton_labels.json`
(generado por `pack_for_cvat` a partir de `keypoint_schema.py`).

**Camino manual (fallback si la API cambia):** UI → Projects → Create new
project → Constructor → pestaña **Raw** → pegar el contenido de
`data/processed/uco_audit/cvat_skeleton_labels.json` → Done.

## Qué se audita

`select_audit_subset` arma la selección (998 frames al 2026-09-28):

| motivo | frames | por qué |
|---|---|---|
| `sin_deteccion` | 152 | el detector no disparó: se anota de cero |
| `discrepancia_gt` | 459 | el pseudo discrepa del GT de UCO en las articulaciones del ejercicio (> 100 px, o una articulación del GT perdida a `v=0`) |
| `score_bajo` | 187 | score medio < 0.5 o < 10 keypoints visibles |
| `multi_persona_muestra` | 50 | muestra de los 1976 multi-persona que coinciden con el GT (excluidos: el esqueleto ya es del paciente) |
| `control` | 150 | frames sanos, estratificados: línea base de la tasa de corrección |

`discrepancia_gt` es el criterio que encuentra errores reales: el flag de
confianza del teacher marcaba solo 3 de los 339 frames con error > 100 px.

## Flujo de auditoría completo

```bash
# 1. Selección (≈1 min: compara 74k pseudo-labels contra el GT)
python -m src.audit.select_audit_subset --config configs/audit_cvat.yaml

# 2. Packs: round-trip (5 frames) y completo. El orden de los frames se mezcla
#    (seed) para que el anotador no distinga control de marcados.
python -m src.audit.pack_for_cvat --config configs/audit_cvat.yaml --limit 5
python -m src.audit.pack_for_cvat --config configs/audit_cvat.yaml

# 3. GATE (bloquea la auditoría real): monta el pack de 5 frames por el MISMO
#    camino que la real (task con jobs de 2 frames), exporta job por job sin
#    tocar nada y verifica visibilidad idéntica + noise floor.
python scripts/cvat_roundtrip_gate.py --config configs/audit_cvat.yaml

# 4. Montar la auditoría real: proyecto nuevo, 1 task, jobs de 100 frames.
#    Escribe el mapa frame → job en cvat.frames_csv (se niega a pisarlo).
python scripts/cvat_audit.py montar --config configs/audit_cvat.yaml --project auditoria-v2

# 5. Anotar en la UI (ver "Cómo corregir"). Al terminar un job:
#    Menu → Change job state → completed. ESO es lo que lo marca como auditado.
python scripts/cvat_audit.py estado --config configs/audit_cvat.yaml    # progreso

# 6. Exportar SOLO los jobs completados (+ auditados.txt) y respaldar
python scripts/cvat_audit.py exportar --config configs/audit_cvat.yaml
python scripts/cvat_audit.py backup --config configs/audit_cvat.yaml --project-id <id>

# 7. Reconciliar → gold set + métrica de tesis
python -m src.audit.reconcile_corrections --config configs/audit_cvat.yaml

# 8. Aplicar → dataset corregido + folds + YAMLs (≈2 min: 74k frames)
python -m src.audit.apply_corrections --config configs/audit_cvat.yaml
```

Los pasos 6–8 se pueden correr con la auditoría **a medias**: cuentan solo los
jobs completados, y todo lo derivado se regenera de cero en cada corrida.

**Por qué "auditado" = job completed y no "presente en el export"** (verificado
el 2026-09-28 en v2.74.1): CVAT exporta todos los frames con pre-anotación, se
hayan revisado o no, y un frame revisado al que se le borraron todas las
personas **no aparece** en el export. La presencia no dice nada; el estado del
job sí.

Resultados del gate:

- 2026-09-05 (import a nivel proyecto): visibilidad idéntica (`v=0 ⇒ outside`,
  `v=1 ⇒ occluded`, `v=2 ⇒ visible`), noise floor **0.005 px**.
- 2026-09-28 (camino nuevo: task con `segment_size`, subida ordenada, import a
  nivel task, export por job): visibilidad idéntica, noise floor **0.005 px**,
  3 jobs para 5 frames, orden mezclado respetado por CVAT.

Ambos muy por debajo del `corregido_thr_px: 3.0` del YAML.

## Cómo corregir en la UI

- El mapeo de visibilidad que **hay que respetar** al anotar:
  - keypoint no visible / fuera de cuadro → marcarlo **outside** (`v=0`);
  - keypoint ocluido pero estimable → **occluded** (`v=1`);
  - keypoint visible → dejarlo normal (`v=2`).
- Revisar **todos** los frames del job antes de marcarlo completed: un frame no
  editado dentro de un job completed cuenta como "revisado y correcto".
- Frames sin pre-anotación (`sin_deteccion`): dibujar el skeleton de cero
  (tecla `N` con el label `person` seleccionado).
- Frames multi-persona: el esqueleto tiene que estar sobre el **paciente**; si
  está sobre otra persona, borrarlo y dibujar el del paciente.
- El **bbox no importa**: CVAT lo recalcula desde el skeleton al exportar y la
  reconciliación lo rehace con la convención GT (hull + 5 %). Corregir solo
  keypoints y visibilidad.

## Entregables

- `data/processed/uco_gold/` (reconcile) — gold set verificado por humano:
  `images/all/`, `labels/all/`, `gold_manifest.csv`, `gold_stems.txt` y
  `audit_report.json` (tasa de corrección, desplazamiento medio px/normalizado,
  desglose por motivo y postura).
- `data/processed/uco_pseudo_corrected/` (apply) — el pool pseudo con el gold
  aplicado, `provenance.csv`, folds y `apply_summary.json`; más los YAMLs
  `pose_dataset_pseudo_corrected_fold{k}.yaml`, `..._gold_fold{k}.yaml` (el
  subset humano del val de cada fold) y `..._coco_fold0.yaml` (mix v3, lo usa
  `configs/prune_yolov8n_deploy_v4.yaml`).
- **Held-out.** Con `aplicar.usar_en_train: true` (default) las correcciones
  entran al train de los folds donde su sujeto es train y se evalúan en el
  fold donde es val: el held-out es por **sujeto**. Con `false` se excluyen de
  todo train.
- DVC: versionar `uco_gold/`, `uco_pseudo_corrected/labels` + `folds/` y los
  CSV de `uco_audit/`. **No** las imágenes hardlinkeadas (DVC las duplicaría en
  su caché; `apply_corrections` las regenera).

## Troubleshooting

| Síntoma | Causa / solución |
|---|---|
| `400 "No such file were uploaded"` al importar por API | No pasar `filename` en la query: CVAT asume upload TUS previo. El upload de una sola request usa headers `Upload-Start` + `Upload-Finish` y el archivo en el body (`annotation_file` para anotaciones, `client_files[i]` para imágenes); ya resuelto en `scripts/cvat_client.py`. |
| `401` en `/api/server/annotation/formats` | El endpoint requiere auth (Basic con las credenciales del superuser). |
| Puerto 8080 ocupado | `Get-NetTCPConnection -State Listen` y liberar, o editar `ports:` del servicio `traefik` en `cvat/docker-compose.yml`. |
| El import "no matchea" keypoints con sublabels | El orden de los sublabels del skeleton no es el orden COCO: recrear el proyecto desde `cvat_skeleton_labels.json`. |
| Export sin `result_url` | El job asíncrono falló: revisar `GET /api/requests/<rq_id>` y los logs (`docker logs cvat_server`). |
| `montar` aborta con `FileExistsError` | `cvat.frames_csv` ya tiene el mapa de otra task. Pisarlo solo si esa task se descartó (`--sobrescribir-mapa`). |
| `exportar` aborta: "el job … no existe" | La task del mapa se borró o se recreó: el mapa es de otra task. |
| El orden de los frames en CVAT no es el del pack | `montar` lo avisa; el mapa sigue siendo correcto (sale de `data/meta`), solo se pierde el ciego del control. |
