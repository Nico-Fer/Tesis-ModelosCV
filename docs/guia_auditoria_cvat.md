# Guía de auditoría en CVAT: de cero a un job terminado

Guía práctica para levantar CVAT, entrar a la auditoría y corregir los
pseudo-labels. Los detalles técnicos (instalación desde cero, API, cómo se
armó la selección) están en [cvat_setup.md](cvat_setup.md) y el porqué del
diseño en [plan_cvat_correcciones.md](plan_cvat_correcciones.md).

Los atajos y menús citados se verificaron contra el CVAT desplegado
(v2.74.1) el 2026-09-28.

---

## 0. Qué vas a hacer

La auditoría ya está montada en CVAT: proyecto **`auditoria-v2`** (id 6),
task **`auditoria`** (id 4), **998 frames repartidos en 10 jobs** (del 6 al 15;
nueve de 100 frames y uno de 98). Cada frame trae el esqueleto que predijo
ViTPose como pre-anotación. Tu trabajo es:

1. revisar cada frame y corregir el esqueleto si está mal;
2. cuando terminás **todos** los frames de un job, marcarlo como terminado
   (*Finish the job*).

Un job terminado es la **única** señal de que sus frames fueron revisados. Un
frame que no tocaste dentro de un job terminado cuenta como "revisado y
correcto".

Los frames están **mezclados a propósito**: la mayoría son frames sospechosos
y algunos son de control (frames sanos). No mires `selection.csv` mientras
anotás. Si sabés cuáles son de control vas a revisarlos distinto, y la
comparación entre grupos, que es un resultado de la tesis, queda sesgada.

---

## 1. Levantar CVAT

**Requisitos:** Docker Desktop instalado y el repo clonado con la carpeta
`cvat/` (ya existe). Las credenciales están en `cvat_credentials.local`, en la
raíz del repo; ese archivo no se commitea.

1. **Abrí Docker Desktop** y esperá a que diga *Engine running*. Para
   verificarlo desde una terminal:
   ```bash
   docker info
   ```
   Si da error, Docker todavía no arrancó.

2. **Levantá CVAT** desde la raíz del repo:
   ```bash
   cd cvat
   docker compose up -d
   ```
   La primera vez descarga imágenes (unos 5 GB); después tarda segundos.

3. **Esperá a que responda.** El servidor tarda entre 30 s y 1 min en quedar
   listo. Abrí <http://localhost:8080>. Si ves la pantalla de login, está
   listo. Si la página no carga, esperá un poco y recargá. Para ver el estado
   de los contenedores:
   ```bash
   docker compose ps          # todos deberían estar "running"
   ```

4. **Iniciá sesión** con el usuario y la contraseña de
   `cvat_credentials.local` (`CVAT_USER` / `CVAT_PASS`).

5. **Comprobá que la auditoría sigue ahí.** Desde la raíz del repo (sirve el
   Python de `venv-torch` o cualquiera con `requests`, `pyyaml` y `opencv`):
   ```bash
   python scripts/cvat_audit.py estado --config configs/audit_cvat.yaml
   ```
   Tiene que listar los jobs 6 a 15 con su estado. Si dice que un job "NO
   EXISTE", los datos de CVAT se perdieron: ver
   [Problemas frecuentes](#9-problemas-frecuentes).

### Apagar CVAT al terminar

```bash
cd cvat
docker compose down        # apaga; las anotaciones quedan guardadas
```

> ⚠️ **Nunca** uses `docker compose down -v`. La `-v` borra los volúmenes, es
> decir, **todas las anotaciones** y la base de datos de CVAT.

---

## 2. Llegar a un job

1. En la barra superior entrá a **Projects** y abrí **`auditoria-v2`**.
2. Abrí la task **`auditoria`**. Abajo aparece la lista de **jobs** con su
   rango de frames y su estado.
3. Hacé click en un job (por ejemplo, *Job #6*) para abrir el editor.
   También podés ir directo a `http://localhost:8080/tasks/4/jobs/<id>`, con
   `<id>` entre 6 y 15.

Trabajá **un job por vez** y de principio a fin. El orden entre jobs no
importa.

---

## 3. (Recomendado) Practicar primero en un proyecto descartable

Antes de tocar la auditoría real conviene practicar en un proyecto aparte, de
5 frames, que después se puede tirar. Este comando lo crea (es el mismo que
se usa como verificación técnica del flujo):

```bash
python -m src.audit.pack_for_cvat --config configs/audit_cvat.yaml --limit 5
python scripts/cvat_roundtrip_gate.py --config configs/audit_cvat.yaml
```

Aparece un proyecto `gate-roundtrip-<fecha>` con jobs de 2 frames. Ahí podés
probar todo lo de las secciones 4 y 5 (mover puntos, *outside*, *occluded*,
dibujar, borrar, terminar el job) sin afectar nada. Cuando termines, borralo
desde la UI (Projects → menú de acciones de la tarjeta del proyecto → Delete).

---

## 4. La interfaz del editor

```
┌──────────────────────── barra superior ──────────────────────────┐
│ Menu │ Save │ Undo/Redo │  ◀◀ ◀ [nº de frame] ▶ ▶▶  │ ...       │
├────┬─────────────────────────────────────────────────┬──────────┤
│ he │                                                 │ Objects  │
│ rr │                                                 │ (lista   │
│ am │                  imagen (canvas)                │ de       │
│ ie │                                                 │ esquele- │
│ nt │                                                 │ tos y sus│
│ as │                                                 │ puntos)  │
└────┴─────────────────────────────────────────────────┴──────────┘
```

- **Barra superior:** *Menu* (opciones del job, incluido terminarlo), *Save*,
  deshacer/rehacer y la navegación entre frames.
- **Panel izquierdo:** herramientas. La que vas a usar es la de **esqueleto**
  (tooltip *Draw a skeleton*, al final del grupo de herramientas de dibujo).
  También está el botón para reencuadrar la imagen.
- **Canvas:** la imagen con el esqueleto encima.
- **Panel derecho (Objects):** una fila por esqueleto. Con la flecha de la
  fila se despliegan sus **17 puntos por nombre**, y cada punto tiene sus
  propios botones (*outside*, *occluded*, etc.; el tooltip dice cuál es cuál).

### Moverse

| Acción | Cómo |
|---|---|
| Frame siguiente / anterior | `F` / `D` |
| Saltar varios frames adelante / atrás | `V` / `C` |
| Zoom | rueda del mouse |
| Mover la imagen | arrastrar sobre una zona vacía del canvas |
| Ver todos los atajos | `F1` |
| Configuración | `F2` |

**Activá el autoguardado:** `F2` → pestaña *Workspace* → **Enable auto save**,
con un intervalo de pocos minutos. Igual podés guardar a mano en cualquier
momento con `Ctrl+S`.

---

## 5. Qué hacer en cada frame

Mirá el esqueleto y preguntate: **¿los puntos están sobre las articulaciones
correctas de la persona correcta?**

### 5.1 Los puntos: qué es cada uno

Son los 17 keypoints de COCO. **Izquierda y derecha son las de la persona, no
las tuyas**: si la persona mira a la cámara, su hombro izquierdo aparece a la
**derecha** de la imagen.

| Punto(s) | Dónde va |
|---|---|
| `nose` | punta de la nariz |
| `left_eye`, `right_eye` | centro de cada ojo |
| `left_ear`, `right_ear` | cada oreja (suelen estar ocluidas) |
| `left_shoulder`, `right_shoulder` | centro de la articulación del hombro |
| `left_elbow`, `right_elbow` | centro del codo |
| `left_wrist`, `right_wrist` | centro de la muñeca |
| `left_hip`, `right_hip` | centro de la articulación de la cadera (no el borde del cuerpo) |
| `left_knee`, `right_knee` | centro de la rodilla |
| `left_ankle`, `right_ankle` | tobillo (maléolo) |

Lo más importante son **las articulaciones del ejercicio**: caderas, rodillas
y tobillos en los ejercicios de pierna; hombros, codos y muñecas en los de
brazo. Corregí también los demás puntos si están claramente mal.

### 5.2 La regla más importante: si está bien, no lo toques

Un punto que se movió más de **3 px** cuenta como "corregido" en la métrica de
la tesis. Si retocás puntos que ya estaban bien, la tasa de corrección sube
por ruido tuyo y deja de medir los errores del modelo. Corregí solo lo que
**claramente** está mal; si dudás, dejalo como está.

### 5.3 Corregir un punto mal ubicado

Arrastralo con el mouse hasta la articulación correcta. Si el punto quedó
debajo de otro, hacé zoom.

### 5.4 Visibilidad: visible, *occluded* u *outside*

Cada punto tiene uno de tres estados. Para el entrenamiento, **lo que importa
es si el punto existe (visible u occluded) o no existe (outside)**; visible y
occluded se entrenan igual.

| Estado en CVAT | Cuándo | Cómo se marca |
|---|---|---|
| **visible** (normal) | se ve la articulación | estado por defecto |
| **occluded** | está tapada (andador, ropa, la otra pierna, la silla) **pero podés estimar dónde está** con razonable seguridad | mouse encima del punto + `Q` |
| **outside** | no se ve **y no podés estimar dónde está**, o está fuera de la imagen | mouse encima del punto + `O` |

- `Q` y `O` actúan sobre el punto que tenés **bajo el mouse**, que se resalta
  al pasar por encima. Volvé a pulsar la tecla para deshacer.
- Un punto en *outside* **desaparece del canvas**. Para recuperarlo, en el
  panel derecho desplegá el esqueleto, buscá el punto por nombre y desactivá
  su botón *outside*. Después arrastralo a su lugar.
- Si ViTPose dejó un punto en *outside* pero la articulación **sí se ve**,
  recuperalo así y ubicalo. Es de las correcciones más valiosas.

### 5.5 Frame sin esqueleto (el detector no vio a nadie)

Hay unos 150 frames así. Hay que dibujar el esqueleto de cero:

1. En el panel izquierdo, pasá el mouse por la herramienta de **esqueleto**
   (tooltip *Draw a skeleton*). Se abre el recuadro *Draw new skeleton*.
2. En *Label* elegí **`person`** y pulsá **Shape**.
3. Dibujá un rectángulo alrededor de la persona: un click en una esquina y
   otro en la opuesta. Aparece un esqueleto modelo dentro del rectángulo.
4. Arrastrá cada punto a su articulación y marcá *occluded* / *outside* lo que
   corresponda.
5. Para el siguiente frame vacío, `N` repite la herramienta con la misma
   configuración.

### 5.6 Esqueleto sobre la persona equivocada o de más

En algunos frames aparece otra persona (por ejemplo, un terapeuta). El
esqueleto tiene que estar **sobre el paciente**.

- Si el esqueleto está sobre otra persona: seleccionalo, borralo con `Supr`
  (`Del`) y dibujá el del paciente (5.5).
- Si hay un esqueleto de más: borralo con `Supr`.
- Si borraste algo por error: `Ctrl+Z`.

### 5.7 Lo que NO hace falta

- **El rectángulo (bbox) no importa.** Se recalcula solo a partir de los
  puntos. Corregí solo puntos y visibilidad.
- No uses *tracks*, *propagate* ni interpolación: son herramientas de video y
  acá cada frame es independiente.

---

## 6. Terminar un job

Cuando revisaste **todos** los frames del job:

1. Guardá (`Ctrl+S`).
2. **Menu → Finish the job** → *Continue*. Esto guarda y pone el job en estado
   *completed*.

Alternativa equivalente: **Menu → Change job state → completed**.

Si lo marcaste por error antes de terminar: **Menu → Change job state →
in progress**. Mientras no esté en *completed*, sus frames no cuentan.

Opcional, para la tesis: anotá la hora de inicio y fin de cada job. El paper
tiene pendiente el dato "horas humanas por frame auditado".

---

## 7. Guardar el trabajo fuera de CVAT (hacelo seguido)

Las anotaciones viven en volúmenes de Docker. Al final de cada sesión,
**respaldá el proyecto**:

```bash
python scripts/cvat_audit.py backup --config configs/audit_cvat.yaml --project-id 6
```

El respaldo queda en `data/processed/uco_audit/backups/proyecto6_<fecha>.zip` y
se puede restaurar en CVAT si algo se rompe.

Para ver el avance en cualquier momento:

```bash
python scripts/cvat_audit.py estado --config configs/audit_cvat.yaml
```

---

## 8. Cuando termines (o para ver resultados parciales)

Estos pasos se pueden correr aunque la auditoría esté a medias: solo cuentan
los jobs en *completed*, y todo se regenera de cero en cada corrida.

```bash
# 1. Exportar los jobs terminados (+ auditados.txt)
python scripts/cvat_audit.py exportar --config configs/audit_cvat.yaml

# 2. Reconciliar: métrica de la tesis + gold set
python -m src.audit.reconcile_corrections --config configs/audit_cvat.yaml

# 3. Aplicar: dataset corregido + folds (≈2 min)
python -m src.audit.apply_corrections --config configs/audit_cvat.yaml
```

- El paso 2 imprime la **tasa de corrección** y deja el reporte completo en
  `data/processed/uco_gold/audit_report.json` (con desglose por motivo y
  postura).
- El paso 3 deja listo `data/processed/uco_pseudo_corrected/` para entrenar el
  modelo v4 (`configs/prune_yolov8n_deploy_v4.yaml`); ver
  [plan_cvat_correcciones.md](plan_cvat_correcciones.md), fase F6.

---

## 9. Problemas frecuentes

| Problema | Qué hacer |
|---|---|
| <http://localhost:8080> no carga | ¿Docker Desktop está corriendo? `docker compose ps` dentro de `cvat/`. Si algún contenedor no está *running*, `docker compose up -d` de nuevo y esperá un minuto. Logs: `docker logs cvat_server`. |
| El puerto 8080 está ocupado | Ver [cvat_setup.md](cvat_setup.md), sección Troubleshooting. |
| Un punto desapareció | Lo pusiste en *outside*: recuperalo desde el panel derecho (5.4). |
| Moví o borré algo sin querer | `Ctrl+Z` (deshacer), `Ctrl+Shift+Z` (rehacer). |
| No puedo agarrar un punto | Hacé zoom; si otro objeto lo tapa, ocultá temporalmente el otro con `H` sobre él. |
| Borré un frame (`Alt+Supr`) | **No uses ese atajo.** Un frame borrado no cuenta como auditado; `exportar` avisa cuántos hay. |
| Terminé un job antes de tiempo | Menu → Change job state → in progress. |
| `estado` dice que los jobs "NO EXISTEN" | Se perdieron los datos de CVAT (por ejemplo, con `down -v`). Si hay backup, restauralo (Projects → menú del botón **+** → *Create from backup*). Si no, hay que volver a montar la auditoría: `cvat_audit.py montar --config configs/audit_cvat.yaml --project auditoria-v3 --sobrescribir-mapa`. Esto pierde lo anotado. |
| `exportar` falla con "el mapa es de otra task" | `data/processed/uco_audit/cvat_frames.csv` no corresponde al proyecto actual. No lo edites a mano; revisá qué proyecto se montó último. |

---

## Resumen de una página

```
PREPARAR     Docker Desktop → cd cvat && docker compose up -d → localhost:8080 → login
IR AL JOB    Projects → auditoria-v2 → task auditoria → Job #N
POR FRAME    ¿Puntos sobre las articulaciones correctas de la persona correcta?
               mal ubicado → arrastrar        tapado pero estimable → Q
               no se ve / fuera de imagen → O   frame vacío → herramienta esqueleto
               persona equivocada → Supr + dibujar
               bien → NO TOCAR
NAVEGAR      F siguiente · D anterior · Ctrl+S guardar · Ctrl+Z deshacer · F1 atajos
TERMINAR     Revisar TODOS los frames → Menu → Finish the job
AL SALIR     python scripts/cvat_audit.py backup --config configs/audit_cvat.yaml --project-id 6
             cd cvat && docker compose down      (NUNCA -v)
```
