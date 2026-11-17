# Videos fuente de las referencias DTW

Clips de los que sale cada `references/{exerciseId}.json`. **Binarios: van a DVC,
no a git** (`.gitignore` ignora `data/references/*`, igual que `data/raw` y
`data/processed`). Van acá y no en `src/references/` porque ese es el paquete
Python del generador — el código se commitea, los videos no.

```
data/references/
  README.md            <- este archivo (provenance)
  videos/              <- clips fuente, un archivo por ejercicio
```

Salida del generador: `references/{exerciseId}.json` (ver `output_dir` en
`configs/references.yaml`), apuntable a `app/src/main/assets/references/`.

## Regenerar

```bash
# venv-torch, desde la raíz del repo
python -m src.references --config configs/references.yaml --multi
```

`--multi` es obligatorio con estos clips: cada uno trae varias repeticiones
seguidas, así que el generador las detecta todas y se queda con la **mediana por
medoide DTW**, no con la primera.

El backend es el **`.tflite` int8 que corre la app**, no un `.pt`: así la referencia
y la ejecución en vivo comparten el sesgo del modelo y la distancia DTW mide al
usuario, no la diferencia entre dos modelos.

Antes de cablear un clip nuevo en la config, **medirlo** con el probe (corre el
mismo pipeline sin emitir JSON y cachea keypoints en `<video>.probe_kpts.npz`,
ignorado por git junto con el resto de `data/references/*`):

```bash
python scripts/probe_reference_angles.py --config configs/references.yaml \
    --cache --video data/references/videos/<clip> [--reps <dim>] [--series <dim>]
```

De ese calibrado salieron dos features del generador (2026-08): **`trimStartS`/
`trimEndS`** por ejercicio (clips que no empiezan/terminan en reposo envenenan
el baseline) y **overrides por ejercicio de `cycle:`/`multi:`** (la excursión
varía por gesto — sentadilla parcial ~0.3 rad, extensión de rodilla ~0.6 — y
una prominencia global no puede servir a todos).

El filename de salida va en **minúscula** (`c1000000-…json`, contrato de assets
de la app); el campo `exerciseId` adentro conserva la `C` mayúscula.

## Catálogo

| exerciseId | Ejercicio (backend) | Video | Origen |
|---|---|---|---|
| `C1000000-0000-0000-0000-000000000001` | Elevación lateral de hombros | `videos/C1000000-0000-0000-0000-000000000001.mp4` | Grabación propia, frontal |
| `C1000000-0000-0000-0000-000000000002` | Sentadilla parcial | `videos/c1000000-0000-0000-0000-000000000002.MOV` | Grabación propia v2 (2026-08), frontal, bilateral |
| `C1000000-0000-0000-0000-000000000005` | Elevación de talones | `videos/c1000000-0000-0000-0000-000000000005.MOV` | Grabación propia, frontal |
| `C1000000-0000-0000-0000-000000000006` | Extensión de rodillas | `videos/c1000000-0000-0000-0000-000000000006_SinManos.MOV` | Grabación propia, frontal (ganó vs `_CpnManos`, ver abajo) |

Los clips son de celular con la orientación como **metadata del contenedor**
(90°). cv2 la ignora salvo que se le pida, así que `extract.py` activa
`CAP_PROP_ORIENTATION_AUTO`; sin eso el detector recibe una persona acostada,
que es su peor caso. Rotar es seguro — el ángulo es signado vía
`atan2(cross, dot)` y una rotación preserva ambos productos — pero **espejar no
lo es**, porque invierte el signo.

### C1000000-…-0001 — Elevación lateral de hombros

Sentado, cámara frontal, **bilateral**, brazos hasta arriba de la cabeza.
Reemplaza al clip UCO s26/e09/cam0, que era unilateral izquierdo y de perfil; por
eso ahora se declaran **dos** dimensiones (`hombro_izq`, `hombro_der`) en vez de
una. El `exerciseId` no cambió.

Resultado medido del generador (`--multi`):

- 0/342 cuadros sin persona detectada; **0 dimensiones inválidas** en el ciclo.
- 2 reps detectadas, 0 outliers; ciclo elegido = cuadros 39–161 (T=123, 4.10 s).
- Conductor `hombro_izq`: reposo −1.307 rad → pico **−2.709 rad**, excursión
  1.40 rad. `hombro_der` recorre 1.558 → 3.089 en el mismo ciclo.
- Las dos reps miden 1.53 y 1.49 rad de amplitud: consistentes entre sí.

**Sin ángulo de codo, a propósito.** Tiene excursión real en este clip (1.14 rad,
el sujeto flexiona los codos en el reposo), pero no es parte del gesto prescrito
y exigir esa flexión penalizaría una ejecución correcta.

### C1000000-…-0002 — Sentadilla parcial (v2, 2026-08)

De pie, cámara frontal, 4 reps. Reemplaza al clip v1 (`.mp4`), que era usable
sólo del lado izquierdo por escorzo; en esta toma **el lado derecho sí
acompaña** (~0.2 rad, picos alineados a los izquierdos, medido con el probe),
así que la referencia es **bilateral**: `cadera_izq/der` + `rodilla_izq/der`.
Los imbalances entre lados son parte del gesto real y se aceptan.

Resultado medido del generador (`--multi`):

- 0/495 cuadros sin persona (post-trim); **0 dimensiones inválidas** en el ciclo.
- 4 reps detectadas, 0 outliers; ciclo elegido = cuadros 174–227 (T=54, 1.80 s).
- Conductor `cadera_izq`: reposo 2.914 → pico 2.644 rad (excursión 0.27; las 4
  reps miden 0.29–0.40 respecto del baseline de reposo).

Dos particularidades del clip, medidas con el probe:

- **Arranca EN sentadilla** (hip_y ~1200 px vs ~1005 de pie hasta el cuadro
  ~55): sin recorte el baseline de reposo queda envenenado y la señal parece
  una onda cuadrada alrededor de un cero falso. De ahí `trimStartS: 1.9` —
  equivale a recortar el video, no inventa datos.
- La sentadilla es más superficial que en v1 (0.27 vs 0.49 rad): el
  `prominence_rad` global (0.35) no detectaría **ninguna** rep, por eso la
  entrada overridea `prominence_rad: 0.15` y `valley_tol_rad: 0.10`.

La trampa geométrica de la vista frontal documentada en v1 sigue vigente: de
pie toda cadena de 3 puntos vale ~π, el ángulo signado vive sobre ±π y el
pipeline (y la app) deben comparar con wraparound.

### C1000000-…-0005 — Elevación de talones

De pie, cámara frontal, 7 reps.

**Advertencia epistemológica, medida:** COCO-17 no tiene pies/dedos y una
elevación de talones es casi traslación pura del cuerpo — los ángulos entre
articulaciones son invariantes a traslación, así que «no debería» haber señal.
La hay (`rodilla_izq` 0.36 rad, `cadera_izq` 0.33, 7 reps alineadas entre ambas
dims) porque el **modelo corre los keypoints** al rotar el pie: es un artefacto
del detector, no cinemática real. Vale como referencia porque la app usa el
MISMO `.tflite` y comparte el sesgo — pero **regenerar SÍ o SÍ si cambia el
modelo de la app**, acá más que en ningún otro ejercicio.

Resultado medido del generador (`--multi`, overrides `prominence_rad: 0.18`,
`valley_tol_rad: 0.12`):

- 0/745 cuadros sin persona; **0 dimensiones inválidas** en el ciclo.
- 7 reps detectadas (amplitudes 0.19–0.36 rad), 0 outliers; ciclo elegido =
  cuadros 449–509 (T=61, 2.03 s).
- Conductor `rodilla_izq`: reposo −2.784 → pico −2.618 rad.

Lado derecho excluido (0.17–0.18 rad, mitad que el izquierdo: mismo escorzo por
vista girada que la sentadilla v1). Tronco/hombros excluidos: la parte superior
del cuerpo no es parte del gesto prescrito.

### C1000000-…-0006 — Extensión de rodillas

Sentado, cámara frontal, pierna izquierda. Se grabaron **dos tomas** — manos
sobre la rodilla (`_CpnManos`) y manos atrás del cuerpo (`_SinManos`) — y se
decidió por métricas del probe, no por intuición. **Ganó `_SinManos`:**

| métrica | `_SinManos` | `_CpnManos` |
|---|---|---|
| amplitud de reps | 0.592–0.621 rad (CV ~2.5%) | 0.49–0.68 rad (CV ~17%) |
| jitter cuadro a cuadro | 0.019 rad | 0.024 rad |
| estructura | reposo→reps→reposo limpio | ~21 s de preámbulo en otra postura + corta sin volver al reposo |
| detección | 0 sin persona, conf 0.99 | 0 sin persona, conf 0.99 |

Dato útil: las manos sobre la rodilla **no** confundieron al detector (empate
en confianza y detección); la toma perdió por estructura y consistencia entre
reps. Con `trimStartS: 21` `_CpnManos` era rescatable, pero la otra toma es
mejor sin cirugía.

Resultado medido del generador (`--multi`, overrides `prominence_rad: 0.30`,
`valley_tol_rad: 0.28` — los valles entre reps no bajan de ~0.21–0.25 porque
la pierna no vuelve a colgar del todo):

- 0/908 cuadros sin persona; **0 dimensiones inválidas** en el ciclo.
- 3 reps detectadas, 0 outliers; ciclo elegido = cuadros 584–620 (T=37, 1.23 s).
- Conductor `rodilla_izq`: cruza ±π dentro del ciclo (reposo −3.040 → pico
  +2.853): otro caso real que exige wraparound en la app.

**Una sola dimensión a propósito** (`rodilla_izq`): el gesto prescrito es
extender la rodilla; una dim de cadera/tronco sólo registraría el balanceo
compensatorio y penalizaría una ejecución quieta y correcta (mismo criterio que
el codo en 0001). La pierna derecha no se mueve en el clip.

## Bug encontrado al generar estas referencias

`cycle.py` suavizaba el conductor con `moving_average` (promedio **crudo**) antes
de detectar el ciclo, pese a que `angles.py` define `smooth_wrapped` justo para
esto y advierte del riesgo. Con una señal cerca de ±π el promedio mezcla +3.11
con −3.12 y da ~0, o sea **inventa una excursión de π rad donde el miembro no se
movió**: la señal de la sentadilla aparecía como una onda cuadrada de ±2.5 rad.
El clip UCO nunca lo destapó porque su señal vivía lejos de ±π. Arreglado, con
dos tests de regresión en `tests/test_reference_cycle.py`.

## Los otros dos ejercicios del backend

UCO no los cubre — no hay clip que copiar, hay que grabarlos:

| exerciseId | Ejercicio | Por qué no está en UCO |
|---|---|---|
| `C1000000-…-0003` | Extensión lumbar en decúbito prono | UCO filma **supino**, nunca prono. |
| `C1000000-…-0004` | Flexión de rodilla con banda elástica | Lo más parecido es la **extensión de rodilla sentado** (ejercicios 01/02 izq, 05/06 der), misma cadena `HIP → KNEE → ANKLE`. Mejor candidato medido: sujeto **8**, ejercicio **01**, **cam1** (49.6 s, 4 reps limpias, amplitud 1.32 rad ≈ 76°, reposo plano en ambos extremos). Ojo: el movimiento arranca **extendiendo**, no flexionando, y no hay banda elástica. |
