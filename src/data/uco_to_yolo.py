"""Conversor del dataset UCO al formato YOLOv8-pose de Ultralytics.

UCO entrega video + anotaciones de pose 2D/3D en JSON/txt. Ultralytics necesita,
en cambio:

* imágenes sueltas en un pool plano ``images/all`` (los folds del GroupKFold
  por sujeto se arman como listas ``.txt`` en ``folds/``, sin duplicar JPEG);
* un label ``.txt`` por imagen en formato **YOLO-pose** (una línea por persona)::

      class  cx cy w h  px1 py1 v1  px2 py2 v2  …  pxK pyK vK

  con bbox y keypoints **normalizados 0–1** respecto al tamaño de la imagen, y
  ``v ∈ {0,1,2}`` (0 = no anotado, 1 = anotado ocluido, 2 = visible);
* un **dataset YAML** con ``path``, ``train``, ``val``, ``kpt_shape``,
  ``flip_idx`` y ``names`` — el que ``configs/baseline_yolov8n.yaml`` espera en
  ``data: data/processed/pose_dataset.yaml``.

Como la tesis compara contra modelos COCO-17 (YOLO=17, MoveNet=17,
BlazePose 33→17), las keypoints de UCO se **remapean a COCO-17** para que la
comparación sea homogénea. El esqueleto COCO canónico vive en
:mod:`src.inference.keypoint_schema` (única fuente de verdad, reutilizada acá).

UCO es un dataset de **rehabilitación**: cada clip anota sólo los ~3 joints
relevantes al ejercicio (p.ej. ``l_hip/l_knee/l_ankle``) y **nunca la cara**. El
vocabulario completo son 12 joints corporales que mapean 1:1 a los slots COCO
5–16 (ver :data:`UCO_TO_COCO`); los slots de cara (0–4) salen siempre con
``visibility=0``. La fuente de labels por frame son los ``camN_p2d.txt`` (una
fila por frame, ``id`` contiguo ``0..N-1``, coords en píxeles): a diferencia del
``dataset_2d.json`` — que sólo anota estos ejercicios para el **sujeto 0** —
cubren los **27 sujetos**. Como el ``.txt`` no trae nombres de joint, el orden de
columnas por ejercicio se aprende del JSON (:func:`build_exercise_joint_order`),
donde el sujeto 0 los anota con nombre y coincide byte a byte con su ``.txt``.

Pipeline de integración::

    src.data.uco_extract   # 1. reensambla + extrae el zip partido (no aplica al sample)
    src.data.uco_to_yolo   # 2. (este módulo) extrae frames + parsea anotaciones → labels + YAML

``convert()`` hace **extracción integrada**: abre los ``.mp4`` y escribe cada
imagen junto a su label, garantizando pairing 1:1. :mod:`src.data.uco_frames`
queda como utilidad opcional para volcar frames crudos, fuera del camino crítico.

El pipeline es **data-driven**: procesa todo ``(folder, exercise)`` con video y
orden de joints conocido; el resto se saltea con aviso.

Sólo depende de ``cv2`` (en requirements-base) + el esquema COCO (sin frameworks ML).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.inference.keypoint_schema import COCO_KEYPOINT_NAMES, NUM_COCO_KEYPOINTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapeo de keypoints UCO → COCO-17 (nombre UCO → nombre COCO).
#
# UCO anota sólo joints corporales (12 distintos en total, según el ejercicio);
# nunca cara. Los nombres mapean 1:1 a los slots COCO 5–16. Los joints COCO sin
# correspondencia en UCO (nariz, ojos, orejas: 0–4) salen con visibility=0.
# ---------------------------------------------------------------------------
UCO_TO_COCO: dict[str, str] = {
    "l_shoulder": "left_shoulder",
    "r_shoulder": "right_shoulder",
    "l_elbow": "left_elbow",
    "r_elbow": "right_elbow",
    "l_wrist": "left_wrist",
    "r_wrist": "right_wrist",
    "l_hip": "left_hip",
    "r_hip": "right_hip",
    "l_knee": "left_knee",
    "r_knee": "right_knee",
    "l_ankle": "left_ankle",
    "r_ankle": "right_ankle",
}

# Slot COCO (índice 0..16) por nombre de joint UCO, precomputado una vez.
_UCO_TO_COCO_IDX: dict[str, int] = {
    uco: COCO_KEYPOINT_NAMES.index(coco) for uco, coco in UCO_TO_COCO.items()
}

# flip_idx para data augmentation (espejado horizontal): por cada keypoint, el
# índice de su contraparte L/R. Derivado del orden COCO canónico.
COCO_FLIP_IDX: list[int] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


@dataclass(frozen=True)
class PersonAnnotation:
    """Anotación de una persona en un frame, ya en COCO-17 y en píxeles.

    keypoints: lista de 17 tuplas ``(x_px, y_px, v)`` en orden COCO.
    """

    image_path: Path
    img_w: int
    img_h: int
    keypoints: list[tuple[float, float, int]]


@dataclass(frozen=True)
class StemParts:
    """Componentes de un ``stem`` del pool plano (ver :func:`parse_stem`)."""

    subject: str
    exercise: str
    cam: str
    frame: int


# Formato del stem, definido UNA sola vez. `make_stem` lo escribe y `parse_stem`
# lo lee; ningún otro módulo debe partir el nombre a mano.
_STEM_RE = re.compile(r"^(?P<subject>[^_]+)_(?P<exercise>[^_]+)_cam(?P<cam>[^_]+)_(?P<frame>\d+)$")


def make_stem(subject: str, exercise: str, cam: str, frame_idx: int) -> str:
    """Nombre base (sin extensión) de un frame en el pool plano.

    El stem es el **contrato** que une ``images/all/<stem>.jpg`` con
    ``labels/all/<stem>.txt`` y con el sujeto usado para el GroupKFold; por eso
    su formato vive acá y no interpolado en el sitio de uso.

    Args:
        subject: identificador de sujeto (``folder`` de UCO, p.ej. ``"11"``).
        exercise: identificador de ejercicio (p.ej. ``"01"``).
        cam: número de cámara sin el prefijo ``cam`` (p.ej. ``"0"``).
        frame_idx: índice de frame dentro del video.

    Returns:
        ``"{subject}_{exercise}_cam{cam}_{frame_idx:06d}"``.
    """
    return f"{subject}_{exercise}_cam{cam}_{frame_idx:06d}"


def parse_stem(stem: str) -> StemParts:
    """Inversa de :func:`make_stem`: recupera sujeto/ejercicio/cámara/frame.

    Es la **única** fuente de estos campos para cualquier consumidor del pool
    (agrupamiento por sujeto para folds, columnas de manifests, filtros por
    ejercicio). Derivarlos por un ``split`` ad-hoc en otro módulo abre la puerta
    a que dos consumidores discrepen en silencio.

    Args:
        stem: nombre base sin extensión, tal como lo produce :func:`make_stem`.

    Returns:
        Los componentes del stem.

    Raises:
        ValueError: si ``stem`` no respeta el formato (falla ruidosamente en vez
            de devolver un sujeto inventado).
    """
    match = _STEM_RE.match(stem)
    if match is None:
        raise ValueError(
            f"El stem {stem!r} no respeta el formato "
            f"'{{sujeto}}_{{ejercicio}}_cam{{N}}_{{frame:06d}}'."
        )
    return StemParts(
        subject=match["subject"],
        exercise=match["exercise"],
        cam=match["cam"],
        frame=int(match["frame"]),
    )


def load_annotations_2d(dataset_root: str | Path) -> dict[tuple[str, str], dict]:
    """Carga ``dataset_2d.json`` y lo indexa por ``(folder, exercise)``.

    Args:
        dataset_root: raíz del dataset (contiene ``dataset_2d.json``).

    Returns:
        Mapa ``(folder, exercise) -> entrada`` (cada entrada trae ``cameras``,
        ``side``, ``position``, etc.). Indexar acá evita rescanear la lista de
        ``data`` por cada clip.
    """
    path = Path(dataset_root) / "dataset_2d.json"
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return {(e["folder"], e["exercise"]): e for e in payload["data"]}


def build_exercise_joint_order(ann_index: dict[tuple[str, str], dict]) -> dict[str, list[str]]:
    """De cada ejercicio, el orden de joints tal como aparece en ``dataset_2d.json``.

    Los ``camN_p2d.txt`` traen sólo números (una fila por frame, columnas ``x,y``
    por joint) **sin nombres**: la identidad y el orden de las columnas de un
    ejercicio se aprenden del JSON, donde el sujeto 0 anota todos los ejercicios
    con nombre. Se asume orden de joints estable por ejercicio — verificado
    contra el sujeto 0, donde el ``.txt`` coincide byte a byte con el JSON.

    Args:
        ann_index: índice de :func:`load_annotations_2d`.

    Returns:
        Mapa ``exercise -> [nombre_joint, …]`` en el orden de columnas del ``.txt``.
    """
    order: dict[str, list[str]] = {}
    for (_, exercise), entry in ann_index.items():
        if exercise in order:
            continue
        for cam_data in entry["cameras"].values():
            frames = cam_data.get("frames")
            if frames:
                order[exercise] = list(frames[0]["joints"].keys())
                break
    return order


def parse_p2d_file(path: Path, joint_names: list[str]) -> dict[int, dict[str, dict[str, str]]]:
    """Parsea un ``camN_p2d.txt`` a la misma estructura que un frame del JSON.

    Cada línea ``i`` es el frame de ``id=i`` (contiguo desde 0) y trae
    ``len(joint_names)`` pares ``x y`` en el orden de :func:`build_exercise_joint_order`.
    Las coords ``≤ 0`` se tratan como **fuera de cuadro** y el joint se omite
    (queda no anotado, ``v=0``). Las líneas con un número de columnas inesperado
    se saltean con aviso.

    Args:
        path: ruta del ``camN_p2d.txt``.
        joint_names: nombres de joint UCO en el orden de columnas del ejercicio.

    Returns:
        Mapa ``frame_id -> {nombre_joint: {"x": str, "y": str}}`` apto para :func:`to_coco17`.
    """
    expected = 2 * len(joint_names)
    frames: dict[int, dict[str, dict[str, str]]] = {}
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            vals = line.split()
            if not vals:
                continue
            if len(vals) != expected:
                logger.warning(
                    "%s línea %d: %d columnas (esperadas %d); saltada.",
                    path,
                    idx,
                    len(vals),
                    expected,
                )
                continue
            joints: dict[str, dict[str, str]] = {}
            for j, name in enumerate(joint_names):
                x, y = vals[2 * j], vals[2 * j + 1]
                if float(x) <= 0 or float(y) <= 0:  # fuera de cuadro → no visible
                    continue
                joints[name] = {"x": x, "y": y}
            frames[idx] = joints
    return frames


def to_coco17(joints: dict[str, dict[str, str]]) -> list[tuple[float, float, int]]:
    """Remapea los joints UCO de un frame a las 17 keypoints COCO (en píxeles).

    UCO entrega ``joints`` como ``{nombre: {"x": str, "y": str}}`` con sólo los
    joints anotados para ese ejercicio (y sin flag de visibilidad). Cada joint
    presente cae en su slot COCO con ``v=2``; los slots sin dato (incl. toda la
    cara) quedan en ``(0.0, 0.0, 0)``.

    Args:
        joints: dict ``joints`` de un frame de ``dataset_2d.json``.

    Returns:
        Lista de 17 tuplas ``(x_px, y_px, v)`` en orden COCO.
    """
    coco: list[tuple[float, float, int]] = [(0.0, 0.0, 0)] * NUM_COCO_KEYPOINTS
    for name, xy in joints.items():
        idx = _UCO_TO_COCO_IDX.get(name)
        if idx is None:
            logger.warning("Joint UCO desconocido '%s' ignorado.", name)
            continue
        coco[idx] = (float(xy["x"]), float(xy["y"]), 2)
    return coco


def yolo_label_line(ann: PersonAnnotation) -> str:
    """Serializa una :class:`PersonAnnotation` a una línea de label YOLO-pose.

    El bbox se calcula como la caja envolvente de los keypoints visibles (con
    un pequeño margen), normalizado a 0–1. Una sola clase (``0`` = "person").

    Args:
        ann: anotación de la persona en COCO-17 y píxeles.

    Returns:
        Línea de texto ``"0 cx cy w h px1 py1 v1 …"`` normalizada.
    """
    w, h = ann.img_w, ann.img_h
    vis = [(x, y) for x, y, v in ann.keypoints if v > 0]
    if not vis:
        # Sin keypoints visibles: bbox que cubre toda la imagen como fallback.
        cx = cy = 0.5
        bw = bh = 1.0
    else:
        xs = [x for x, _ in vis]
        ys = [y for _, y in vis]
        margin = 0.05  # 5 % del rango como margen alrededor de los keypoints.
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        mx = (x1 - x0) * margin
        my = (y1 - y0) * margin
        x0, x1 = max(0.0, x0 - mx), min(float(w), x1 + mx)
        y0, y1 = max(0.0, y0 - my), min(float(h), y1 + my)
        cx = ((x0 + x1) / 2) / w
        cy = ((y0 + y1) / 2) / h
        bw = (x1 - x0) / w
        bh = (y1 - y0) / h

    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for x, y, v in ann.keypoints:
        parts += [f"{x / w:.6f}", f"{y / h:.6f}", str(v)]
    return " ".join(parts)


def assign_subject_folds(subjects: list[str], n_folds: int = 5, seed: int = 42) -> dict[str, int]:
    """Reparte los sujetos en ``n_folds`` grupos para GroupKFold **por sujeto**.

    Agrupar por sujeto (no por frame) evita data leakage: frames del mismo
    sujeto/clip son casi idénticos, así que ninguno debe caer en train y val a la
    vez. Cada sujeto entero pertenece a un único fold; en el fold ``k`` ese grupo
    es el set de **validación** y los otros ``n_folds-1`` grupos son **train**.

    El reparto es round-robin sobre los sujetos barajados con ``seed``: mantiene
    los folds balanceados en cantidad de sujetos y es reproducible.

    Args:
        subjects: identificadores de sujeto (p.ej. ``["0", "1", …]``); se deduplican.
        n_folds: cantidad de folds (k).
        seed: semilla del shuffle para reproducibilidad.

    Returns:
        Dict ``{sujeto: fold_idx}`` con ``fold_idx ∈ 0..n_folds-1``.

    Raises:
        ValueError: si ``n_folds < 2`` o hay menos sujetos que folds.
    """
    uniq = sorted(set(subjects))
    if n_folds < 2:
        raise ValueError(f"n_folds debe ser ≥ 2 (recibido {n_folds}).")
    if len(uniq) < n_folds:
        raise ValueError(
            f"{len(uniq)} sujetos < {n_folds} folds; no alcanza para GroupKFold por sujeto."
        )
    rng = random.Random(seed)
    rng.shuffle(uniq)
    return {subj: i % n_folds for i, subj in enumerate(uniq)}


def write_dataset_yaml(
    output_yaml: str | Path,
    dataset_path: str | Path,
    train_rel: str = "images/train",
    val_rel: str = "images/val",
) -> Path:
    """Escribe el dataset YAML de Ultralytics para YOLOv8-pose.

    Args:
        output_yaml: ruta del YAML a generar (p.ej. ``data/processed/pose_dataset.yaml``).
        dataset_path: ``path`` raíz del dataset (las rutas train/val son relativas a él).
        train_rel, val_rel: subrutas de imágenes de train y val.

    Returns:
        La ruta ``output_yaml``.
    """
    import yaml  # noqa: PLC0415

    spec = {
        "path": str(dataset_path),
        "train": train_rel,
        "val": val_rel,
        "kpt_shape": [NUM_COCO_KEYPOINTS, 3],  # 17 keypoints × (x, y, v)
        "flip_idx": COCO_FLIP_IDX,
        "names": {0: "person"},
    }
    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, allow_unicode=True)
    return output_yaml


def _reset_output_dirs(output_root: Path) -> None:
    """Borra y recrea el pool plano ``images/all`` + ``labels/all`` y ``folds/``.

    Con GroupKFold cada frame se extrae **una sola vez** a un pool plano; los
    folds son manifiestos de texto livianos (``folds/fold{k}_{train,val}.txt``)
    sobre ese pool, así los ~200 GB de JPEG no se duplican k veces.

    El output es regenerable (DVC/gitignored); limpiar evita mezclar corridas.
    Se eliminan también los ``*.cache`` de Ultralytics (uno por manifiesto de
    fold, p.ej. ``folds/fold0_train.cache``) y cualquier layout viejo
    (``images/train`` etc.), que de quedar obsoletos harían que YOLO leyera
    labels de una corrida anterior.
    """
    for sub in ("images", "labels"):
        d = output_root / sub
        if d.exists():
            shutil.rmtree(d)
        (d / "all").mkdir(parents=True, exist_ok=True)
    folds_dir = output_root / "folds"
    if folds_dir.exists():
        shutil.rmtree(folds_dir)
    folds_dir.mkdir(parents=True, exist_ok=True)
    for cache in output_root.rglob("*.cache"):
        cache.unlink()


def _extract_camera(
    mp4_path: Path,
    folder: str,
    exercise: str,
    cam: str,
    frames_by_id: dict[int, dict],
    output_root: Path,
    *,
    stride: int,
) -> list[str]:
    """Vuelca los frames anotados de un ``.mp4`` con su label YOLO-pose.

    Itera el video secuencialmente; para cada índice múltiplo de ``stride`` que
    tenga anotación (``id`` en ``frames_by_id``) escribe la imagen y su label
    pareados en el **pool plano** ``images/all`` + ``labels/all``. La asignación
    a folds no ocurre acá: se resuelve después por sujeto (:func:`write_fold_manifests`).

    Returns:
        Lista de ``stem`` escritos (nombre base sin extensión), en orden.
    """
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(mp4_path))
    written: list[str] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0 and idx in frames_by_id:
                h, w = frame.shape[:2]
                stem = make_stem(folder, exercise, cam, idx)
                img_path = output_root / "images" / "all" / f"{stem}.jpg"
                lbl_path = output_root / "labels" / "all" / f"{stem}.txt"
                cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                ann = PersonAnnotation(img_path, w, h, to_coco17(frames_by_id[idx]))
                lbl_path.write_text(yolo_label_line(ann) + "\n", encoding="utf-8")
                written.append(stem)
            idx += 1
    finally:
        cap.release()
    return written


def write_fold_manifests(
    output_root: Path,
    output_yaml: Path,
    stems_by_subject: dict[str, list[str]],
    *,
    n_folds: int,
    seed: int,
) -> list[Path]:
    """Genera los manifiestos + YAMLs de GroupKFold sobre el pool plano ya extraído.

    Cada frame se escribió una sola vez en ``images/all``; acá se reparten los
    **sujetos** en ``n_folds`` grupos (:func:`assign_subject_folds`) y, por fold,
    se emiten dos listas de rutas de imágenes (``folds/fold{k}_{train,val}.txt``)
    más un dataset YAML que las referencia. Ultralytics acepta ``train``/``val``
    como un ``.txt`` de rutas y deriva el label reemplazando ``images`` por
    ``labels`` en cada ruta — por eso el pool es ``images/all`` ↔ ``labels/all``.

    Las rutas de los ``.txt`` son **absolutas** para no depender del CWD.

    Returns:
        Las rutas de los ``n_folds`` YAMLs generados (``…_fold0`` … ``…_fold{n-1}``).
    """
    folds = assign_subject_folds(list(stems_by_subject), n_folds, seed)
    images_all = output_root.resolve() / "images" / "all"
    yaml_paths: list[Path] = []
    for k in range(n_folds):
        train_lines: list[str] = []
        val_lines: list[str] = []
        for subj, stems in stems_by_subject.items():
            target = val_lines if folds[subj] == k else train_lines
            target.extend(str(images_all / f"{stem}.jpg") for stem in stems)
        (output_root / "folds" / f"fold{k}_train.txt").write_text(
            "\n".join(sorted(train_lines)) + "\n", encoding="utf-8"
        )
        (output_root / "folds" / f"fold{k}_val.txt").write_text(
            "\n".join(sorted(val_lines)) + "\n", encoding="utf-8"
        )
        fold_yaml = output_yaml.with_name(f"{output_yaml.stem}_fold{k}{output_yaml.suffix}")
        yaml_paths.append(
            write_dataset_yaml(
                fold_yaml,
                output_root.resolve(),
                train_rel=f"folds/fold{k}_train.txt",
                val_rel=f"folds/fold{k}_val.txt",
            )
        )
    val_counts = {k: sum(folds[s] == k for s in folds) for k in range(n_folds)}
    logger.info("Sujetos por fold (val): %s", val_counts)
    return yaml_paths


def convert(
    dataset_root: str | Path,
    output_root: str | Path,
    output_yaml: str | Path,
    *,
    stride: int = 30,
    n_folds: int = 5,
    seed: int = 42,
) -> list[Path]:
    """Orquesta la conversión completa UCO → YOLOv8-pose con GroupKFold por sujeto.

    Pasos:
        1. Del ``dataset_2d.json`` aprende el orden de joints por ejercicio
           (:func:`build_exercise_joint_order`).
        2. Limpia el output y recorre ``clips_mp4/*/*``.
        3. Para cada cámara con video, lee su ``camN_p2d.txt`` — la fuente que
           cubre los **27 sujetos** (el JSON sólo los anota para el sujeto 0) — y
           vuelca frames (1 de cada ``stride``) con su label COCO-17 al **pool
           plano** ``images/all`` + ``labels/all`` (una sola copia).
        4. Reparte los sujetos en ``n_folds`` folds y genera, por fold, los
           manifiestos train/val + su dataset YAML (:func:`_write_fold_manifests`).

    Es **data-driven**: procesa todo ejercicio con video y orden de joints conocido;
    el resto se saltea con aviso. Agrupar por sujeto evita data leakage.

    Args:
        dataset_root: raíz del dataset UCO (contiene ``clips_mp4/`` y ``dataset_2d.json``).
        output_root: raíz del dataset YOLO de salida (tendrá ``images/``, ``labels/``, ``folds/``).
        output_yaml: ruta base del YAML; se le agrega ``_fold{k}`` por fold.
        stride: submuestreo temporal (stride=30 ⇒ 1 frame de cada 30).
        n_folds: cantidad de folds (k) del GroupKFold por sujeto.
        seed: semilla del reparto de folds.

    Returns:
        Las rutas de los ``n_folds`` dataset YAML generados.

    Raises:
        RuntimeError: si no se escribió ningún frame (sin anotaciones/videos válidos).
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    output_yaml = Path(output_yaml)
    ann_index = load_annotations_2d(dataset_root)
    exercise_joints = build_exercise_joint_order(ann_index)
    _reset_output_dirs(output_root)

    clips_dir = dataset_root / "clips_mp4"
    stems_by_subject: dict[str, list[str]] = {}
    total_written = 0
    skipped_clips = 0
    for action_dir in sorted(clips_dir.glob("*/*")):
        folder, exercise = action_dir.parent.name, action_dir.name
        joint_names = exercise_joints.get(exercise)
        if joint_names is None:
            logger.warning(
                "Ejercicio %s (clip %s) sin orden de joints en el JSON; saltado.",
                exercise,
                folder,
            )
            skipped_clips += 1
            continue
        # Fuente de labels: camN_p2d.txt (cubre los 27 sujetos), no el JSON —
        # que sólo anota estos ejercicios para el sujeto 0.
        for mp4_path in sorted(action_dir.glob("cam*.mp4")):
            cam = mp4_path.stem.removeprefix("cam")
            p2d_path = action_dir / f"cam{cam}_p2d.txt"
            if not p2d_path.exists():
                logger.warning("Falta %s; cámara saltada.", p2d_path)
                continue
            frames_by_id = parse_p2d_file(p2d_path, joint_names)
            stems = _extract_camera(
                mp4_path,
                folder,
                exercise,
                cam,
                frames_by_id,
                output_root,
                stride=stride,
            )
            stems_by_subject.setdefault(folder, []).extend(stems)
            total_written += len(stems)

    if not stems_by_subject:
        raise RuntimeError(
            f"No se escribió ningún frame desde {clips_dir}: revisá que haya videos "
            "con anotación en dataset_2d.json."
        )

    yaml_paths = write_fold_manifests(
        output_root, output_yaml, stems_by_subject, n_folds=n_folds, seed=seed
    )
    logger.info(
        "Conversión completa: %d frames de %d sujetos, %d folds, %d clips sin anotación saltados.",
        total_written,
        len(stems_by_subject),
        n_folds,
        skipped_clips,
    )
    return yaml_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convierte el dataset UCO a YOLOv8-pose.")
    parser.add_argument("--dataset-root", default="data/raw/uco", help="Raíz UCO.")
    parser.add_argument("--output-root", default="data/processed/uco_yolo", help="Salida YOLO.")
    parser.add_argument(
        "--output-yaml", default="data/processed/pose_dataset.yaml", help="Dataset YAML."
    )
    parser.add_argument("--stride", type=int, default=30, help="1 frame de cada N.")
    parser.add_argument(
        "--folds", type=int, default=5, help="Cantidad de folds (GroupKFold por sujeto)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla del reparto de folds.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    yaml_paths = convert(
        args.dataset_root,
        args.output_root,
        args.output_yaml,
        stride=args.stride,
        n_folds=args.folds,
        seed=args.seed,
    )
    print(f"{len(yaml_paths)} dataset YAML (GroupKFold por sujeto) generados:")
    for p in yaml_paths:
        print(f"  {p}")
