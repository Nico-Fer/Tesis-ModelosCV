"""Implementación concreta de :class:`PoseModel` para YOLOv8-pose.

Modelo PRINCIPAL del proyecto. Se usa EXCLUSIVAMENTE la API de Ultralytics
(no se escribe código PyTorch directo). Cubre Fase 1 (baseline) y deja los
ganchos necesarios para Fases 2, 4 y 5 (pruning built-in, distillation y
exportación cuantizada).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.evaluation import metrics as metric_utils
from src.evaluation import pose_eval
from src.models.base import ModelMetrics, PoseModel
from src.utils import strict

logger = logging.getLogger(__name__)

# Desactiva el AutoUpdate de Ultralytics. `ultralytics` se importa de forma
# diferida dentro de los métodos, así que basta con fijar la env var al cargar
# este módulo. Sin esto, la exportación a TFLite de 8.3.x reinstala su toolchain
# esperado (onnx2tf>=1.26, Keras 3, etc.) y rompe los pins de
# requirements-torch.txt. Todas las deps de export están declaradas ahí, así que
# preferimos que falle de forma explícita antes que mutar el entorno.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# Claves de config que se pasan tal cual a ``YOLO.train`` cuando están definidas,
# con el tipo al que se castean. Ultralytics tiene defaults propios para todas, así
# que omitirlas es válido; declararlas es lo que permite aprovechar la GPU
# (``device``/``batch``/``workers``/``amp``) y cortar temprano (``patience``).
_TRAIN_OVERRIDES: dict[str, Any] = {
    "batch": int,
    "device": str,
    "workers": int,
    "amp": bool,
    "patience": int,
    "cache": str,  # "ram" / "disk" / False; Ultralytics acepta ambos
    "fraction": float,  # fracción del split train a usar (runs cortos)
    "optimizer": str,  # ver nota abajo: imprescindible si se declara lr0
    "lr0": float,
}

# NOTA sobre ``optimizer`` y ``lr0``: el default de Ultralytics es
# ``optimizer="auto"``, y en ese modo ``build_optimizer`` **sobreescribe el lr0
# recibido** con uno que deriva del tamaño del dataset ("ignoring 'lr0=…' and
# determining best 'optimizer', 'lr0' and 'momentum' automatically" en el log).
# Es decir: pasar ``lr0`` sin fijar ``optimizer`` no tiene ningún efecto. Importa
# porque el fine-tuning post-poda reduce el lr a 1/10 a propósito
# (:func:`src.optimization.pruning.fine_tune_after_pruning`), y con "auto" esa
# reducción se descartaba en silencio. Las configs de Fase 2 y 5 declaran
# ``optimizer`` explícitamente por eso.


def train_overrides(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Extrae de ``config`` los kwargs de entrenamiento a reenviar a Ultralytics.

    Centraliza el pass-through para que los tres caminos que entrenan un YOLO
    (:meth:`YoloPoseModel.train`, el fine-tuning post-poda de
    :mod:`src.optimization.pruning` y la distillation de
    :mod:`src.optimization.distillation`) usen exactamente los mismos parámetros.
    Sin esto, los dos últimos llamaban a ``yolo.train()`` con los defaults de
    Ultralytics e ignoraban ``batch``/``workers``/``device`` de la config.

    Args:
        config: configuración de la fase.
        smoke: si es un smoke test, se omite ``cache`` (el llamador fuerza False).

    Returns:
        Diccionario con sólo las claves presentes en ``config``, ya casteadas.
    """
    out: dict[str, Any] = {}
    for key, caster in _TRAIN_OVERRIDES.items():
        if smoke and key == "cache":
            continue
        value = config.get(key)
        if value is None:
            continue
        out[key] = value if caster is str else caster(value)
    return out


# Hiperparámetros de augmentation de Ultralytics reenviables desde la sección
# ``augmentation:`` del YAML (un solo nivel, compatible con ``_flatten_params``).
# Todos tienen default propio en Ultralytics, así que la sección es opcional.
_AUG_OVERRIDES: dict[str, Any] = {
    "hsv_h": float,
    "hsv_s": float,
    "hsv_v": float,
    "degrees": float,
    "translate": float,
    "scale": float,
    "shear": float,
    "perspective": float,
    "flipud": float,
    "fliplr": float,
    "mosaic": float,
    "mixup": float,
    "copy_paste": float,
    "close_mosaic": int,
    "multi_scale": bool,
}


def augmentation_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Extrae la sección ``augmentation`` de la config como kwargs de Ultralytics.

    Igual que :func:`train_overrides`, centraliza el pass-through para que el
    entrenamiento normal y el fine-tuning post-poda apliquen exactamente la misma
    augmentation. Una clave desconocida aborta la fase (convención fail-fast):
    una typo acá entrenaría horas con la augmentation default en silencio.

    Args:
        config: configuración de la fase (sección opcional ``augmentation``).

    Returns:
        Diccionario con sólo las claves declaradas, ya casteadas.
    """
    aug: dict[str, Any] = config.get("augmentation") or {}
    unknown = sorted(set(aug) - set(_AUG_OVERRIDES))
    if unknown:
        strict.fail(
            f"Claves desconocidas en 'augmentation': {unknown}. Válidas: {sorted(_AUG_OVERRIDES)}."
        )
    return {
        key: (value if _AUG_OVERRIDES[key] is str else _AUG_OVERRIDES[key](value))
        for key, value in aug.items()
        if key in _AUG_OVERRIDES
    }


class YoloPoseModel(PoseModel):
    """Familia YOLOv8-pose (variantes n/s/x) vía Ultralytics.

    Claves de ``config`` esperadas:
        model (str): checkpoint o nombre, p. ej. ``"yolov8n-pose.pt"``.
        weights (str, opcional): path a ``.pt`` para cargar pesos adicionales.
        data (str, opcional): ruta al dataset YAML de Ultralytics.
        epochs (int): épocas de entrenamiento (default 100).
        imgsz (int): tamaño de imagen (default 640).
        project (str, opcional): carpeta raíz de runs (default ``"runs/pose"``).
        smoke_test (bool, opcional): si True, fuerza epochs=1, imgsz=32, cache=False
            y degrada los fallos del pipeline a warnings (ver :mod:`src.utils.strict`).

    Además, cualquier clave de :data:`_TRAIN_OVERRIDES` (``batch``, ``device``,
    ``workers``, ``amp``, ``patience``, ``cache``, ``fraction``, ``lr0``) se
    reenvía tal cual a ``YOLO.train`` si está definida.
    """

    def load(self) -> None:
        """Instancia el modelo Ultralytics desde el checkpoint configurado.

        Idempotente: no recarga si ``self._model`` ya está inicializado.
        Si ``config["weights"]`` está presente, carga esos pesos sobre el
        modelo base (útil para reanudar o transferir pesos fine-tuneados).
        """
        from ultralytics import YOLO  # diferido: solo disponible en venv-torch
        from ultralytics.utils import SETTINGS  # noqa: PLC0415

        if self._model is not None:
            return

        # El pipeline gestiona MLflow vía src.utils.mlflow_utils. El callback de
        # MLflow propio de Ultralytics (que se dispara en train()) duplica runs y,
        # con un MLFLOW_TRACKING_URI ambiente inválido, rompe el entrenamiento al
        # intentar registrar el modelo. Se desactiva sólo en memoria con
        # ``dict.__setitem__`` (el ``SettingsManager.__setitem__`` podría persistir
        # al config global de Ultralytics del usuario, lo que evitamos).
        dict.__setitem__(SETTINGS, "mlflow", False)

        self._model = YOLO(self.config["model"])

        # El tracking de MLflow lo gestiona el proyecto vía src.utils.mlflow_utils
        # (el runner abre el run y loguea params/métricas/artefactos). El logger
        # MLflow integrado de Ultralytics, en cambio, llama a mlflow.set_experiment
        # con el nombre del proyecto durante el entrenamiento: eso intercambia el
        # experimento activo y fragmenta el run (params y .tflite terminan en un
        # experimento, las curvas de Ultralytics en otro). Su callback se anula al
        # importarse si SETTINGS["mlflow"] no es True (el módulo se importa de forma
        # diferida recién en train()). Fijamos ese flag a False ANTES de entrenar,
        # SOLO en memoria: dict.__setitem__ evita que SettingsManager lo persista en
        # la config global de Ultralytics. Así mlflow_utils es la única fuente de
        # verdad sin alterar el entorno de otros proyectos.
        from ultralytics.utils import SETTINGS  # noqa: PLC0415

        dict.__setitem__(SETTINGS, "mlflow", False)

        weights = self.config.get("weights")
        if weights:
            self._model.load(str(weights))

    def train(self) -> None:
        """Entrena / fine-tunea el modelo con la API de Ultralytics.

        Si ``config["data"]`` no está definido se aborta la fase
        (:class:`~src.utils.strict.PipelineError`): entrenar sin dataset no
        produce un modelo válido, y degradarlo a un no-op silencioso hacía que un
        run vacío fuese indistinguible de uno sano. Sólo en un smoke test
        declarado (``smoke_test: true``) se retorna sin entrenar.

        Cuando ``config["smoke_test"]`` es True se aplican parámetros mínimos
        (epochs=1, imgsz=32, cache=False) para un ciclo rápido sin datos reales.
        """
        self.load()
        data = self.config.get("data")
        if not data:
            strict.fail(
                f"{self.name}: falta 'data' en la config; sin dataset no hay entrenamiento. "
                "Declará 'smoke_test: true' si la intención es correr el esqueleto sin datos."
            )
            return

        assert self._model is not None

        smoke = bool(self.config.get("smoke_test", False))
        train_kwargs: dict[str, Any] = {
            "data": data,
            "epochs": 1 if smoke else int(self.config.get("epochs", 100)),
            "imgsz": 32 if smoke else int(self.config.get("imgsz", 640)),
            # Absoluto: con project relativo Ultralytics lo resuelve contra su
            # runs_dir global (posiblemente otro proyecto). Absoluto lo mantiene
            # en el repo (carpeta runs/, gitignored).
            "project": str(Path(self.config.get("project", "runs/pose")).resolve()),
            "name": self.name,
        }
        if smoke:
            train_kwargs["cache"] = False

        # Incluir sólo si están definidos; Ultralytics tiene sus propios defaults.
        # ``lr0`` viene reducido en el fine-tuning post-poda (ver optimization.pruning).
        train_kwargs.update(train_overrides(self.config, smoke=smoke))
        train_kwargs.update(augmentation_overrides(self.config))

        self._model.train(**train_kwargs)

    def evaluate(self) -> ModelMetrics:
        """Evalúa el modelo y devuelve métricas comparables.

        Sin ``config["data"]`` se aborta la fase salvo en smoke test declarado
        (donde sólo se reporta ``size_mb``): un run sin métricas de precisión no
        es comparable y no debe pasar por válido. Con dataset se obtiene mAP50-95
        de la validación de Ultralytics y OKS/PCK con el evaluador unificado
        (:mod:`src.evaluation.pose_eval`), que corre inferencia sobre el set de
        validación y puntúa contra el GT de UCO normalizando por un bbox de
        persona derivado del detector (el bbox de UCO es del tamaño de un miembro).

        La latencia se mide sobre el ``.tflite`` exportado (no sobre el ``.pt``,
        que no es comparable), por lo que aquí queda en ``None``.

        Returns:
            :class:`ModelMetrics` con los campos que se pueden computar.
        """
        self.load()

        # size_mb: guard para checkpoints referenciados por nombre (no en disco aún).
        model_path = Path(self.config["model"])
        size_mb: float | None = None
        if model_path.exists():
            size_mb = metric_utils.model_size_mb(model_path)

        data = self.config.get("data")
        if not data:
            strict.fail(
                f"{self.name}: falta 'data' en la config; sin dataset no hay métricas de "
                "precisión y el run no es comparable."
            )
            return ModelMetrics(size_mb=size_mb)

        assert self._model is not None
        imgsz = int(self.config.get("imgsz", 640))
        # ``project`` absoluto por el mismo motivo que en train(): sin él,
        # Ultralytics resuelve el destino contra su ``runs_dir`` GLOBAL, que puede
        # apuntar a otro proyecto del usuario y saca los resultados del repo.
        results = self._model.val(
            data=data,
            imgsz=imgsz,
            project=str(Path(self.config.get("project", "runs/pose")).resolve()),
            name=f"val_{Path(str(self.config['model'])).stem}",
        )

        pose_result = getattr(results, "pose", None)
        map50_95: float | None = getattr(pose_result, "map", None)

        # OKS/PCK con el evaluador unificado: usa este mismo modelo como detector
        # (para el bbox de persona normalizador) y como predictor de keypoints.
        # Un fallo acá deja el run sin su métrica primaria (PCK@0.2), así que se
        # propaga: antes se degradaba a None y el run pasaba por bueno.
        oks: float | None = None
        pck: float | None = None
        try:
            ctx = pose_eval.build_eval_context(data, self._model, imgsz)
            if ctx is not None:
                score = pose_eval.score_stage(
                    pose_eval.run_predictions(self._model, ctx.img_paths, imgsz), ctx
                )
                oks, pck = score.oks, score.pck
        except Exception as exc:  # noqa: BLE001
            if not strict.is_smoke():
                raise
            logger.warning("[smoke] No se pudo computar OKS/PCK: %s", exc)

        return ModelMetrics(
            map=map50_95,
            oks=oks,
            pck=pck,
            latency_ms=None,
            size_mb=size_mb,
        )

    def export_tflite(self, output_dir: Path, quantize: bool) -> Path:
        """Exporta a TFLite vía la cadena de export de Ultralytics.

        La cadena completa es PyTorch → ONNX → TensorFlow → TFLite, por lo que
        **este método requiere venv-torch**: tensorflow, onnx2tf y onnxslim deben
        estar instalados en el mismo entorno que Ultralytics. Ver
        ``requirements-torch.txt`` y el spike de export del README.

        Args:
            output_dir: carpeta destino informativa; Ultralytics determina el
                nombre final del archivo según el checkpoint de origen.
            quantize: ``True`` aplica cuantización INT8 (Fases 3/5) y requiere
                ``config["data"]`` como set de calibración; ``False`` exporta en
                Float32 (Fase 1 baseline).

        Returns:
            :class:`~pathlib.Path` al archivo ``.tflite`` generado.

        Notas:
            La calibración INT8 de Ultralytics carga imágenes del split ``val`` del
            dataset. Con un ``val`` grande (p. ej. los 16k frames de un fold UCO)
            esto agota la memoria (intenta materializar todo el set a la vez). Por
            eso se acota con ``config["calib_fraction"]`` (fracción del split usada
            para calibrar); si no se define, se usa el default de Ultralytics (1.0).
        """
        self.load()
        assert self._model is not None
        output_dir.mkdir(parents=True, exist_ok=True)

        export_kwargs: dict[str, Any] = {
            "format": "tflite",
            "imgsz": int(self.config.get("imgsz", 640)),
        }
        if quantize:
            export_kwargs["int8"] = True
            if self.config.get("data"):
                export_kwargs["data"] = self.config["data"]
                # Acota las imágenes de calibración para no agotar memoria.
                calib_fraction = self.config.get("calib_fraction")
                if calib_fraction is not None:
                    export_kwargs["fraction"] = float(calib_fraction)

        exported = self._model.export(**export_kwargs)
        return Path(exported)
