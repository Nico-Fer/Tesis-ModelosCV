"""Knowledge distillation Maestro → Estudiante (Fase 4).

Expone dos niveles de API:

``distill_yolo``
    Función principal orientada a archivos: recibe los pesos del maestro y la
    config del estudiante, entrena al estudiante con la loss combinada y devuelve
    el path al ``best.pt``. Es la lógica real de esta técnica.

``build_distillation_loss`` / ``run_distillation_training``
    Wrappers orientados a instancias de :class:`~src.models.base.PoseModel`. Los
    runners de Fase 4 y 5 los llaman sin cambios. El dispatch por
    ``config["family"]`` mantiene los runners agnósticos al modelo: ``yolo`` usa
    el trainer custom de abajo; las familias congeladas (``movenet`` /
    ``blazepose``) hacen un no-op grácil.

Entorno requerido: **venv-torch** (Ultralytics + PyTorch).

Técnica (feature-based, *attention transfer*):
    Ultralytics 8.3 no expone una API de distillation. Se subclasea su
    ``PoseTrainer`` e inyecta la distillation loss **antes** del backward, vía un
    *forward hook* sobre el modelo del estudiante que reemplaza el ``(loss,
    loss_items)`` devuelto por una versión combinada.

    La señal de distillation es **attention transfer** (Zagoruyko & Komodakis,
    2017) sobre los feature maps del neck (``P3``/``P4``/``P5``, las 3 entradas
    del head): para cada nivel se calcula el mapa de atención espacial
    ``A(F) = normalize(mean_c F²)`` y la loss es el MSE entre las atenciones del
    maestro y del estudiante. Es **agnóstico al número de canales** (el maestro
    yolov8x y el estudiante yolov8n tienen distinto C), por lo que no requiere
    capas de adaptación ni tocar el optimizador de Ultralytics. Las dimensiones
    espaciales coinciden (mismo ``imgsz`` y strides); si difirieran, el maestro
    se reescala al tamaño del estudiante.

    Combinación::

        L = α · L_task + (1 − α) · (β/2) · B · L_AT

    * ``L_AT`` es la del código oficial de Zagoruyko & Komodakis: media de los
      cuadrados de la diferencia de atenciones, sumada sobre niveles, con **β = 1000**
      (su default). La media divide por H·W; β es lo que la devuelve a una escala útil.
    * ``× B`` (tamaño de batch): ``v8PoseLoss`` devuelve ``loss * batch_size``, así que
      ``L_task`` es una **suma** sobre el batch mientras ``L_AT`` es una media. Sin este
      factor el término de imitación pesaba B veces menos de lo que dice β.
    * **No hay T².** Viene de Hinton et al., donde compensa el 1/T² del gradiente de un
      softmax suavizado sobre *logits*; los mapas de atención no tienen softmax ni
      temperatura. La versión anterior usaba ``T² · L_AT`` sin β ni ``× B``, con lo que el
      término era ~0.04 % de la loss: el run que "probó que distillation empeora" en
      realidad no destilaba (ver AGENTS.md).

    El LR de esta etapa es ``lr0/10`` (o ``distillation.lr0`` si se declara): el
    estudiante siempre llega entrenado (COCO, o podado + fine-tune en la Fase 5), y
    reiniciarlo al ``lr0`` completo lo hacía divergir. Mismo criterio que
    :func:`~src.optimization.pruning.fine_tune_after_pruning`. Por la misma razón el
    warmup de los bias arranca en 0 (``distillation.warmup_bias_lr``, default 0.0) y no
    en el 0.1 de Ultralytics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.models.base import PoseModel
from src.utils import strict

logger = logging.getLogger(__name__)


_DEFAULT_ALPHA = 0.5
_DEFAULT_BETA = 1000.0  # β de Zagoruyko & Komodakis (código oficial de attention-transfer)
_KD_LOG_EVERY = 500  # batches entre logs del peso real de cada término

# Claves válidas de la sección ``distillation:``. ``temperature`` se retiró a propósito
# (ver docstring del módulo): aceptarla la dejaría logueada en MLflow sin efecto alguno.
_DISTILL_KEYS = frozenset({"alpha", "beta", "epochs", "val", "lr0", "warmup_bias_lr"})

# Ultralytics arranca el LR de los bias en 0.1 durante el warmup (3 epochs) mientras los
# pesos arrancan en 0. Sobre un estudiante ya entrenado, con AdamW, eso mueve cada bias
# ~0.1 por paso: medido en 1 epoch parcial sobre el v3 podado+FT, pose_loss saltó de 0.90
# a 2.3 y el OKS cayó −0.014; con 0.0 (bias rampan desde 0 como los pesos) −0.006.
_DEFAULT_WARMUP_BIAS_LR = 0.0


def _distill_section(config: dict[str, Any]) -> dict[str, Any]:
    """Sección ``distillation`` validada (fail-fast ante claves obsoletas o typos)."""
    dist_cfg: dict[str, Any] = config.get("distillation") or {}
    if "temperature" in dist_cfg:
        strict.fail(
            "'distillation.temperature' ya no existe: T² es de Hinton et al. (softmax sobre "
            "logits) y no aplica a attention transfer. El peso del término es "
            "'distillation.beta' (default 1000). Quitá la clave del YAML."
        )
    unknown = sorted(set(dist_cfg) - _DISTILL_KEYS - {"temperature"})
    if unknown:
        strict.fail(
            f"Claves desconocidas en 'distillation': {unknown}. Válidas: {sorted(_DISTILL_KEYS)}."
        )
    return dist_cfg


def distillation_lr0(config: dict[str, Any]) -> float | None:
    """LR efectivo de la etapa de distillation: ``distillation.lr0`` o ``lr0/10``.

    ``None`` si la config no declara ``lr0`` (Ultralytics elige solo con
    ``optimizer: auto``, y en ese modo descarta cualquier ``lr0`` de todos modos).
    """
    dist_cfg = config.get("distillation") or {}
    if dist_cfg.get("lr0") is not None:
        return float(dist_cfg["lr0"])
    if config.get("lr0") is not None:
        return float(config["lr0"]) / 10.0
    return None


def _build_distill_trainer_cls(
    teacher_nn: Any,  # noqa: ANN401
    alpha: float,
    beta: float,
    student_module: Any = None,  # noqa: ANN401
) -> type:
    """Crea una subclase de ``PoseTrainer`` con la distillation loss inyectada.

    Se usa una *factory* (no parámetros de constructor) porque Ultralytics
    instancia el trainer como ``Cls(overrides=..., _callbacks=...)`` y no admite
    argumentos extra; el maestro y los hiperparámetros se hornean como atributos
    de clase / variables de closure.

    Args:
        teacher_nn: ``torch.nn.Module`` del maestro (``YOLO().model``), congelado.
        alpha: peso de la task loss frente a la distillation loss.
        beta: peso β del término de attention transfer (se aplica como ``β/2``).
        student_module: si se provee, el ``torch.nn.Module`` **ya podado** del
            estudiante. El trainer sobreescribe ``get_model`` para devolverlo, de
            modo que ``PoseTrainer`` no reconstruya la arquitectura desde el YAML
            (lo que desharía la poda). Combina así las dos responsabilidades:
            hooks de distillation + preservación de la arquitectura podada
            (cf. ``pruning._build_pruned_trainer_cls``).

    Returns:
        La clase ``DistillPoseTrainer`` lista para pasar a ``YOLO.train(trainer=...)``.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from ultralytics.models.yolo.pose import PoseTrainer  # noqa: PLC0415
    from ultralytics.nn.modules.head import Detect  # noqa: PLC0415
    from ultralytics.utils import LOGGER  # noqa: PLC0415

    def _attention(feat: "torch.Tensor") -> "torch.Tensor":
        """Mapa de atención espacial normalizado de un feature map ``(B,C,H,W)``.

        En float32 aunque el forward corra con AMP: ``F²`` y la norma en fp16 son frágiles.
        """
        att = feat.float().pow(2).mean(dim=1)  # (B, H, W) — colapsa canales
        att = att.flatten(1)  # (B, H*W)
        return F.normalize(att, p=2, dim=1)

    def _attention_transfer_loss(
        student_feats: dict[int, "torch.Tensor"],
        teacher_feats: dict[int, "torch.Tensor"],
    ) -> "torch.Tensor":
        """``L_AT`` del código oficial de Zagoruyko & Komodakis, sumada sobre niveles."""
        total = None
        for level, fs in student_feats.items():
            ft = teacher_feats[level]
            if ft.shape[-2:] != fs.shape[-2:]:
                ft = F.interpolate(ft, size=fs.shape[-2:], mode="bilinear", align_corners=False)
            level_loss = (_attention(fs) - _attention(ft)).pow(2).mean()
            total = level_loss if total is None else total + level_loss
        assert total is not None
        return total

    from src.utils.checkpoints import Fp32OptimizerCheckpointMixin  # noqa: PLC0415

    # El mixin guarda el optimizer en fp32 en last.pt: en fp16 la reanudación diverge.
    class DistillPoseTrainer(Fp32OptimizerCheckpointMixin, PoseTrainer):  # type: ignore[misc, valid-type]
        """``PoseTrainer`` con attention-transfer distillation desde un maestro."""

        _teacher_nn = teacher_nn
        _kd_alpha = float(alpha)
        _kd_beta = float(beta)
        _pruned_module = student_module

        def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> Any:  # noqa: ANN401
            # Con un módulo podado horneado, evitar que PoseTrainer reconstruya la
            # arquitectura desde el YAML (deshaciendo la poda). setup_model respeta
            # un nn.Module ya instanciado. Sin él, comportamiento estándar.
            if self._pruned_module is not None:
                # Al reanudar (resume=True) llegan los pesos EMA de last.pt, con la
                # misma arquitectura podada: se copian sobre el módulo horneado. La loss
                # cacheada se descarta (ver pruning._build_pruned_trainer_cls: viene de
                # last.pt con tensores en CPU y los hiperparámetros de otra corrida).
                if weights is not None and weights is not self._pruned_module:
                    self._pruned_module.load_state_dict(weights.float().state_dict())
                self._pruned_module.criterion = None
                return self._pruned_module
            return super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        def _setup_train(self) -> None:
            super()._setup_train()
            self._student_feats: dict[int, Any] = {}
            self._teacher_feats: dict[int, Any] = {}

            # Maestro al device del trainer, congelado y en modo eval.
            self._teacher_nn.to(self.device).eval()
            for p in self._teacher_nn.parameters():
                p.requires_grad_(False)

            # Pre-hook en el head: su entrada es la lista [P3, P4, P5].
            student_head = self._find_head(self.model)
            teacher_head = self._find_head(self._teacher_nn)
            student_head.register_forward_pre_hook(self._student_head_pre_hook)
            teacher_head.register_forward_pre_hook(self._teacher_head_pre_hook)
            # Hook sobre el modelo del estudiante: reemplaza la loss devuelta.
            self.model.register_forward_hook(self._student_loss_hook)
            self._kd_step = 0
            # Mensajes en ASCII: en Windows el LOGGER de Ultralytics no codifica α/β/·.
            LOGGER.info(
                f"DistillPoseTrainer: L = {self._kd_alpha:g}*L_task + "
                f"{1 - self._kd_alpha:g}*({self._kd_beta:g}/2)*B*L_AT"
            )

        @staticmethod
        def _find_head(module: Any) -> Any:  # noqa: ANN401
            heads = [m for m in module.modules() if isinstance(m, Detect)]
            if not heads:
                raise RuntimeError("No se encontró el head Detect/Pose para los hooks de KD.")
            return heads[-1]

        def _student_head_pre_hook(self, _module: Any, args: tuple) -> None:  # noqa: ANN401
            feats = args[0]  # lista de 3 tensores (con grad)
            self._student_feats = {i: f for i, f in enumerate(feats)}

        def _teacher_head_pre_hook(self, _module: Any, args: tuple) -> None:  # noqa: ANN401
            feats = args[0]
            self._teacher_feats = {i: f.detach() for i, f in enumerate(feats)}

        def _student_loss_hook(self, module: Any, inputs: tuple, output: Any) -> Any:  # noqa: ANN401
            # Sólo en entrenamiento el modelo devuelve (loss, loss_items).
            if not module.training or not isinstance(output, (tuple, list)) or len(output) != 2:
                return output

            loss, loss_items = output
            batch = inputs[0]
            imgs = batch["img"] if isinstance(batch, dict) else batch

            # El maestro se ejecuta una vez por batch; su pre-hook llena _teacher_feats.
            with torch.no_grad():
                self._teacher_nn(imgs)

            if not self._student_feats or not self._teacher_feats:
                return output

            at_loss = _attention_transfer_loss(self._student_feats, self._teacher_feats)
            # × B: L_task viene sumada sobre el batch (v8PoseLoss: loss * batch_size) y
            # L_AT es una media; sin el factor, β quedaría dividido por el batch.
            distill = (self._kd_beta / 2.0) * imgs.shape[0] * at_loss

            task_term = self._kd_alpha * loss.sum()
            distill_term = (1.0 - self._kd_alpha) * distill.to(task_term.dtype)
            new_loss = task_term + distill_term

            # El defecto anterior (término de imitación ~0.04 % de la loss) pasó
            # inadvertido porque nada mostraba su peso: se loguea explícito.
            if self._kd_step % _KD_LOG_EVERY == 0:
                share = float(distill_term.detach() / new_loss.detach().clamp_min(1e-12))
                LOGGER.info(
                    f"KD step {self._kd_step}: alpha*L_task={float(task_term):.4g} "
                    f"(1-alpha)*beta/2*B*L_AT={float(distill_term):.4g} "
                    f"(L_AT={float(at_loss):.3g}) -> imitacion = {share:.1%} de la loss"
                )
            self._kd_step += 1
            return (new_loss, loss_items)

    return DistillPoseTrainer


def distill_yolo(
    teacher_weights: str | Path,
    student_cfg: str | Path,
    output_dir: str | Path,
    distill_cfg: dict[str, Any],
    student_yolo: Any = None,  # noqa: ANN401
    resume_from: Path | None = None,
    on_model_save: Any = None,  # noqa: ANN401
) -> Path:
    """Entrena un estudiante YOLOv8-pose supervisado por el maestro (attention transfer).

    Construye un :class:`DistillPoseTrainer` (vía :func:`_build_distill_trainer_cls`)
    con el maestro horneado, y lo pasa a ``YOLO.train(trainer=...)``. El trainer
    registra los hooks que combinan la task loss con la distillation loss antes del
    backward.

    Args:
        teacher_weights: path al checkpoint ``.pt`` del maestro (p. ej. ``yolov8x-pose.pt``).
        student_cfg: path/nombre del checkpoint inicial del estudiante.
        output_dir: directorio donde Ultralytics guarda el run del estudiante.
        distill_cfg: hiperparámetros. Claves: ``data`` (obligatoria), ``epochs``,
            ``imgsz``, ``alpha`` (default 0.5), ``beta`` (default 1000) y
            ``train_kwargs`` (dict opcional reenviado tal cual a ``YOLO.train``:
            ``device``, ``batch``, ``workers``, ``amp``, ``patience``, ``lr0``, …).
        student_yolo: instancia ``YOLO`` del estudiante **ya podado** en memoria.
            Si se provee, se entrena esa instancia (no una nueva desde
            ``student_cfg``) y el trainer preserva su arquitectura podada. Es el
            camino de la Fase 5: la poda debe sobrevivir a la distillation.
        resume_from: ``last.pt`` de un run interrumpido. Si se pasa, se reanuda ese
            run (``resume=True``: hiperparámetros, epoch, optimizer y EMA salen del
            checkpoint) en vez de arrancar uno nuevo.
        on_model_save: callback opcional de Ultralytics (``on_model_save``), p. ej.
            para registrar el avance por epoch (ver :mod:`src.utils.checkpoints`).

    Returns:
        :class:`~pathlib.Path` al ``best.pt`` del estudiante entrenado.

    Raises:
        ImportError: si se invoca fuera de venv-torch.
        ValueError: si ``distill_cfg`` no incluye ``data``.
    """
    from ultralytics import YOLO  # noqa: PLC0415
    from ultralytics.utils import SETTINGS  # noqa: PLC0415

    # Desactiva el callback de MLflow de Ultralytics: el pipeline gestiona MLflow
    # aparte y el callback rompe el train() con un tracking URI ambiente inválido.
    # ``dict.__setitem__`` evita que SettingsManager persista el cambio al config
    # global del usuario.
    dict.__setitem__(SETTINGS, "mlflow", False)

    teacher_weights = Path(teacher_weights)
    output_dir = Path(output_dir)

    data = distill_cfg.get("data")
    if not data:
        raise ValueError(
            "distill_yolo requiere 'data' en distill_cfg para entrenar. "
            "Sin dataset no se puede ejecutar el ciclo de distillation."
        )

    alpha = float(distill_cfg.get("alpha", _DEFAULT_ALPHA))
    beta = float(distill_cfg.get("beta", _DEFAULT_BETA))
    epochs = int(distill_cfg.get("epochs", 100))
    imgsz = int(distill_cfg.get("imgsz", 640))

    teacher = YOLO(str(teacher_weights))
    student: Any
    if resume_from is not None:
        # El estudiante sale del propio last.pt: su EMA ya tiene la arquitectura podada.
        student = YOLO(str(resume_from))
        pruned_module = student.model
    else:
        # Estudiante: el ya podado en memoria (Fase 5) o uno nuevo desde el checkpoint.
        student = student_yolo if student_yolo is not None else YOLO(str(student_cfg))
        pruned_module = student.model if student_yolo is not None else None

    trainer_cls = _build_distill_trainer_cls(
        teacher.model, alpha, beta, student_module=pruned_module
    )
    if on_model_save is not None:
        student.add_callback("on_model_save", on_model_save)

    # Validación por-epoch de Ultralytics: computa métricas de task (mAP) que este
    # pipeline NO usa (la evaluación comparable se hace aparte con pose_eval). Con
    # val=False se evita revalidar el split completo en cada epoch (costoso en CPU).
    do_val = bool(distill_cfg.get("val", True))

    # Resto de kwargs de entrenamiento (device/batch/workers/amp/patience/…), los
    # mismos que usa YoloPoseModel.train: sin esto la distillation corría con los
    # defaults de Ultralytics e ignoraba la config de la fase.
    extra: dict[str, Any] = dict(distill_cfg.get("train_kwargs") or {})

    if resume_from is not None:
        # Los hiperparámetros salen del propio last.pt; sólo workers puede cambiar.
        resume_kw = {"workers": extra["workers"]} if "workers" in extra else {}
        student.train(trainer=trainer_cls, resume=True, **resume_kw)
        return Path(student.trainer.best)  # type: ignore[union-attr]

    output_dir.mkdir(parents=True, exist_ok=True)
    student.train(
        trainer=trainer_cls,
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        val=do_val,
        project=str(output_dir),
        name="distill_student",
        **extra,
    )

    # Ruta autoritativa del checkpoint: Ultralytics resuelve el directorio de run
    # según su settings ``runs_dir`` global (no necesariamente bajo output_dir).
    return Path(student.trainer.best)  # type: ignore[union-attr]


def build_distillation_loss(config: dict[str, Any]) -> dict[str, Any]:
    """Devuelve los hiperparámetros de la loss combinada como diccionario logueable.

    ``L = α · L_task + (1 − α) · (β/2) · B · L_AT`` (ver docstring del módulo). Loguea
    los valores **efectivos**, incluidos los defaults que el YAML no declara y el LR
    real de la etapa, que difiere del ``lr0`` general.

    Args:
        config: configuración de la fase, incluyendo la sección ``distillation``.

    Returns:
        Diccionario ``{"distillation.alpha": α, "distillation.beta": β,
        "distillation.lr0_efectivo": lr}`` listo para
        :func:`~src.utils.mlflow_utils.log_params` (sin ``lr0_efectivo`` si la config
        no declara ``lr0``).
    """
    dist_cfg = _distill_section(config)
    out: dict[str, Any] = {
        "distillation.alpha": float(dist_cfg.get("alpha", _DEFAULT_ALPHA)),
        "distillation.beta": float(dist_cfg.get("beta", _DEFAULT_BETA)),
        "distillation.warmup_bias_lr": float(
            dist_cfg.get("warmup_bias_lr", _DEFAULT_WARMUP_BIAS_LR)
        ),
    }
    lr = distillation_lr0(config)
    if lr is not None:
        out["distillation.lr0_efectivo"] = lr
    return out


def run_distillation_training(
    teacher: PoseModel,
    student: PoseModel,
    config: dict[str, Any],
) -> None:
    """Ejecuta el ciclo de distillation del estudiante supervisado por el maestro.

    Dispatch por familia (patrón de ``registry.build_model``):

    * ``yolo``: delega en :func:`distill_yolo`.
    * ``movenet`` / ``blazepose``: modelos congelados → no-op grácil (no se destilan).

    Sin ``config["data"]`` aborta la fase (salvo smoke test declarado): sin ciclo
    de entrenamiento el "estudiante destilado" es el checkpoint inicial intacto.

    Args:
        teacher: modelo Maestro ya construido; sólo se usa en inferencia.
        student: modelo Estudiante que se entrena con la loss combinada.
        config: configuración de la fase, incluyendo la sección ``distillation``.
    """
    family = str(config.get("family", ""))
    if family != "yolo":
        logger.info("Distillation no aplicable a la familia %r; se omite.", family)
        return

    data = config.get("data")
    if not data:
        strict.fail(
            "Distillation sin 'data': el estudiante nunca se entrena y la fase devuelve "
            "el checkpoint inicial sin optimizar. Declará 'smoke_test: true' para el esqueleto."
        )
        return

    dist_cfg = _distill_section(config)
    teacher_weights = config.get("teacher_model", "")
    student_cfg_path = config.get("model", "")
    # Ruta absoluta anclada al CWD (raíz del repo): con un ``project`` relativo,
    # Ultralytics lo resuelve contra su ``runs_dir`` GLOBAL (que puede apuntar a
    # otro proyecto del usuario), sacando los checkpoints fuera del repo. Absoluto
    # garantiza que el modelo entrenado quede dentro del repo (carpeta gitignored).
    output_dir = Path(config.get("project", "runs/pose")).resolve() / "distillation"

    from src.models.yolo import train_overrides  # noqa: PLC0415
    from src.utils import checkpoints  # noqa: PLC0415

    # CV reanudable (``checkpoint_dir``, ver src.utils.checkpoints): run dir fijo dentro
    # del fold; si un corte lo dejó a mitad se reanuda desde su last.pt, y si ya había
    # terminado se toma su best.pt sin reentrenar.
    resume_from: Path | None = None
    on_model_save = None
    fold_ckpt = checkpoints.FoldCheckpoint.from_config(config)
    if fold_ckpt is not None:
        output_dir = fold_ckpt.dir.resolve() / "distillation"
        point, ckpt_path = checkpoints.training_resume_point(output_dir / "distill_student")
        if point == "finished":
            student.config["model"] = str(ckpt_path)
            student._model = None
            return
        if point == "resume":
            resume_from = ckpt_path
        on_model_save = checkpoints.epoch_callback(fold_ckpt, "distilled")

    # ``epochs`` de la sección distillation manda sobre el de la config general, y el
    # LR es el de la etapa (lr0/10 salvo distillation.lr0), no el lr0 general: con
    # ``train_overrides`` a secas la distillation reiniciaba a un estudiante ya
    # entrenado al lr0 completo (AdamW 0.01) y divergía (OKS 0.935 → 0.626).
    train_kwargs = {k: v for k, v in train_overrides(config).items() if k != "epochs"}
    lr = distillation_lr0(config)
    if lr is not None:
        train_kwargs["lr0"] = lr
    train_kwargs["warmup_bias_lr"] = float(dist_cfg.get("warmup_bias_lr", _DEFAULT_WARMUP_BIAS_LR))

    full_distill_cfg = {
        "data": data,
        "epochs": dist_cfg.get("epochs", int(config.get("epochs", 100))),
        "imgsz": int(config.get("imgsz", 640)),
        "alpha": dist_cfg.get("alpha", _DEFAULT_ALPHA),
        "beta": dist_cfg.get("beta", _DEFAULT_BETA),
        "val": dist_cfg.get("val", True),
        "train_kwargs": train_kwargs,
    }

    # Fase 5: el estudiante llega **ya podado en memoria** (etapa 1 de combined).
    # Se entrena esa misma instancia para que la poda sobreviva a la distillation;
    # si no hubiera modelo cargado (p. ej. Fase 4 pura), se destila desde el
    # checkpoint. distill_yolo preserva la arquitectura podada vía get_model.
    student_yolo = None
    if resume_from is None:
        student.load()
        student_yolo = student._model

    best_pt = distill_yolo(
        teacher_weights=teacher_weights,
        student_cfg=student_cfg_path,
        output_dir=output_dir,
        distill_cfg=full_distill_cfg,
        student_yolo=student_yolo,
        resume_from=resume_from,
        on_model_save=on_model_save,
    )

    # Actualizar config del estudiante para que evaluate() use el checkpoint distilado.
    student.config["model"] = str(best_pt)
    student._model = None  # forzar recarga con el nuevo checkpoint
