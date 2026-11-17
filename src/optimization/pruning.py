"""Poda estructurada de modelos de pose estimation (Fase 2).

Expone dos niveles de API:

``prune_yolo_model``
    Función orientada a archivos: recibe un ``.pt``, poda y escribe el modelo
    podado (objeto recargable) en ``output_path``.

``apply_structured_pruning`` / ``fine_tune_after_pruning``
    Wrappers orientados a instancias de :class:`~src.models.base.PoseModel`. Los
    runners de Fase 2 y 5 los llaman sin cambios. El dispatch por
    ``config["family"]`` (mismo patrón que ``registry.build_model``) mantiene los
    runners agnósticos al modelo: ``yolo`` usa poda real con ``torch_pruning``;
    las familias pre-entrenadas y congeladas (``movenet`` / ``blazepose``) hacen
    un no-op grácil, igual que su ``train()``.

Método de poda (YOLO) — **estructurada** real con ``torch_pruning``:
    Elimina canales/filtros completos siguiendo el grafo de dependencias
    (DepGraph), reduciendo parámetros, FLOPs y —tras exportar— tamaño y latencia
    en el dispositivo (a diferencia de la poda *unstructured*, que sólo pone pesos
    a cero sin cambiar dimensiones).

    Particularidades de integración con Ultralytics 8.3:

    * El ``chunk()`` de ``C2f`` rompe el tracer ⇒ se reemplaza cada ``C2f`` por un
      ``C2f_v2`` equivalente (ver :mod:`src.optimization._c2f_v2`) antes de podar.
    * El tracer de ``torch_pruning`` es por autograd ⇒ hay que habilitar
      ``requires_grad`` (los checkpoints vienen congelados) y poner el modelo en
      modo ``train`` (así el head devuelve los feature maps crudos, trazables).
    * El head de detección/pose se excluye (``ignored_layers``) para no alterar
      las dimensiones de salida (keypoints / clases).
    * El fine-tuning post-poda **no puede** pasar por el ``YOLO.train`` normal: éste
      reconstruye el modelo desde el YAML (arquitectura original) y desharía la
      poda. Se usa un trainer custom cuyo ``get_model`` devuelve el módulo ya
      podado (``setup_model`` lo respeta por ser ya un ``nn.Module``).

Entornos requeridos:
    - YOLOv8-pose  → **venv-torch** (Ultralytics + torch + torch_pruning).
    - MoveNet / BlazePose → modelos congelados, no se podan en este pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.models.base import PoseModel
from src.utils import strict

logger = logging.getLogger(__name__)


def _prune_yolo_module(nn_model: Any, sparsity: float, imgsz: int) -> None:  # noqa: ANN401
    """Poda estructurada **in-place** del ``torch.nn.Module`` de un YOLOv8-pose.

    Args:
        nn_model: ``torch.nn.Module`` subyacente (``YOLO().model``). Se modifica
            en el sitio (incluida la cirugía ``C2f`` → ``C2f_v2``).
        sparsity: fracción de canales a eliminar por capa (0.0–1.0).
        imgsz: lado de la imagen de entrada usada para trazar el grafo.
    """
    import torch  # noqa: PLC0415
    import torch_pruning as tp  # noqa: PLC0415
    from ultralytics.nn.modules.head import Detect  # noqa: PLC0415

    from src.optimization._c2f_v2 import replace_c2f_with_v2  # noqa: PLC0415

    # 1. Cirugía C2f → C2f_v2 (elimina el chunk() que el tracer no soporta).
    replace_c2f_with_v2(nn_model)

    # 2. El tracer de torch_pruning es por autograd: habilitar grad + modo train.
    for p in nn_model.parameters():
        p.requires_grad_(True)
    nn_model.train()

    device = next(nn_model.parameters()).device
    example_inputs = torch.randn(1, 3, imgsz, imgsz, device=device)

    # El head (Pose subclasea Detect) no se poda: alteraría las dims de salida.
    ignored_layers = [m for m in nn_model.modules() if isinstance(m, Detect)]

    importance = tp.importance.GroupMagnitudeImportance()
    pruner = tp.pruner.GroupNormPruner(
        nn_model,
        example_inputs,
        importance=importance,
        pruning_ratio=sparsity,
        ignored_layers=ignored_layers,
    )
    pruner.step()
    nn_model.eval()


def _build_pruned_trainer_cls(pruned_module: Any) -> type:  # noqa: ANN401
    """Subclase de ``PoseTrainer`` cuyo ``get_model`` devuelve el módulo ya podado.

    Evita que ``YOLO.train`` reconstruya el modelo desde el YAML (lo que desharía
    la poda). ``BaseTrainer.setup_model`` no rehace nada al ver un ``nn.Module``.

    Al **reanudar** (``resume=True``), Ultralytics llama a ``get_model`` con los pesos
    EMA de ``last.pt`` (misma arquitectura podada) y espera que el modelo los traiga:
    se copian sobre el módulo podado. En un arranque normal ``weights`` es el propio
    módulo y no hay nada que copiar.

    Además se descarta la loss cacheada (``criterion``): un módulo salido de un
    ``last.pt`` la trae creada por la validación por epoch, con sus tensores en CPU
    (Ultralytics carga los checkpoints a CPU) y los hiperparámetros de la corrida
    anterior. Medido: reanudar sin esto revienta en el primer batch con
    ``Expected all tensors to be on the same device``. Ultralytics la reconstruye en el
    primer ``loss()``, ya en el device y con los args de este entrenamiento.
    """
    from ultralytics.models.yolo.pose import PoseTrainer  # noqa: PLC0415

    from src.utils.checkpoints import Fp32OptimizerCheckpointMixin  # noqa: PLC0415

    # El mixin guarda el optimizer en fp32 en last.pt: en fp16 la reanudación diverge.
    class PrunedPoseTrainer(Fp32OptimizerCheckpointMixin, PoseTrainer):  # type: ignore[misc, valid-type]
        def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> Any:  # noqa: ANN401
            if weights is not None and weights is not pruned_module:
                pruned_module.load_state_dict(weights.float().state_dict())
            pruned_module.criterion = None
            return pruned_module

    return PrunedPoseTrainer


def prune_yolo_model(
    model_path: str | Path,
    output_path: str | Path,
    sparsity: float = 0.3,
    imgsz: int = 640,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Aplica poda estructurada a un checkpoint YOLOv8-pose y lo guarda.

    Carga el checkpoint, poda el módulo en memoria vía :func:`_prune_yolo_module`
    y guarda un checkpoint **recargable** (el objeto modelo completo, no un
    ``state_dict`` suelto, para que la arquitectura de canales reducidos se
    preserve).

    Nota: el objeto guardado contiene módulos ``C2f_v2``; recargarlo requiere que
    ``src.optimization._c2f_v2`` sea importable (lo es dentro de este repo).

    Args:
        model_path: path al checkpoint ``.pt`` a podar.
        output_path: path donde se guarda el checkpoint podado.
        sparsity: fracción de canales a eliminar (0.0–1.0, default 0.3 = 30 %).
        imgsz: tamaño de imagen para trazar el grafo de dependencias.
        cfg: configuración extra (reservado).

    Returns:
        :class:`~pathlib.Path` al checkpoint podado en ``output_path``.

    Raises:
        ImportError: si se invoca fuera de venv-torch.
        FileNotFoundError: si ``model_path`` no existe.
    """
    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {model_path}")

    yolo = YOLO(str(model_path))
    _prune_yolo_module(yolo.model, sparsity=sparsity, imgsz=imgsz)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": yolo.model}, str(output_path))
    return output_path


def apply_structured_pruning(model: PoseModel, config: dict[str, Any]) -> PoseModel:
    """Aplica poda estructurada al modelo y devuelve la instancia podada.

    Dispatch por familia (patrón de ``registry.build_model``):

    * ``yolo``: carga el modelo y poda **in-place** el módulo subyacente con
      ``torch_pruning`` (sin round-trip a disco). El resto del pipeline
      (fine-tuning, evaluate, export) opera sobre el modelo ya podado en memoria.
    * ``movenet`` / ``blazepose``: modelos Google pre-entrenados y congelados → no-op
      grácil (coherente con su ``train()``).

    Si el checkpoint no existe en disco se aborta la fase: "podar" un modelo
    inexistente devolvía la instancia intacta y el run seguía como si la Fase 2
    hubiera corrido. Sólo en smoke test declarado se retorna sin modificar.

    Args:
        model: instancia de :class:`~src.models.base.PoseModel` ya construida.
        config: configuración de la fase, incluyendo la sección ``pruning``.

    Returns:
        La misma instancia de :class:`~src.models.base.PoseModel`, podada in-place
        cuando la familia y el checkpoint lo permiten.
    """
    family = str(config.get("family", ""))
    if family != "yolo":
        logger.info("Poda estructurada no aplicable a la familia %r; se omite.", family)
        return model

    pruning_cfg: dict[str, Any] = config.get("pruning", {})
    sparsity: float = float(pruning_cfg.get("sparsity", 0.3))
    imgsz: int = int(config.get("imgsz", 640))

    model_path = Path(config.get("model", ""))
    if not model_path.exists():
        strict.fail(
            f"No existe el checkpoint a podar ({model_path}); sin pesos reales la poda "
            "es un no-op y la fase no produce el modelo optimizado que declara."
        )
        return model

    model.load()
    yolo = model._model
    assert yolo is not None
    _prune_yolo_module(yolo.model, sparsity=sparsity, imgsz=imgsz)
    logger.info("Poda estructurada aplicada (sparsity=%.2f) a %s.", sparsity, model_path)

    return model


def fine_tune_after_pruning(model: PoseModel, config: dict[str, Any]) -> None:
    """Fine-tunea el modelo podado para recuperar la precisión perdida.

    Usa ``config["pruning"]["fine_tune_epochs"]`` épocas y un learning rate
    reducido a 1/10 del base (``config["lr0"]``, default 0.01). Para familias sin
    entrenamiento (MoveNet/BlazePose) es un no-op grácil; para ``yolo`` sin
    dataset aborta (el modelo podado sin recuperación no sirve como resultado).

    Para YOLO el fine-tuning usa un trainer custom (ver
    :func:`_build_pruned_trainer_cls`) que preserva la arquitectura podada; de lo
    contrario ``YOLO.train`` reconstruiría el modelo desde el YAML y desharía la poda.

    Args:
        model: :class:`~src.models.base.PoseModel` ya podado.
        config: configuración de la fase; ``pruning.fine_tune_epochs`` controla la
            duración y ``lr0`` el LR base.
    """
    pruning_cfg: dict[str, Any] = config.get("pruning", {})
    fine_tune_epochs: int = int(pruning_cfg.get("fine_tune_epochs", 20))
    ft_lr: float = float(config.get("lr0", 0.01)) / 10.0

    family = str(config.get("family", ""))
    if family == "yolo":
        if config.get("data"):
            _fine_tune_yolo_pruned(model, config, epochs=fine_tune_epochs, lr0=ft_lr)
            return
        strict.fail(
            "Fine-tuning post-poda sin 'data': un modelo podado sin recuperación pierde "
            "precisión y no es un resultado válido de la Fase 2."
        )

    # Camino genérico (otras familias / smoke test sin dataset).
    ft_config = {**config, "epochs": fine_tune_epochs, "lr0": ft_lr}
    model.config = ft_config
    model.train()
    model.config = config


def save_pruned_checkpoint(yolo: Any, out_pt: Path) -> Path:  # noqa: ANN401
    """Guarda el módulo podado en memoria a un ``.pt`` recargable (para exportarlo).

    Se guarda el objeto módulo completo (no un ``state_dict``) para preservar la
    arquitectura de canales reducidos, igual que :func:`prune_yolo_model`.
    Exportar/evaluar desde una copia en disco evita reusar el wrapper recién
    entrenado (revienta en NMS y en un segundo ``train()``, ver combined.py).
    Recargar el checkpoint requiere ``src.optimization._c2f_v2`` importable.
    """
    import torch  # noqa: PLC0415

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": yolo.model}, str(out_pt))
    return out_pt


def _fine_tune_yolo_pruned(
    model: PoseModel, config: dict[str, Any], epochs: int, lr0: float
) -> None:
    """Fine-tunea un YOLO podado con un trainer que conserva la arquitectura podada.

    Con ``checkpoint_dir`` en la config (CV reanudable, ver :mod:`src.utils.checkpoints`)
    el run va a un directorio fijo del fold y, si un corte lo dejó a mitad, se reanuda
    desde su ``last.pt`` en vez de reiniciarse; si ya había terminado, se toma su
    ``best.pt`` sin reentrenar.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    from src.models.yolo import augmentation_overrides, train_overrides  # noqa: PLC0415
    from src.utils import checkpoints  # noqa: PLC0415

    # Mismos kwargs de entrenamiento que YoloPoseModel.train (device/batch/workers/…
    # y la sección ``augmentation``), con lr0 y epochs propios de la etapa de
    # recuperación pisando los de la config.
    overrides = train_overrides(config)
    overrides.update(augmentation_overrides(config))
    overrides.update(epochs=epochs, lr0=lr0)
    # Absoluto: evita que Ultralytics use su runs_dir global (fuera del repo).
    project = Path(config.get("project", "runs/pose")).resolve()

    fold_ckpt = checkpoints.FoldCheckpoint.from_config(config)
    if fold_ckpt is not None:
        from ultralytics.utils import SETTINGS  # noqa: PLC0415

        # Estos caminos no pasan por YoloPoseModel.load(), que es quien apaga el
        # callback MLflow de Ultralytics (cambiaría el experimento activo del run).
        dict.__setitem__(SETTINGS, "mlflow", False)
        project = fold_ckpt.dir.resolve()
        point, ckpt_path = checkpoints.training_resume_point(project / "pruned_finetune")
        if point == "finished":
            # Mismo estado que deja ``YOLO.train`` al terminar: el wrapper con best.pt.
            model._model = YOLO(str(ckpt_path))
            return
        if point == "resume":
            resumed = YOLO(str(ckpt_path))
            resumed.add_callback(
                "on_model_save", checkpoints.epoch_callback(fold_ckpt, "pruned_ft")
            )
            # Los hiperparámetros salen del propio last.pt; sólo workers puede cambiar.
            resume_kw = {"workers": overrides["workers"]} if "workers" in overrides else {}
            resumed.train(
                trainer=_build_pruned_trainer_cls(resumed.model), resume=True, **resume_kw
            )
            model._model = resumed
            return

    model.load()
    yolo = model._model
    assert yolo is not None
    if fold_ckpt is not None:
        yolo.add_callback("on_model_save", checkpoints.epoch_callback(fold_ckpt, "pruned_ft"))
    yolo.train(
        trainer=_build_pruned_trainer_cls(yolo.model),
        data=config["data"],
        imgsz=int(config.get("imgsz", 640)),
        project=str(project),
        name="pruned_finetune",
        **overrides,
    )
