"""Regresión de los dos bugs del camino de poda hallados el 2026-09-29.

1. ``C2f → C2f_v2`` debe ser **exacto**: los BN nuevos de ``cv0``/``cv1`` heredan
   ``eps``/``momentum`` del ``C2f`` original (antes nacían con los defaults de PyTorch
   y la cirugía costaba −0.0013 de OKS sin haber podado nada).
2. ``pose_eval.run_predictions`` no debe modificar un ``YOLO`` en memoria:
   ``YOLO.predict`` fusiona Conv+BN *in-place*, y en ``combined.py`` eso dejaba al
   estudiante podado sin BatchNorm antes del fine-tune.

Cada test verifica además su **premisa** (que el defecto existiría sin el arreglo),
para que no pase en verde por no ejercitar nada.

A diferencia del resto de ``tests/``, requiere **venv-torch** (torch + ultralytics).
No descarga pesos: el modelo se construye desde el YAML de Ultralytics.

Ejecutar (con ``PYTHONPATH`` en la raíz del repo)::

    venv-torch/Scripts/python.exe tests/test_pruning_fixes.py
    pytest tests/test_pruning_fixes.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f
from ultralytics.utils.torch_utils import initialize_weights

from src.optimization._c2f_v2 import C2f_v2, replace_c2f_with_v2


def _trained_like_c2f() -> nn.Sequential:
    """``C2f`` con BN como los de un checkpoint de Ultralytics y stats no triviales.

    ``running_var`` chico (1e-4–1e-2) a propósito: es donde eps=1e-5 vs 1e-3 cambia
    la normalización de forma apreciable.
    """
    torch.manual_seed(0)
    seq = nn.Sequential(C2f(16, 32, n=2, shortcut=True))
    initialize_weights(seq)  # eps=1e-3, momentum=0.03 (lo que trae un checkpoint real)
    for m in seq.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(1e-4, 1e-2)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.5, 0.5)
    return seq.eval()


def _rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """``max|a-b| / max|b|``: las varianzas chicas inflan la escala de las activaciones."""
    return float((a - b).abs().max() / b.abs().max())


def test_c2f_v2_is_exact() -> None:
    seq = _trained_like_c2f()
    x = torch.randn(2, 16, 32, 32)
    with torch.no_grad():
        y_ref = seq(x)
        replace_c2f_with_v2(seq)
        assert isinstance(seq[0], C2f_v2), "premisa: el reemplazo no ocurrió"
        y_v2 = seq(x)
    for bn in (seq[0].cv0.bn, seq[0].cv1.bn):
        assert (bn.eps, bn.momentum) == (1e-3, 0.03), (bn.eps, bn.momentum)
    rel = _rel_diff(y_v2, y_ref)
    assert rel < 1e-5, f"C2f_v2 no es equivalente: max|Δ|/max|y|={rel:.3g}"


def test_c2f_v2_premise_default_eps_would_differ() -> None:
    """Sin heredar eps (defaults de PyTorch) la salida cambia: el test de arriba mide algo."""
    seq = _trained_like_c2f()
    x = torch.randn(2, 16, 32, 32)
    with torch.no_grad():
        y_ref = seq(x)
        replace_c2f_with_v2(seq)
        for bn in (seq[0].cv0.bn, seq[0].cv1.bn):
            bn.eps = 1e-5
        y_bug = seq(x)
    assert _rel_diff(y_bug, y_ref) > 1e-3, "premisa rota: eps no afecta la salida"


def _n_bn(module: nn.Module) -> int:
    return sum(isinstance(m, nn.BatchNorm2d) for m in module.modules())


def _pose_yolo_from_yaml():  # noqa: ANN202
    """yolov8n-pose sin descargar pesos; ``kpt_shape`` lo asigna el trainer en un .pt real."""
    from ultralytics import YOLO

    yolo = YOLO("yolov8n-pose.yaml")
    yolo.model.kpt_shape = [17, 3]
    return yolo


def _tiny_image(tmp: Path) -> Path:
    img = tmp / "frame.jpg"
    cv2.imwrite(str(img), np.random.default_rng(0).integers(0, 255, (96, 96, 3), dtype=np.uint8))
    return img


def test_run_predictions_leaves_in_memory_model_untouched() -> None:
    from src.evaluation import pose_eval

    yolo = _pose_yolo_from_yaml()
    bn0 = _n_bn(yolo.model)
    params0 = sum(p.numel() for p in yolo.model.parameters())
    assert bn0 > 0
    with tempfile.TemporaryDirectory() as tmpdir:
        img = _tiny_image(Path(tmpdir))
        out = pose_eval.run_predictions(yolo, [img], imgsz=64)
    assert img in out
    assert _n_bn(yolo.model) == bn0, (
        f"run_predictions fusionó el modelo del llamador ({bn0}→{_n_bn(yolo.model)} BN)"
    )
    assert sum(p.numel() for p in yolo.model.parameters()) == params0


def test_run_predictions_premise_predict_fuses_in_place() -> None:
    """``YOLO.predict`` a secas SÍ fusiona in-place: por eso run_predictions no lo usa directo."""
    yolo = _pose_yolo_from_yaml()
    assert _n_bn(yolo.model) > 0
    with tempfile.TemporaryDirectory() as tmpdir:
        yolo.predict(source=str(_tiny_image(Path(tmpdir))), imgsz=64, verbose=False)
    assert _n_bn(yolo.model) == 0, "premisa rota: predict ya no fusiona in-place"


# --------------------------------------------------------------------------- #
# Runner sin pytest
# --------------------------------------------------------------------------- #


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
