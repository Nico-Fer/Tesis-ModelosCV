"""``C2f_v2``: variante de ``C2f`` sin ``chunk()``, apta para ``torch_pruning`` (Fase 2).

El módulo ``C2f`` de YOLOv8 usa ``self.cv1(x).chunk(2, 1)`` en su forward; el
tracer de dependencias de ``torch_pruning`` no infiere correctamente las
dependencias de canales a través de ese ``chunk``, lo que rompe la poda
estructurada. ``C2f_v2`` es **matemáticamente equivalente** pero reemplaza la
única conv de ``2c`` canales (luego partida en dos) por **dos convs separadas**
(``cv0`` y ``cv1`` de ``c`` canales cada una), eliminando el ``chunk`` y haciendo
el grafo totalmente trazable.

Este código vive en su propio módulo (con imports de torch/ultralytics a nivel de
módulo) por dos motivos:

* **Picklabilidad**: Ultralytics serializa el modelo al guardar ``best.pt`` durante
  el fine-tuning post-poda. Una clase definida a nivel de módulo es picklable
  (``src.optimization._c2f_v2.C2f_v2``); una definida dentro de una función no lo
  sería.
* **Aislamiento de venv**: ``src/optimization/pruning.py`` debe seguir importándose
  en venv-tf (sin torch). Importa este módulo de forma **diferida**, dentro de las
  funciones que sólo corren en venv-torch.

Referencia: enfoque del ejemplo oficial de poda de YOLOv8 de ``torch_pruning``
(VainF), adaptado a la familia pose.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from ultralytics.nn.modules import Bottleneck, C2f, Conv


class C2f_v2(nn.Module):
    """``C2f`` equivalente que reemplaza el ``chunk()`` por dos convs (``cv0``/``cv1``).

    La salida es idéntica a la del ``C2f`` original (salvo ruido de float) cuando
    los pesos se transfieren con :func:`transfer_weights`.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            # k=((3,3),(3,3)) replica exactamente lo que hace el C2f de Ultralytics
            # (su firma tipa k como tuple[int,int], pero acepta el par anidado).
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)  # type: ignore[arg-type]
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_weights(c2f: C2f, v2: C2f_v2) -> None:
    """Copia los pesos de un ``C2f`` a un ``C2f_v2`` equivalente (exacto).

    ``cv2`` y los bloques ``m`` se reusan tal cual; la conv ``cv1`` original (de
    ``2c`` canales) se parte por la mitad en ``cv0`` (primeros ``c``) y ``cv1``
    (últimos ``c``), replicando el ``chunk(2, 1)`` del forward original. Los BN
    nuevos heredan además ``eps``/``momentum`` del original: sin eso no es exacto.
    """
    v2.cv2 = c2f.cv2
    v2.m = c2f.m

    sd = c2f.state_dict()
    c = v2.c
    v2.cv0.conv.weight.data.copy_(sd["cv1.conv.weight"][:c])
    v2.cv1.conv.weight.data.copy_(sd["cv1.conv.weight"][c:])
    for attr in ("weight", "bias", "running_mean", "running_var"):
        val = sd[f"cv1.bn.{attr}"]
        getattr(v2.cv0.bn, attr).data.copy_(val[:c])
        getattr(v2.cv1.bn, attr).data.copy_(val[c:])
    nbt = sd["cv1.bn.num_batches_tracked"]
    v2.cv0.bn.num_batches_tracked.data.copy_(nbt)
    v2.cv1.bn.num_batches_tracked.data.copy_(nbt)
    # eps/momentum no viven en el state_dict. Los BN de cv0/cv1 nacen con los defaults
    # de PyTorch (1e-5 / 0.1), pero el checkpoint trae los de ``initialize_weights`` de
    # Ultralytics (1e-3 / 0.03). Sin copiarlos la salida NO es idéntica: medido, −0.0013
    # de OKS sobre yolov8n-pose antes de podar (scripts/diagnose_pruning.py, B1 vs B2).
    for bn in (v2.cv0.bn, v2.cv1.bn):
        bn.eps = c2f.cv1.bn.eps
        bn.momentum = c2f.cv1.bn.momentum


def replace_c2f_with_v2(module: Any) -> None:  # noqa: ANN401
    """Reemplaza recursivamente, **in-place**, cada ``C2f`` por un ``C2f_v2``.

    Preserva los atributos que el parser de Ultralytics asigna a cada capa
    (``i``/``f``/``type``/``np``), necesarios para ``BaseModel._predict_once``.
    """
    for name, child in module.named_children():
        if isinstance(child, C2f):
            c1 = child.cv1.conv.in_channels
            c2 = child.cv2.conv.out_channels
            n = len(child.m)
            shortcut = bool(child.m[0].add) if n > 0 else False
            v2 = C2f_v2(c1, c2, n, shortcut, e=child.c / c2)
            transfer_weights(child, v2)
            # Un módulo nuevo nace en modo train: sin esto, reemplazar sobre un modelo
            # en eval dejaba los BN de cv0/cv1 normalizando con stats del batch.
            v2.train(child.training)
            for attr in ("i", "f", "type", "np"):
                if hasattr(child, attr):
                    setattr(v2, attr, getattr(child, attr))
            setattr(module, name, v2)
        else:
            replace_c2f_with_v2(child)
