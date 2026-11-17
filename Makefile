.DEFAULT_GOAL := help

# Ajustar si Python 3.11 se invoca de otra forma en el sistema (p. ej. "py -3.11" en Windows).
PYTHON := python3.11

# ── Setup ──────────────────────────────────────────────────────────────────────

.PHONY: setup-torch
setup-torch: ## Crea venv-torch e instala dependencias PyTorch/Ultralytics (Fases 1/4/5)
	$(PYTHON) -m venv venv-torch
	venv-torch/Scripts/pip install --upgrade pip
	venv-torch/Scripts/pip install -r requirements-base.txt -r requirements-torch.txt

# El `-c constraints-venv-torch.txt` NO es opcional: sin el, resolver desde el
# indice de PyTorch puede arrastrar un upgrade de numpy/protobuf/TF y clobberear
# el toolchain de export TFLite (onnx2tf 1.22 <-> TF 2.15), que es lo mas fragil
# del entorno. Mismas versiones que la build CPU, sólo cambia el build.
.PHONY: setup-torch-cuda
setup-torch-cuda: ## Reemplaza torch/torchvision de venv-torch por la build CUDA (RTX 3090)
	venv-torch/Scripts/pip install -c constraints-venv-torch.txt \
		--index-url https://download.pytorch.org/whl/cu126 \
		--extra-index-url https://pypi.org/simple \
		torch==2.12.1+cu126 torchvision==0.27.1+cu126
	@echo "--- Verificacion de humo: cuda debe ser True ---"
	venv-torch/Scripts/python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin GPU')"

.PHONY: setup-tf
setup-tf: ## Crea venv-tf e instala dependencias TensorFlow/MediaPipe (Fases 2/3 TF)
	$(PYTHON) -m venv venv-tf
	venv-tf/Scripts/pip install --upgrade pip
	venv-tf/Scripts/pip install -r requirements-base.txt -r requirements-tf.txt

.PHONY: setup-mmpose
# PIP_CONSTRAINT es OBLIGATORIO en cada linea: sin el, `mim install mmdet`
# arrastra opencv-python 5.x -> numpy>=2 -> rompe torch/mmcv (ver
# constraints-mmpose.txt). Cada receta de make corre en su propia shell, asi que
# la variable se exporta por comando y no una sola vez arriba.
setup-mmpose: export PIP_CONSTRAINT = constraints-mmpose.txt
setup-mmpose: ## Crea venv-mmpose con ViTPose-H + RTMDet (pseudo-labeling, requiere GPU)
	$(PYTHON) -m venv venv-mmpose
	venv-mmpose/Scripts/pip install --upgrade pip
	venv-mmpose/Scripts/pip install torch==2.1.2 torchvision==0.16.2 \
		--index-url https://download.pytorch.org/whl/cu121
	venv-mmpose/Scripts/pip install -r requirements-mmpose.txt
	venv-mmpose/Scripts/mim install mmengine
	venv-mmpose/Scripts/mim install "mmcv==2.1.0"
	venv-mmpose/Scripts/mim install "mmdet==3.3.0"
	venv-mmpose/Scripts/mim install "mmpose==1.3.2"
	@echo "--- Verificacion de humo: numpy debe ser 1.26.x y CUDA True ---"
	venv-mmpose/Scripts/python -c "import numpy, torch, mmcv, mmdet, mmpose; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

.PHONY: teachers
teachers: ## Descarga los checkpoints teacher (ViTPose-H + RTMDet) a models/teachers/
	venv-mmpose/Scripts/mim download mmpose --config td-hm_ViTPose-huge_8xb64-210e_coco-256x192 --dest models/teachers
	venv-mmpose/Scripts/mim download mmdet --config rtmdet_m_8xb32-300e_coco --dest models/teachers
	@echo "--- Copiar los nombres REALES de config/checkpoint al YAML de config ---"
	@ls -1 models/teachers

# ── Infraestructura ────────────────────────────────────────────────────────────

.PHONY: mlflow
mlflow: ## Arranca el servidor MLflow en http://127.0.0.1:5000
	mlflow server \
		--backend-store-uri ./mlruns \
		--host 127.0.0.1 \
		--port 5000

# ── Limpieza ───────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Elimina __pycache__ y archivos .pyc
	find . -type d -name "__pycache__" -not -path "./venv-*" -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./venv-*" -delete

# ── Ayuda ──────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Lista los targets disponibles con descripción
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
