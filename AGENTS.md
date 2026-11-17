# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Purpose

ML benchmarking and optimization framework for pose estimation models targeting on-device deployment (Android mid-range, offline-first, TFLite INT8 export). Thesis research project comparing YOLOv8-pose, MoveNet, and BlazePose across a 5-phase optimization pipeline.

**Phase pipeline:**
| Phase | Technique | Target output |
|-------|-----------|---------------|
| 1 | Baseline Float32 | YOLOv8-pose n/s/x, MoveNet, BlazePose |
| 2 | Pruning + fine-tuning | Same families |
| 3 | Post-training quantization (FP32→INT8) | Same families |
| 4 | Knowledge distillation (yolov8x → yolov8n) | Student model |
| 5 | Combined (prune + distill + INT8) | Best Phase 4 student |

## Environment Setup

**Two separate Python 3.11 venvs are required** for the phase pipeline — framework conflicts make a single env impossible. Both install `requirements-base.txt` (shared deps) plus their own framework file.

```bash
# PyTorch/Ultralytics venv (phases 1, 4, 5, and ALL TFLite export)
python3.11 -m venv venv-torch
venv-torch\Scripts\activate          # Windows
pip install -r requirements-base.txt -r requirements-torch.txt

# TensorFlow/MediaPipe venv (phases 2-3 for MoveNet/BlazePose, pruning with tfmot)
python3.11 -m venv venv-tf
venv-tf\Scripts\activate             # Windows
pip install -r requirements-base.txt -r requirements-tf.txt
```

A **third venv, `venv-mmpose`**, exists only for the ViTPose-H auto-labeling script (see Pseudo-labeling below) and is **not** part of the 5-phase pipeline. It does *not* install `requirements-base.txt`: `mmcv` ships as a wheel precompiled against a specific `(torch, cuda)` pair and pulls its own torch, so it is pinned to `torch 2.1.2+cu121` / `mmcv 2.1.0` and kept isolated — installing it into `venv-torch` would clobber the validated TFLite export toolchain. Install order matters (CUDA torch first, then `mim`, never plain `pip install mmcv`); `requirements-mmpose.txt` documents it and `make setup-mmpose` automates it.

**`venv-torch` runs on CUDA (`torch 2.12.1+cu126`, RTX 3090).** Ultralytics resolves torch itself but installs the CPU build, which makes training unusable (~58k frames/fold). After `make setup-torch`, swap in the CUDA build with `make setup-torch-cuda`. That target passes `-c constraints-venv-torch.txt` and it is **not** optional: resolving from the PyTorch index without it can pull a newer numpy/protobuf/TF and clobber the export toolchain. It pins the *same* versions as the CPU build (only the build differs), and `cu126` rather than `cu128` because torch on the cu128 index stops at 2.11. Re-run the export spike after any torch change.

The `Makefile` wraps the above as `make setup-torch` / `make setup-tf` / `make setup-mmpose` (plus `make teachers`, `make mlflow`, `make clean`, `make help`). Note its recipes use `venv-*/Scripts/...` paths — i.e. they assume Git Bash / WSL on Windows, not cmd.exe.

### Lint & type-check

Configured in `pyproject.toml` (ruff line-length 100, rules `E/F/W/I`; mypy non-strict, `ignore_missing_imports`):

```bash
ruff check src/
ruff format src/
mypy src/
```

## Running Experiments

```bash
# Start MLflow tracking server first (or: make mlflow)
mlflow server --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
# In another terminal, set MLFLOW_TRACKING_URI, activate venv-torch, then run a phase:
python -m src.training.train_baseline --config configs/baseline_yolov8n.yaml   # Phase 1
python -m src.training.prune          --config configs/prune_yolov8n.yaml      # Phase 2
python -m src.training.quantize       --config configs/quantize_yolov8n.yaml   # Phase 3
python -m src.training.distill        --config configs/distill_yolov8n.yaml    # Phase 4
python -m src.training.combined       --config configs/combined_yolov8n.yaml   # Phase 5
```

Once an experiment is first created by the tracking **server**, its artifact location is stored with the `mlflow-artifacts:` scheme — later runs must also point `MLFLOW_TRACKING_URI` at the running server, or `log_artifact` fails (a bare `file://./mlruns` run can read it but cannot upload artifacts).

Every runner takes only `--config <yaml>` (there is **no** `--smoke` CLI flag, and no `--no-cv` one either — the cross-validation switch is a YAML key, see below).

To produce the **deployable model** without paying the ~5× of cross-validation, run `configs/combined_yolov8n_deploy.yaml` (`cross_validation: false`, `fold: 0`) instead of `combined_yolov8n.yaml`. Same teacher/student/dataset, one training run, logged to the separate `fase5-deploy` experiment so it never mixes with the CV comparison table.

**The full Phase 5 CV is resumable** (it is ~19 h per fold, ~4 days). Launch *and* resume with the same command, `venv-torch\Scripts\python.exe scripts\run_fase5_cv.py` (starts MLflow if needed, sets `MLFLOW_TRACKING_URI`, tees to `runs/fase5_cv/train.log`). It works because `combined_yolov8n.yaml` declares `checkpoint_dir: runs/fase5_cv` — opt-in, see `src/utils/checkpoints.py` — which checkpoints at three levels: finished **folds** are skipped and the parent/nested MLflow runs are **reopened by `run_id`** (so the mean ± σ still lands in one parent run); finished **stages** are skipped and the next one starts from the artifact on disk (`student_pruned.pt`, `student_pruned_ft.pt`, the distillation `best.pt`, plus the pickled eval context so the person-bbox normalizer is not recomputed); the in-flight **training** resumes from Ultralytics' `last.pt` (`resume=True`). The custom trainers' `get_model` copy the EMA weights of `last.pt` onto the baked pruned module on resume — verified: start epoch, pruned architecture and weights (max diff 0.0) all restored, `results.csv` continuous. Two resume traps found with real `taskkill /F` tests, both fixed in the custom trainers: (1) **Ultralytics saves the optimizer state in fp16** (`convert_optimizer_state_dict_to_fp16`), which zeroes AdamW's `exp_avg_sq` (≈ g² ~1e-9, below fp16's minimum — 46.5 % of entries measured) so the resumed fine-tune went to `box_loss = nan` within ~5 batches; `Fp32OptimizerCheckpointMixin` keeps it fp32 (A/B, same seed: NaN vs 3.39 → 2.58 → 2.30). This affects *any* AdamW resume through Ultralytics, not just pruned models. (2) A module loaded from `last.pt` carries the loss `criterion` cached by per-epoch val, with tensors on CPU (Ultralytics loads checkpoints to CPU) and the previous run's hyperparameters — `get_model` resets it to `None`. Any config change except `workers` aborts the resume (config fingerprint), since mixing recipes in one mean ± σ would be silent. Progress is rendered between markers in `docs/fase5_cv_bitacora.md` (`checkpoint_report`). Not restored on resume: the augmentation/shuffle RNG state and the early-stopping counter — declare it if the run was interrupted.

**The pipeline is fail-fast by default** (`src/utils/strict.py`). Missing `data`, a `data:` whose `*_fold{k}.yaml` siblings don't exist, a checkpoint that isn't on disk, a calibration set under 100 images, a fold that yields no metrics, or a failed TFLite export all raise `PipelineError` instead of degrading the run. This inverts the earlier default: those escapes existed while there was no annotated dataset, and they made a degraded run indistinguishable from a good one in MLflow.

The **only** way to run the skeleton without data is `smoke_test: true` in the YAML (see `configs/smoke_yolov8n.yaml`), which flips the process into smoke mode: `strict.fail` degrades to a warning and the run logs `pipeline_smoke_test = 1.0`. Each runner calls `strict.set_smoke_mode(...)` in its `_run_once`, and `_kfold.run_phase` does too (it deep-copies the config per fold). The flag is process-global on purpose — `src/evaluation/pose_eval.py` never receives the phase config and still has to honour it.

## Architecture

### Abstract base + registry pattern

- **`src/models/base.py`** — `PoseModel` ABC (load, train, evaluate, export_tflite) and `ModelMetrics` dataclass; all model families implement this interface.
- **`src/models/registry.py`** — `build_model(config)` factory; dispatches `config["family"]` to a concrete class via **branch dispatch with deferred per-family imports** (so the module loads in either venv without the other's framework installed). Add new families as a new `if` branch.
- **`src/models/yolo.py`** — `YoloPoseModel`; wraps Ultralytics (trains, evaluates, exports). The most complete family (only one with a real training/distill/export path). Sets `YOLO_AUTOINSTALL=false` at import time to stop Ultralytics from auto-updating the export toolchain (see Critical Dependency Constraints).
- **`src/models/movenet.py`** — `MoveNetModel` (`lightning`/`thunder`). Not trained: downloads the `.tflite` from a config `model_url` to `cache_dir` (default `data/models/`) and runs `tf.lite.Interpreter`. `export_tflite` copies the cached file (or dynamic-range quantizes via `quantization.quantize_tflite`). Interpreter output is `(1,1,17,3)` = `(y,x,conf)` — **Y before X**, normalized 0–1.
- **`src/models/blazepose.py`** — `BlazePoseModel` (`lite`/`full`/`heavy` → `model_complexity` 0/1/2) via the MediaPipe SDK. `export_tflite` raises `NotImplementedError` (SDK doesn't expose the TFLite). Holds `BLAZEPOSE_TO_COCO` (33→17 keypoint map), the single source of truth reused by the normalizer.
- MoveNet/BlazePose `train()` are no-ops (pre-trained Google models); `evaluate()` reports `size_mb` only (MoveNet) or empty (BlazePose) until dataset eval lands.

### Training phase runners

All five runners exist and share the same shape (`parse_args` → `load_config` → `_flatten_params` → `run_*`). They are thin **orchestrators**: they build models via the registry, call into `src/optimization/`, evaluate, and export. `train_baseline.py` is the canonical template.

**GroupKFold wiring (`src/training/_kfold.py`).** Each `run_*(config)` is now a thin wrapper that delegates to `kfold.run_phase(config, _run_once, experiment=…, tags=…)`; the actual per-phase work lives in the module's `_run_once(config)`, which assumes a MLflow run is already active and logs its own params/metrics/artifacts. `run_phase` discovers the sibling `pose_dataset_fold{k}.yaml` of `config["data"]`: with folds it opens a **parent** run and one **nested** run per fold (each on its fold's YAML), then logs `{metric}_mean`/`{metric}_std` across folds to the parent; with **no** folds (single-fold dataset or smoke test) it runs `_run_once` once in a single run — preserving prior behavior. So k-fold multiplies training cost ~k× per model. The `{metric}_mean` above only covers the `ModelMetrics` each fold returns (in Phase 5, the final `int8` stage); `_log_fold_runs_aggregate` additionally reads every metric the nested runs logged back from MLflow and writes `{key}_mean`/`_std`/`_n_folds` to the parent (`oks_pruned_ft_mean`, `latency_ms_base_mean`, …) — before, the per-stage table of the thesis existed only in the nested runs.

**Disabling cross-validation (`cross_validation: false`).** Set it in the YAML — optionally with `fold: k` (default `0`) — and `run_phase` runs `_run_once` on that one fold in a **flat** run (no parent, no nested runs), tagged `cv=single-fold` with `n_folds=1`. It is the way to get the deployable artifact from a single training run. Three things are deliberate:
- It deliberately logs **no** `{metric}_mean`/`_std`/`_n_folds`. A mean over one split with `std = 0.0` is indistinguishable in the MLflow table from a healthy CV result — the exact confusion `_log_aggregate`'s per-metric fold count exists to prevent. A single-split run is therefore *not* comparable with a CV run.
- The branch sits **after** the "no fold YAMLs found" `strict.fail`, so turning off CV never doubles as a way to skip the `data:` validation: a typo still aborts.
- `_select_fold` resolves the fold **by filename** (`{stem}_fold{k}{suffix}`), not by index into the sorted glob — with ≥10 folds `fold10` sorts before `fold2`, and an off-by-order index would silently train on the wrong split. Out-of-range `fold` raises `PipelineError`. Covered by `tests/test_kfold_single.py`.

- **`src/training/train_baseline.py`** — Phase 1. Trains, evaluates, exports TFLite (Float32).
- **`src/training/prune.py`** — Phase 2. Calls `pruning.apply_structured_pruning` + `fine_tune_after_pruning`.
- **`src/training/quantize.py`** — Phase 3. Calls `quantization.prepare_calibration_dataset` + `quantize_to_int8`.
- **`src/training/distill.py`** — Phase 4. Builds a teacher (same `family`, `model=config["teacher_model"]`) and a student; calls `distillation.run_distillation_training`. Requires `teacher_model` in the YAML.
- **`src/training/combined.py`** — Phase 5. Prune → **post-prune fine-tune** → distill → INT8 quantize, in that order. Requires `teacher_model`. Evaluated in **5 stages** (`base`, `pruned`, `pruned_ft`, `distilled`, `int8`).

**Phase 5's training order, and why (`combined.py`).** Pruning runs *before* any training so that both the fine-tune and the distillation train the already-reduced architecture; both use trainers that preserve the pruned channels (`_build_pruned_trainer_cls` / `_build_distill_trainer_cls`), because a plain `YOLO.train` rebuilds the model from the YAML and undoes the pruning. Three things that are easy to get wrong here:
- **`pruned` and `pruned_ft` are separate eval stages on purpose**: the first measures the *damage* of pruning, the second the *recovery*. Collapsed into one metric, a benign prune and an aggressive-but-recovered one look identical. `pruned_ft` logs no latency/size — fine-tuning changes weights, not architecture, so both are `pruned`'s by construction.
- **Never reuse a `YOLO` wrapper after `train()`.** Measured, not theoretical: predicting with the just-trained in-memory object dies in NMS (`'list' object has no attribute 'shape'`), and a *second* `train()` on it dies with `KeyError: 'model'` because the first `train()` replaced `self.overrides` with the checkpoint's args. Both stages therefore persist the checkpoint and force a clean reload (`student.config["model"] = …; student._model = None`) — the idiom `run_distillation_training` already used at its end. This only surfaced once fine-tune was inserted *before* distillation, i.e. once two trainings shared one wrapper.
- `pruning.fine_tune_epochs` used to be **dead** in Phase 5 (only `prune.py` read it) while still being logged to MLflow as a param — a run reported 10 fine-tune epochs having run 0. It is live now.
- **RESOLVED (2026-09-29) — the 4,147-param drop between `params_pruned` (1,907,230) and `params_pruned_ft` (1,903,083) was the Phase 5 student losing all its BatchNorm *before* the fine-tune.** The `pruned` stage passed the in-memory `YOLO` to `pose_eval.run_predictions`; `YOLO.predict` builds `AutoBackend(fuse=True)` on `self.model` and fuses Conv+BN **in place** (net −1 param per BN channel; the pruned model has exactly 4,147 BN channels, 71 layers → 0). So every Phase 5 fine-tune + distillation run before the fix trained a **BN-less** network, and `models/exports/student_pruned_ft.pt` has 0 BN layers. `run_predictions` now predicts on a disk snapshot and never mutates the caller's wrapper (`tests/test_pruning_fixes.py`). Consequences: `params_pruned` (1,907,230) is the correct count; the Phase 5 `pruned_ft`/`distilled`/`int8` numbers produced before the fix (incl. the 0.935 below) come from the BN-less model and should be regenerated before quoting. Phase 2 (`prune.py`, deploy v3/v4) never evaluated between prune and fine-tune, so it was **not** affected. Note `YoloPoseModel.evaluate()`'s `.val()` still fuses in place — harmless because it is always the last step before export, but never train a wrapper after calling `evaluate()` on it.
- **Also fixed then: `C2f_v2` was not exactly equivalent to `C2f`.** Its new `cv0`/`cv1` BN were born with PyTorch's `eps=1e-5`/`momentum=0.1` instead of the checkpoint's `1e-3`/`0.03` (not in the `state_dict`), and in train mode even inside an eval model. Cost −0.0013 OKS before pruning anything; `transfer_weights` now copies both and `replace_c2f_with_v2` preserves the training mode (bit-exact, Δ = 0). Checkpoints pruned before the fix carry the old eps baked in, but their fine-tune ran with that eps, so they are internally consistent — no retrain needed for this one.

**Measured cost on the RTX 3090 (fold 0, `uco_pseudo`, yolov8n student @640).** Useful for sizing any run: training ~233 img/s at batch 24 ⇒ **~4.1 min per epoch** over the 57.8k-image train split; per-image eval is **12.8 ms** for a `.pt` on GPU but **196 ms** for the INT8 `.tflite` on CPU ⇒ a single INT8 eval pass over the 16.1k-image val split costs **~53 min**, about half the fixed cost of a run. `distillation.val: false` skips Ultralytics' per-epoch mAP validation (metrics this pipeline discards — `pose_eval` does the comparable evaluation at the end), at the cost of making `patience` inoperative so every epoch runs.

**The student overtakes the teacher after the post-prune fine-tune.** Measured on the full fold-0 val split: `base` (yolov8n COCO) OKS 0.846 → `pruned` 0.000 → `pruned_ft` **0.935**, at 1.90M params (that 0.935 was the BN-less model; the regenerated run below gives **0.9535** with BN). The COCO teacher `yolov8x-pose` scores **0.911** on the same data (600-frame sample; that sample put `base` within 0.004 of its full-split value, so the gap is real). Worth knowing before budgeting epochs for distillation — but note it is **not** what broke the run below.

### FIXED (2026-09-30) — the distillation stage diverged, and the attention-transfer loss was inert

A full deploy run took the student from OKS **0.935 → 0.626** (PCK 0.997 → 0.753) in 20 distillation epochs, and the resulting model predicted infinite keypoint coordinates (a benign-looking `OverflowError: cannot convert float infinity to integer` from Ultralytics' plotting thread is the visible symptom). Three separate defects, diagnosed with evidence (historical record; the fix and two corrections to this diagnosis follow the list):

1. **The learning rate is 10× too high (this is what caused the damage).** The log is explicit: `pruned_finetune → AdamW(lr=0.001)` but `distill_student → AdamW(lr=0.01)`. `fine_tune_after_pruning` deliberately uses `lr0/10` (`pruning.py`), while `distill_yolo` forwards `train_overrides(config)` untouched — so distillation *restarts a converged model* at the full `lr0`. The training loss climbs monotonically from the very first step (0.648 → 0.864 over two epochs). It is not NaN and not an AMP overflow: all 161 params of the resulting checkpoint are finite.
2. **The attention-transfer term is numerically inert — 0.47 % of the total loss.** `_attention_transfer_loss` does `.pow(2).mean()` over the spatial dimension of already-L2-normalized attention vectors, dividing by H·W (6400 at P3) and crushing the term by 3–4 orders of magnitude. Measured on a real batch: `at_loss` = 1.17e-3, so `(1-α)·T²·at_loss` = 9.4e-3 against a task term of ~2.0. The canonical Zagoruyko & Komodakis form uses the **norm** of the difference, not the mean of squares. **So the run that "proved distillation hurts" was not actually distilling** — it was a plain retrain at 10× LR with the task loss halved by α.
3. **`T²` does not belong on an attention-transfer term.** It comes from Hinton et al., where it compensates the 1/T² gradient scaling of the softened-softmax over *logits*. Attention maps have no softmax and no logits, so the ×16 (T=4) is unjustified.

Defects 2 and 3 cancelled each other, which is why nobody noticed: T² inflated a term the normalization had already flattened. **Fixing 2 alone would swing the failure the other way** — with the norm form, `at_loss` ≈ 1.01, so `(1-α)·T²·at_loss` ≈ 8.1 against a task term of 2.0, i.e. the distillation term would outweigh the task by 4×. Any fix has to address the LR, the normalization and the T² weighting together, then re-tune α.

**The fix (`src/optimization/distillation.py`, `tests/test_distillation_config.py`).** Two corrections to the diagnosis above first: (a) the term was even more inert than 0.47 % — `v8PoseLoss` returns `loss * batch_size`, so the task term is a **batch sum** (~48 at batch 24, not ~2) while `at_loss` is a batch mean: the real share was ~0.04 %; (b) the official Zagoruyko & Komodakis *code* does use `.pow(2).mean()` — it is the missing **β = 1000** (applied as β/2) that brings it back to scale, not a switch to the norm. The loss is now `L = α·L_task + (1−α)·(β/2)·B·L_AT`: β configurable (`distillation.beta`), `×B` to match Ultralytics' batch-sum convention, **no T²** (`distillation.temperature` now raises `PipelineError`, as does any unknown key). The stage LR is `lr0/10` (or `distillation.lr0`), logged as `distillation.lr0_efectivo`. A fifth factor surfaced while verifying: Ultralytics' default `warmup_bias_lr: 0.1` moves every bias ~0.1 per AdamW step on an already-trained student during the first 3 epochs, so the stage now uses `distillation.warmup_bias_lr: 0.0` (bias ramp from 0 like the weights). The trainer logs the real share every 500 batches (`KD step N: … -> imitacion = X% de la loss`). Measured on the v3 pruned+FT student (1 epoch, 10 % of fold-0 train): first-batch share **17 %** (L_AT = 1.3e-3, matching the earlier measurement); OKS 0.943 → 0.929 with the default bias warmup (pose_loss spiked 0.90 → 2.3) and → **0.938** with `warmup_bias_lr: 0` (pose_loss 0.90 → 1.08) — stable, no divergence. α was **not** re-tuned (still 0.5); the small residual drop is consistent with a teacher (0.911) that is weaker than the student (0.943) on this domain.

**Regenerated full deploy run (2026-09-30, `combined_yolov8n_deploy.yaml`, MLflow `fase5-deploy` run `42c1c36d…`, both fixes in; full fold-0 val, 16,149 frames):**

| stage | OKS | PCK | params | TFLite latency (desktop CPU) |
|---|---|---|---|---|
| base | 0.846 | 0.981 | 3,295,470 | 43.9 ms (fp32) |
| pruned | 0.000 | 0.000 | 1,907,230 | 30.5 ms (fp32) |
| pruned_ft | **0.9535** | 0.9959 | 1,907,230 | — |
| distilled | **0.9533** | 0.9962 | 1,907,230 | 30.3 ms (fp32) |
| int8 | **0.9496** | 0.9961 | — (2.39 MB) | 185 ms |

Distillation is now stable (−0.0002 vs 0.935 → 0.626 before) but adds nothing with this teacher, which is weaker than the student on UCO. The int8 latency is the dynamic-range `_int8.tflite` on x86 (see the export traps below), not a phone number. **This artifact is UCO-only (like v2) and FAILS the OOD gate** (measured 2026-10-16 on the rebuilt set, see below): span ratio 1.30 (skeletons too *wide*), keypoint error 0.43 of the GT diagonal, person detected at conf 0.25 in only 43/80 images. Do not ship it over v3.

**Full Phase 5 CV (5 folds, `combined_yolov8n.yaml`: FT 10 + KD 80 with per-epoch val + patience 15), finished 2026-10-14, MLflow `fase5-combined` parent run `11734b62…`; full val split of every fold.** OKS mean ± σ (population): base 0.8552 ± 0.0074 → pruned 0.0000 → pruned_ft 0.9569 ± 0.0076 → distilled 0.9676 ± 0.0063 → **int8 0.9640 ± 0.0062** (PCK@0.2 0.9980 ± 0.0011, 2.39 MB, 1.91 M params). Paired per-fold deltas (95 % t-CI, 4 df): pruned_ft→distilled +0.0107 [+0.007, +0.014]; distilled→int8 −0.0036 [−0.0041, −0.0032]. KD effective epochs per fold 50/80/80/80/80 (only fold 0 early-stopped). The distilled-stage gain is **not** attributable to KD itself (incomplete 10-epoch FT, +50–80 epochs with mosaic, imitation term ~2–3 % of the loss) — an α = 1 control is missing. Full write-up, caveats and the run log (8 resumes) in `docs/fase5_cv_bitacora.md`. The thesis-facing package (tables, paired-delta CIs, curves, figures, per-fold INT8 `.tflite`, provenance) is `resultados/fase5_cv/`, regenerated from `runs/fase5_cv/` + the MLflow file store by `scripts/export_fase5_cv_results.py`; its `README.md`/`RESULTADOS.md`/`METODOLOGIA.md` are hand-written and the script never touches them. Two facts it surfaced: the frame pool is sampled at **stride 15** (not the converter's default 30), and the evaluator's OKS uses the **person-bbox area** as s² (more lenient than COCO's segment area) and is a mean OKS, not COCO AP. **None of the 5 CV models is deployable — all fail the OOD gate like v2** (2026-10-16, rebuilt train2017 set): person detected at conf 0.25 in 13–33/80 images (v3 78/80), keypoint error 0.42 ± 0.09 of the GT diagonal (v3 0.039, pass ≤ ~0.05), 38 % of evaluable images collapsed; the fp32 `best.pt` fails too (not quantization), and val2017 (unseen by all of them) agrees. More UCO-only training ⇒ worse OOD detection across lineages (v2 47/80, deploy Fase 5 43/80, CV 13–33/80), and the COCO teacher did not prevent it. The CV numbers characterize the optimization chain in-domain; v3 stays the shipped artifact. Write-up and a LaTeX fragment for the thesis in `resultados/fase5_cv/gate_ood/`.

Note the corollary for the thesis: `base` is a COCO model and `pruned_ft` is the first checkpoint ever trained on UCO, so the `base → pruned_ft` delta mixes pruning damage, recovery **and domain adaptation**. Attributing it needs the Phase 1 baseline (trained on UCO, unpruned) as the reference point.

**Pruning at `sparsity: 0.3` zeroes the model — and this is real damage, not a bug (verified 2026-09-29 with `scripts/diagnose_pruning.py`).** Measured on the full fold-0 val split (16,150 images): OKS 0.846 → 0.000 — it stops detecting a person at all, rather than degrading. The verification: the pruning path at `sparsity: 0` reproduces the base exactly; physically pruned ≡ the same channels masked in the unpruned model (torch_pruning removes the right indices); it already collapses at **0.05**, and OKS stays ≈0 even at `conf=0.001` (keypoints destroyed, not a threshold effect); no single group breaks it — each costs a little confidence logit and the 47 groups add up; random channel selection also collapses by 0.10. Worth knowing if pruning is ever redesigned: the criterion used (L2 magnitude, `GroupMagnitudeImportance`) is *worse than random* at 2–5 %, while L1 (`MagnitudeImportance(p=1)`) keeps OKS 0.756 at 2 %. The gain that justifies it: params 3.30M → 1.91M (−42 %) and TFLite latency 46.7 → 31.8 ms (−32 %). It is recoverable — that is what the fine-tune and distillation are for — but budget recovery epochs for a model starting from nothing, not for a lightly degraded one. `configs/combined_yolov8n_deploy.yaml` therefore runs `fine_tune_epochs: 25`, not the 10 of the CV config.

### RESOLVED — the deployed model's keypoints collapsed out of domain (deploy_v2 → v3)

`student_pruned_ft_deploy_v2_int8.tflite` scored **OKS 0.931 in-domain** yet rendered a
horizontally-squashed skeleton in the Android app. Diagnosed 2026-07-29 with the new
`scripts/diagnose_kpt_ratio.py`; the cause is **appearance-generalization failure of the
post-prune fine-tune**, not quantization, not the app, not the labels:

- `_int8` ≈ `_float32` ≈ `.pt` on every frame ⇒ not quantization, not the TFLite runtime.
- Full UCO frames and UCO crops re-framed like the phone camera (box_w ≈ 0.53) are healthy
  ⇒ not scale/framing. GT pseudo-label x-span is a uniform ~0.81 across all 5 cameras ⇒
  there is no "narrow-x prior" in the data.
- On **79 held-out COCO train2017 images with real GT** (one large frontal full-body
  person; neither student ever saw them), normalizing the keypoint x-span by the **GT box
  width**: v2 = 0.659 with 31/51 images collapsed and a mean keypoint error of 0.358 of
  the GT diagonal — i.e. the skeleton lands *off the body*, not merely narrow. Overlays
  confirm it visually.

The fix is the **v3 lineage**: prune 0.3 → 25-epoch fine-tune on a **mixed** dataset
(`data/processed/pose_dataset_pseudo_coco_fold0.yaml` = uco_pseudo fold-0 + COCO-pose
val2017 ×8 ≈ 24 % of the mix, manifest from `scripts/make_coco_pose_manifest.py`) with
reinforced augmentation → INT8. `configs/prune_yolov8n_deploy_v3.yaml` +
`configs/quantize_deploy_v3.yaml`. Results, all measured:

| | v2 (was deployed) | **v3 (deploy this)** | stock COCO (ceiling) |
|---|---|---|---|
| GT gate, span pred/GT (79 OOD imgs) | 0.701 | **0.913** | 0.977 |
| collapsed images | 40/76 | 1/79 | 0/79 |
| keypoint error (frac. GT diagonal) | 0.314 | 0.049 | 0.022 |
| OKS in-domain (16,149-frame fold-0 val) | 0.931 | **0.941** | — |

Two methodological notes worth keeping: (1) **normalizing by the *predicted* box is
misleading** — a broken model also predicts a bad box (v2 overestimated box_w ~2.5× OOD),
which inflates the denominator and understates the collapse; normalize by GT. (2)
`bus.jpg` is a weak proxy — not even the stock control exceeds 0.75 there (people are
distant, side-on), so use it qualitatively, never as a threshold.

**Corollary for the thesis and for any future deploy checkpoint:** in-domain OKS does not
cover the deployment distribution. Run `scripts/diagnose_kpt_ratio.py --gt-set <dir>` as a
gate on every deployable artifact, in addition to OKS.

### Diagnostic tooling (`scripts/`)

- **`scripts/diagnose_kpt_ratio.py`** — measures keypoint-span ratios with a **raw decode
  faithful to the app** (`tf.lite.Interpreter`, app letterbox, no Ultralytics) for
  `.tflite`, and via Ultralytics for `.pt`, so weights and export are testable separately.
  Modes: default model×frame matrix (includes the stock-INT8 control; its decode is
  verified when it reproduces the recorded values — 0.564 x-ratio on `bus.jpg`, ~0.6–0.85 on
  UCO frames; it never reaches 0.8 on `bus.jpg`); `--per-camera N` (predicted vs **GT**
  x-span per UCO camera); `--zoom N` (same UCO frame full vs re-cropped to app framing —
  isolates scale from appearance); `--crop-image <img>` (per-person crops, so two models are
  compared on the *same* crop); `--gt-set <dir>` (**the acceptance gate**: `images/` +
  `labels/` with real GT; `--json-out` dumps summary + per-image rows; the logic lives in the
  importable `gate_gt_set()`); `--oks-sample N` (sampled OKS via `pose_eval` helpers).
  No MLflow, runs in venv-torch.
- **The gate's image set.** The original 79-image COCO train2017 set (July 2026) was lost —
  no ID list survived. `scripts/build_ood_gt_set.py` rebuilds it with explicit criteria
  (exactly one non-crowd person, 17 kpts annotated, the 12 body kpts visible, bbox height
  ≥ 50 % of the image, portrait bbox, frontal: left shoulder/hip right of right ones and
  shoulder width ≥ 30 % of bbox width) from the official `person_keypoints_*.json`, writing
  `data/processed/ood_gate_coco/{train2017,val2017}/` (gitignored). The **ID lists are
  versioned** in `data/ood_gate/` so it can't be lost again: train2017 = 80 of 578 candidates
  (seed 0), val2017 = all 37 (v3 trained on val2017, so its val2017 row is not OOD).
  Recalibrated references on the rebuilt train2017 set: v2 0.759 / 33/73 collapsed / err
  0.393; v3 0.945 / 0/80 / 0.039; stock 0.984 / 0/80 / 0.017 — same verdicts as July (v2
  0.701 / 40/76 / 0.314, v3 0.913 / 1/79 / 0.049, stock 0.977 / 0/79 / 0.022). **Report the
  keypoint error and detection at conf 0.25, not just the span ratio:** UCO-only models also
  produce over-*wide* skeletons, so a broken model can average ~1.0. `scripts/
  run_ood_gate_fase5.py` runs the gate over the CV, deploy and reference models.
- **`scripts/diagnose_pruning.py`** — isolates every link of the pruning path on a fixed
  600-frame fold-0 sample (C2f surgery, sparsity 0 control, sparsity sweep at conf 0.25 and
  0.001, BN recalibration, the unstructured-L1 method, in-place fusion). Runs in venv-torch,
  no MLflow, ~15 min on GPU. Post-fix, B1/B3 must equal B0 and B8 must report Δparams = 0.
- **`scripts/make_coco_pose_manifest.py`** — builds the absolute-path manifest of the
  COCO-pose val2017 subset with `--repeat K` oversampling.

### Human audit of pseudo-labels (`src/audit/` + CVAT self-hosted)

Full flow in `docs/cvat_setup.md`. CVAT v2.74.1 runs **isolated in Docker** (clone at
`./cvat/`, gitignored; ports 8080/8090 — no collision with MLflow:5000; superuser creds in
`cvat_credentials.local`, gitignored). Design rationale and measurements in
`docs/plan_cvat_correcciones.md`. Pipeline, framework-free except the REST scripts
(which need only `requests`), config `configs/audit_cvat.yaml`:

1. `select_audit_subset` — `sin_deteccion` / `sin_kpts_validos` / **`discrepancia_gt`**
   / `score_bajo`, plus a stratified control; strata = motivo × postura. `discrepancia_gt`
   compares every pseudo-label against the **UCO GT** (the 3 exercise joints) and flags
   error > 100 px or a GT joint lost to `v=0`. It is the criterion that finds real
   errors: the teacher's `baja_confianza` flag caught only **3 of the 339** frames with
   error > 100 px (measured 2026-09-28). `multi_persona` frames that agree with the GT are
   excluded except a 50-frame sample — their GT error equals the control's (p99 48 vs
   47 px). The flag is recomputed with the thresholds from `uco_pseudo/run_metadata.json`
   and must match the manifest, or it aborts.
2. `pack_for_cvat` — images + an **annotations-only** YOLO Pose ZIP +
   `cvat_skeleton_labels.json` (the 17-kpt COCO skeleton whose **sublabel order must stay
   COCO order** — the importer maps txt points by position). Frame order is shuffled
   (seed) so the annotator cannot tell control frames apart.
3. `scripts/cvat_roundtrip_gate.py` — **gate, blocks the real audit**: mounts 5 frames by
   the real path, exports job by job untouched, runs `verify_roundtrip` (2026-09-28:
   identical visibility, noise floor 0.005 px).
4. `scripts/cvat_audit.py montar|estado|exportar|backup` — one task with
   `segment_size`-frame jobs; `montar` writes the frame→job map (`cvat.frames_csv`,
   refuses to overwrite). **"Audited" = the frame's job is `completed`**, written to
   `auditados.txt` by `exportar`. Never infer it from the export: CVAT exports every
   pre-annotated frame whether reviewed or not, and a reviewed frame whose persons were
   all deleted is **absent** from the export (both verified 2026-09-28).
5. `reconcile_corrections` — reads `auditados.txt`; audit metric + gold set at
   `data/processed/uco_gold/`, rebuilt from scratch every run.
6. `apply_corrections` — `data/processed/uco_pseudo_corrected/` (pseudo pool with gold
   applied, `provenance.csv`), folds and YAMLs incl. `..._gold_fold{k}.yaml` (the human
   subset of each val) and `..._coco_fold0.yaml` (v3 mix, used by
   `configs/prune_yolov8n_deploy_v4.yaml`). Subject→fold is **inherited** from the
   `uco_pseudo` manifests, not re-drawn; membership follows `skip_empty`, so the gold can
   only add frames (annotated `sin_deteccion`) or remove them (emptied) — any other change
   aborts. With `aplicar.usar_en_train: true` held-out is **per subject** (via folds).

Note: the pseudo run used `skip_empty: true`, so the 152 `sin_deteccion` frames are **not
in any manifest** today (fold0 train+val = 73,955 = frames with a person) — they are
missing from training, not trained as background. Other API traps (single-request upload
needs `Upload-Start`+`Upload-Finish` and **no** `filename` query param; the detector bbox
does not survive CVAT — gold bboxes are rebuilt as kpt-hull+5 %) live in
`scripts/cvat_client.py` and `docs/cvat_setup.md`. Tests (both-ways pattern):
`tests/test_audit_subset.py`, `tests/test_audit_cvat_frames.py`,
`tests/test_audit_reconcile.py`, `tests/test_audit_apply.py`.

### Config sections and runner behavior added with v3

- **`augmentation:` (new, one level)** — `src\models\yolo.py:augmentation_overrides()`
  forwards Ultralytics augmentation hyperparams (`scale`, `degrees`, `translate`, `mosaic`,
  `hsv_*`, `fliplr`, `mixup`, `close_mosaic`, `multi_scale`, …) from the YAML. Wired into
  **both** `YoloPoseModel.train` and the post-prune fine-tune, which previously ran on
  Ultralytics defaults (verified in `runs/pose/pruned_finetune6/args.yaml`: `scale 0.5`,
  `degrees 0`, `translate 0.1`). An unknown key raises `PipelineError` — a typo would
  otherwise train for hours with the default augmentation, silently.
- **Multi-source `train:`** — Ultralytics 8.3.253 accepts a **list** of manifest `.txt`
  files in a dataset YAML (`check_det_dataset` resolves list values; `get_img_files` reads
  each and keeps oversampling duplicates). The label `.cache` is named after the **first
  sorted** label dir, which is why COCO lives under `data/processed/coco_pose/` — it sorts
  before `uco_pseudo` and so never clobbers `uco_pseudo/labels/all.cache`.
- **`src\training\prune.py` now persists and reloads** the fine-tuned checkpoint to
  `pruning.save_checkpoint` (default `models/exports/student_pruned_ft.pt`) before
  `evaluate()`, via the shared `pruning.save_pruned_checkpoint()` (lifted out of
  `combined.py`, which now calls the same function). Without it, `evaluate()` reused the
  just-trained wrapper and died in NMS — the failure mode already documented for Phase 5.
  Skipped in smoke mode so a smoke run never overwrites a real deliverable.

Two shared conventions in every runner:
- `pipeline_smoke_test = 1.0` is logged **only** in smoke mode, so its presence on a run means "these metrics are not valid". (It used to be logged unconditionally, which defeated the purpose.)
- Export failures re-raise; the `export_skipped.txt` note artifact is now the smoke-mode fallback only. The exported artifact *is* the deliverable of every phase, so losing it invalidates the run.
- `src/models/yolo.py:train_overrides()` is the single place that forwards training kwargs (`device`, `batch`, `workers`, `amp`, `patience`, `cache`, `fraction`, `optimizer`, `lr0`) from the config to Ultralytics. All three training paths use it — `YoloPoseModel.train`, the post-prune fine-tune, and distillation — because the latter two previously called `yolo.train()` directly and silently ignored those keys (so they never used the GPU settings).
- **`lr0` does nothing unless `optimizer` is also set.** Ultralytics defaults to `optimizer="auto"`, and in that mode `build_optimizer` overwrites the incoming `lr0` with one derived from dataset size (the log says so: *"ignoring 'lr0=0.01' … determining best 'optimizer', 'lr0' and 'momentum' automatically"*). This silently discarded the deliberate lr/10 of the post-prune fine-tune, so `configs/prune_yolov8n.yaml` and `configs/combined_yolov8n.yaml` now declare `optimizer: AdamW` explicitly.
- **Always pass an absolute `project=` to any Ultralytics entry point.** With it unset, Ultralytics resolves the output dir against its *global* `runs_dir`, which may point at an unrelated project — `evaluate()`'s `model.val()` was writing results to `~/Documents/DeepLearningCourse/runs/` until it got the same treatment `train()` already had.

### Optimization techniques

- **`src/optimization/pruning.py`** — file-level `prune_yolo_model` (real logic: L1-unstructured on Conv2d, made permanent) + instance-level wrappers `apply_structured_pruning` / `fine_tune_after_pruning` that the runners call. True structured pruning (torch_pruning DepGraph) is a TODO.
- **`src/optimization/quantization.py`** — `prepare_calibration_dataset`, `quantize_to_int8` (largely stubs).
- **`src/optimization/distillation.py`** — `build_distillation_loss` (returns loss hyperparams logged as MLflow params), `run_distillation_training` (stub).

### Evaluation and utilities

- **`src/evaluation/metrics.py`** — `compute_oks` (COCO sigmas), `compute_pck` (PCK@0.2), `measure_latency` (TFLite interpreter, dtype read from `input_details`) and `model_size_mb`, all wired in. Both keypoint metrics mask on `v > 0`.
- **`src/evaluation/pose_eval.py`** — the unified keypoint evaluator (OKS/PCK, with a leg/arm breakdown) shared by `YoloPoseModel.evaluate` and the stages of phase 5. **`run_predictions` must build `YOLO(path, task="pose")`** — a `.tflite` carries no task in its metadata, so without it Ultralytics warns *"Unable to automatically guess model task, assuming 'task=detect'"* and returns 300 boxes with **no keypoints**. Every frame then scores as a miss and the stage reports OKS 0.0000, which reads as catastrophic quantization loss rather than the load error it is. This silently corrupted **every INT8 metric** the pipeline produced (phase 3 and phase 5's `int8` stage) until it was fixed; the real cost of INT8 on the deploy model is only **−0.005 OKS**. Two things to know: the leg/arm group comes from the **stem** via `uco_to_yolo.parse_stem` (exercises 01–08 = leg, 09/13 = arm) — it used to be inferred from which keypoints were visible, which silently collapses to "all leg" on the dense pseudo labels; and `run_predictions` predicts **one image at a time** because an INT8 `.tflite` has batch fixed at 1, so evaluating a 16k-frame val split is slow.
- **`src/utils/mlflow_utils.py`** — `start_run()` context manager plus `log_params` / `log_metrics` / `log_model_artifact`; centralizes the tracking URI and run lifecycle.
- **`src/data/`** — the UCO → YOLOv8-pose dataset pipeline (see **Dataset pipeline** below), implemented end-to-end and producing a GroupKFold-split YOLO dataset over all 27 in-scope subjects.

### Dataset pipeline (`src/data/`)

Converts the raw **UCO** (a rehab dataset — University of Córdoba) into the Ultralytics YOLOv8-pose layout. UCO annotates **only the exercise-relevant joints per clip** (e.g. just `l_hip/l_knee/l_ankle`), **never the face** — 12 distinct body joints total, which map 1:1 onto COCO-17 slots 5–16; the 5 face slots (0–4) are always written with `visibility=0`. All modules depend only on stdlib + `cv2` (in `requirements-base`), so they import in either venv.

**Per-frame labels come from the `camN_p2d.txt` files, not `dataset_2d.json`.** Each `camN_p2d.txt` is one row per frame (contiguous `id` 0..N-1), columns = `x y` per joint in a fixed per-exercise order, px coords on **1280×720** video. The `.txt` files cover **all 27 subjects** for the in-scope exercises (01–09, 13); `dataset_2d.json` only annotates these exercises for **subject 0** (subjects 1–26 have only exercise 16, whose videos are intentionally out of scope). Since the `.txt` has no joint names, the converter learns the column→joint order per exercise **from the JSON** (`build_exercise_joint_order`, using subject 0, which matches its `.txt` byte-for-byte). Exercise→joints: 01–04 left leg, 05–08 right leg, 09 left arm, 13 right arm (3 kpts each).

- **`uco_to_yolo.py`** — the converter (run it: `python -m src.data.uco_to_yolo`, flags `--dataset-root/--output-root/--output-yaml/--stride/--folds/--seed`). `convert()` does **integrated extraction**: builds the exercise→joint-order map from the JSON, wipes the output dirs (incl. stale `*.cache`), then per clip opens each camera `.mp4`, reads its `camN_p2d.txt` (`parse_p2d_file`; coords ≤0 = out-of-frame → `v=0`), and for every `stride`-th annotated frame writes a `.jpg` **and** its label together (1:1 pairing) into a **flat pool** `images/all` + `labels/all` (one copy). `to_coco17(joints)` maps the UCO joints dict to 17 `(x,y,v)` tuples via `UCO_TO_COCO`. `yolo_label_line()` builds the label (`class cx cy w h  px py v ×17`, normalized 0–1; bbox = 5%-margin box around visible keypoints). Reuses `COCO_KEYPOINT_NAMES`/`NUM_COCO_KEYPOINTS` from `src/inference/keypoint_schema.py`.
- **The stem is a contract.** `make_stem(subject, exercise, cam, frame_idx)` → `"{subject}_{exercise}_cam{cam}_{frame:06d}"` and its inverse `parse_stem(stem) -> StemParts` are the **single definition** of that format. The stem is what pairs `images/all/<stem>.jpg` with `labels/all/<stem>.txt` *and* what recovers the subject for GroupKFold, so never split it by hand — `parse_stem` raises `ValueError` on a mismatch rather than returning a bogus subject. Verified round-trip over all 74,107 pool stems.
- **Split — GroupKFold by subject (k=5).** `assign_subject_folds()` round-robins the subjects into k folds (no leakage: a subject is entirely in train or val). `write_fold_manifests()` writes, per fold, two image-list `.txt` manifests `folds/fold{k}_{train,val}.txt` (absolute paths over the shared pool → **JPEGs are never duplicated**) plus a dataset YAML `pose_dataset_fold{k}.yaml`. Data-driven: only `(subject, exercise)` with both a video and a known joint order are processed.
- **`pseudo_label_vitpose.py`** — auto-labeling with a **ViTPose-H teacher** (see below). Reuses `parse_stem` and `write_fold_manifests` from this module so the pseudo dataset is fold-for-fold comparable with the GT one.
- **`uco_extract.py`** / **`uco_frames.py`** — auxiliary, **not** in the critical path. `uco_extract` reassembles the full dataset's **ZIP64 split into 12 parts** (`dataset.zip.001…012`, ~12 GB). `uco_frames` dumps raw frames (the integrated `convert()` superseded it for labeling).

**Local dataset:** `data/raw/uco/` holds the in-scope subset — **27 subjects**, exercises 01–09 and 13 (arm raise / knee flexion / leg extension), 5 cams — labeled via the `camN_p2d.txt` files. A build at stride 30 yields ~5–8 GB of JPEGs (well under budget; lower `--stride` for more frames).

The converter generates **five** `data/processed/pose_dataset_fold{0..4}.yaml` (each: `kpt_shape: [17, 3]`, `flip_idx` = COCO L/R mirror map, `names: {0: person}`, absolute `path` → `data/processed/uco_yolo`, `train`/`val` → the fold `.txt` manifests). The training runners **auto-discover** the sibling `*_fold{k}.yaml` of whatever `data:` declares (see `src/training/_kfold.py`), so the base file itself need not exist. All five configs now point `data:` at `data/processed/pose_dataset_pseudo.yaml` → the pseudo folds; switch the single line back to `pose_dataset.yaml` to train against the sparse GT labels instead.

### Pseudo-labeling with ViTPose-H (`src/data/pseudo_label_vitpose.py`)

UCO's sparse labels (14 of 17 COCO slots always `v=0`) were a known pain point — `configs/combined_yolov8n.yaml` used to set `alpha: 0.0` specifically to avoid depending on them. Now that this script has run, all five configs point `data:` at the pseudo dataset and `alpha` is back to `0.5` in **both** phases 4 and 5 (same value in both, so the phase-5-vs-phase-4 delta is attributable to pruning + INT8 rather than to the loss weighting). This script produces a **dense-label parallel dataset** over the same images using a top-down teacher: RTMDet-m detects persons → ViTPose-H estimates 17 keypoints per crop. Runs in `venv-mmpose`, `python -m src.data.pseudo_label_vitpose --config configs/pseudo_label_vitpose.yaml`.

Output at `data/processed/uco_pseudo/` mirrors the converter's shape (flat pool, `folds/`, `pose_dataset_pseudo_fold{k}.yaml`), so the phase runners consume it by changing only `data:`. Images are **hardlinked** (`os.link`, NTFS same-volume, no admin needed; falls back to copy with a one-time warning) — Ultralytics derives label paths by substituting `images`→`labels`, which is why a parallel `images/` tree is needed at all.

Deliberate divergences from `yolo_label_line()`: the bbox comes from the **detector** (not the keypoint hull), and all coords are **clamped to [0,1]** (ViTPose extrapolates outside the crop). Everything else is byte-compatible, so the two datasets are interchangeable. This is also the **first place in the repo that emits `v=1`**.

Three things to know before touching thresholds:
- **`det_thr` is the most expensive** and must be calibrated first: a missed detection loses the *whole frame*, not a keypoint, and is unrecoverable without redoing the run.
- **`vis_thr_low` is the only threshold that changes what the student learns** — it decides which keypoints enter the loss. It depends on `det_thr` (changing `det_thr` changes the crop population), so calibrate in that order.
- **`vis_thr_high` does not affect training at all.** Ultralytics binarizes visibility (`v>0` is the loss mask), so `v=1` and `v=2` train identically; it only splits `n_v1`/`n_v2` in the manifest for human-review triage. Don't calibrate it expecting model impact.

`--dump-scores` profiles both models over a stratified sample (no labels written) into `calibration/{det_thr_sweep,score_stats,score_samples}.csv`. Defaults in the YAML are placeholders — both models are COCO-calibrated, and the target population (seated older adults, walker, occluded lower limb) is exactly where their scores drop.

`folds.verify_against` re-derives the subject→fold split from the GT dataset's `fold{k}_val.txt` and **aborts on any difference** — without it, "same seed ⇒ same split" is an assumption, and a divergence would silently invalidate every fold-wise GT-vs-pseudo comparison.

Other guarantees: a person whose 17 keypoints all land at `v=0` is dropped (a bbox-only line gives the kpt-loss nothing) and counted as `sin_keypoints_validos` — a high count is the direct symptom of `vis_thr_low` being too high. `manifest.csv` carries a `baja_confianza` flag per image; `run_metadata.json` records dep versions, seeds and checkpoint MD5s.

### Inference pipeline (`src/inference/`)

Unifies the three families' outputs into **COCO Keypoints JSON** (`keypoints` = 51 floats `[x,y,v]×17`, `bbox` in pixels, visibility `2` if `conf > threshold` else `0`). Framework-agnostic except where it must run a model: only `cv2`/`numpy` (both in `requirements-base`), so it imports cleanly in **either** venv.

- **`keypoint_schema.py`** — `COCO_KEYPOINT_NAMES`, `COCO_SKELETON`, `DEFAULT_VISIBILITY_THRESHOLD`, `NUM_COCO_KEYPOINTS`.
- **`normalizer.py`** — `KeypointNormalizer.normalize(raw_output, source, frame_w, frame_h, ...)` → COCO list. Per source: YOLO (one det/person, px already), MoveNet (invert `(y,x)`→`(x,y)`, scale to px), BlazePose (apply `BLAZEPOSE_TO_COCO`, conf = `.visibility`). Imports **no** ML framework — reads ndarrays/landmark attrs only, so it's unit-testable standalone.
- **`pose_estimator.py`** — `PoseEstimator(model, source)`, a context manager that `model.load()`s on enter; `predict(frame, image_id)` runs the right backend (YOLO `.predict`, MoveNet interpreter with dtype read from `input_details`, BlazePose `.process`) and delegates to the normalizer.
- **`video_pipeline.py`** — `run_inference_on_video(estimator, source, output_json, max_frames, show_preview)` (OpenCV loop → JSON) and `draw_keypoints(frame, detections, skeleton)`.

### Configuration

One YAML per experiment in `configs/` (`baseline_`, `prune_`, `quantize_`, `distill_`, `combined_`). Required top-level keys: `family`, `model` (plus `teacher_model` for distill/combined). Base YOLO checkpoints live in `models/` (configs reference them as `models/yolov8n-pose.pt`; Ultralytics downloads there on first use); TFLite/SavedModel export outputs go to `models/exports/`. Both are gitignored (DVC-tracked) — only the `.gitkeep`s are committed. This is distinct from MoveNet's downloaded-`.tflite` cache, which defaults to `data/models/`. Phase-specific nested sections: `mlflow`, `export` (`format`, `quantize`), `pruning` (`sparsity`, `fine_tune_epochs`), `distillation` (`temperature`, `alpha`, `epochs`). Optional top-level `cross_validation` (default `true`) and `fold` (default `0`) control the k-fold switch. The `_flatten_params()` helper serializes nested keys to dot-notation before logging to MLflow — it only flattens **one level deep**, so keep config nesting shallow.

`configs/combined_yolov8n_deploy.yaml` is `combined_yolov8n.yaml` with `cross_validation: false` / `fold: 0` and `experiment: fase5-deploy`, **but it is not otherwise identical** (it used to claim so): the deploy recipe runs `fine_tune_epochs: 25` (vs 10), `distillation.epochs: 20` (vs 80) and `distillation.val: false` (vs per-epoch val + `patience`). Teacher, student, dataset, `sparsity`, α, β and LRs do match. It is the config to run for the app artifact; `combined_yolov8n.yaml` stays the one that produces the thesis's CV numbers — so those CV numbers describe the CV recipe, not the deployed artifact's exact training.

MLflow conventions:
- Experiment name: `fase{n}-{description}` (e.g., `fase1-baseline`)
- Run name: model name (e.g., `yolov8n-pose`), suffixed `-fold{k}` for per-fold and single-fold runs
- Tags: `{"fase": "1", "familia": "yolo"}`, plus `cv` = `groupkfold` (parent) or `single-fold`

## Critical Dependency Constraints

These version pins exist to avoid known incompatibilities — do not change them without testing both venvs:

- **numpy < 2.0** — required by TensorFlow 2.15, tfmot 0.8, and MediaPipe 0.10
- **protobuf 4.x** — `>=4.25` (MediaPipe floor), `<5` (TF/ONNX ceiling)
- **TensorFlow 2.15** — must use native Keras 2 for tfmot 0.8.x compatibility. If ever bumped to TF ≥ 2.16 (Keras 3 default), you must add `tf-keras` and export `TF_USE_LEGACY_KERAS=1` before importing tfmot, or pruning breaks.
- **TFLite export toolchain (venv-torch)** — Ultralytics export runs PyTorch→ONNX→TF→TFLite, so the full TF stack lives here too. Validated pins (spike OK on yolov8n-pose, Float32 + Float16): `onnx2tf~=1.22.0`, `onnx~=1.16.0`, plus `tf-keras~=2.15.0` and `onnx-graphsurgeon>=0.3.26`. The latter two are **imported unconditionally** by onnx2tf at module load, so they must be installed even though they aren't declared install deps; `tf-keras 2.15.x` matches TF 2.15 and does **not** bump it. `onnx-graphsurgeon` is on PyPI and on the NVIDIA index (`pypi.ngc.nvidia.com`).
- **Ultralytics pinned `~=8.3.0`** — `~=8.3` (no patch) resolves to 8.4.x, whose TFLite export AutoUpdate force-installs a newer toolchain (onnx2tf≥1.26, Keras 3, `ai-edge-litert`, protobuf≥5) that clobbers every pin above. `src/models/yolo.py` also sets `YOLO_AUTOINSTALL=false` on import so Ultralytics can never silently mutate the venv — if an export dep is missing, add it to `requirements-torch.txt` rather than letting AutoUpdate install it.

## Two TFLite export traps (Ultralytics 8.3.253)

- **The shipped `*_int8.tflite` is NOT full-integer INT8.** `ultralytics/utils/export/tensorflow.py` renames `*_dynamic_range_quant.tflite` → `*_int8.tflite`: INT8 weights, but float32 activations and float32 I/O. Consequences: (a) the ~300 calibration images (`fraction`, taken as the **first N filenames alphabetically** of the val manifest — not a random sample) do **not** influence the delivered bytes at all; they only shape `*_integer_quant.tflite` / `*_full_integer_quant.tflite`, which the pipeline never returns and `pose_eval` never scores; (b) if the thesis describes Phase 3 as "PTQ INT8 with a calibration set", that is inaccurate for what ships. The upside: float32 I/O is exactly the contract the Android app expects.
- **A smoke run overwrites the stock control.** `configs/smoke_yolov8n.yaml` uses `model: models/yolov8n-pose.pt`, and Ultralytics writes exports next to the source `.pt`, so any smoke run of a runner with `export.format: tflite` regenerates `models/yolov8n-pose_saved_model/` — clobbering the stock `yolov8n-pose_int8.tflite` that both the app comparison and `scripts/diagnose_kpt_ratio.py` use as their control, and leaving a *pruned* 7.7 MB float32 under a stock filename. Byte-identical copies to restore from: the INT8 in `mlruns/545531124405737930/035c95c6c0f84077b29caa32ef48840a/artifacts/model/` (MD5 `EE27DCF9F0AD0C25FCF37868E097AD68`) and the floats in `models/exports/yolov8n-pose_saved_model/`. Check the MD5 after any smoke run before trusting that directory. **A real Phase 5 run used to do it too** (the `base` stage exported `models/yolov8n-pose.pt` for latency, and `export_saved_model` starts with `shutil.rmtree` of `models/yolov8n-pose_saved_model/`); fixed 2026-10-08 — `combined._tflite_latency_from_pt` now exports a **copy** of the `.pt` inside a temp dir under `models/exports/`, so no Phase 5 stage writes next to its source checkpoint. The smoke-run trap above (other runners' `export_tflite`) still applies.

## Data Versioning

DVC is configured with a DagsHub S3-compatible remote (`https://dagshub.com/fernandeznicolas1801/Tesis-ModelosCV.dvc`). Run `dvc pull` to fetch datasets and model artifacts. Large files (models, data, venvs, MLflow artifacts) are gitignored and tracked via DVC; only the `.dvc` pointer files and `.gitkeep`s are committed. Credentials live in `.dvc/config.local` (gitignored) — never commit them.

What's DVC-versioned so far: a **UCO sample** (`data/raw/uco_sample.dvc` — subjects 0–1, actions 01–06, all 5 cams). The in-scope working set now lives locally in `data/raw/uco/` (27 subjects, exercises 01–09 + 13, labeled via `camN_p2d.txt`). `data/processed/` holds the YOLO-format conversion target (`uco_yolo/` flat pool of 74,107 image/label pairs + `folds/`, and the `pose_dataset_fold{0..4}.yaml`) **and** `uco_pseudo/`, the ViTPose-H auto-labeled parallel dataset — it has run: 74,107 images, 73,955 persons written, 69,880 with all 17 keypoints, 152 with no detection, `mean_kpt_score` 0.89, plus `pose_dataset_pseudo_fold{0..4}.yaml`. This is the dataset all five configs now train on.

`models/teachers/` (OpenMMLab configs + checkpoints, regenerable with `make teachers`) and all `*.pth` are gitignored.

**The v3 deploy lineage adds a third dataset dir, `data/processed/coco_pose/`** — a manually fetched COCO-pose subset (`images/val2017` + `labels/val2017`, 2,346 pose-labeled images ≈ 1 GB) used only to restore appearance generality in the post-prune fine-tune. It is **not** auto-downloadable through Ultralytics (that would pull the 19 GB train2017 into the global `datasets_dir`): fetch `coco2017labels-pose.zip` from the ultralytics/assets release and `val2017.zip` from `images.cocodataset.org`, extract *only* val2017 under that path, then run `python scripts/make_coco_pose_manifest.py --repeat 8`. Like every other dataset YAML, `data/processed/pose_dataset_pseudo_coco_fold0.yaml` lives under the gitignored `data/processed/` and is a DVC concern, not a git one.

## Notebooks & tests

- `notebooks/` — `01_eda.ipynb` (exploratory analysis), `02_benchmarking.ipynb`.
- **pytest is not installed and there is no pytest config.** `tests/test_pseudo_label_format.py` is written to work both ways: `test_*` functions for pytest, plus a `main()` runner so it works today with `python tests/test_pseudo_label_format.py` (needs `PYTHONPATH` at the repo root). It covers the YOLO-pose label invariants (56 tokens, coords in `[0,1]`, `v ∈ {0,1,2}`, `v=0 ⇒ coords zeroed`), the stem round-trip, the stratified sampler and the fold-divergence guard — all framework-free, so it runs in any venv.
- `tests/test_kfold_single.py` follows the same both-ways pattern and covers `_kfold._select_fold`: the requested fold is returned, the ≥10-fold lexicographic-order trap (`fold10` sorting before `fold2`), out-of-range → `PipelineError`, and the smoke-mode degradation. It touches only paths (no training, no MLflow run), so it runs in any venv.
- `tests/test_checkpoints.py` (same pattern; the `last.pt` cases need torch, skipped without it) covers the resumable-CV state: config fingerprint (only `workers` may change), fold/stage state surviving a reopen, the report being rewritten **only** between its markers, fresh / mid-training / finished / corrupt `last.pt` detection, and the fp32-optimizer mixin (with a premise test showing Ultralytics' fp16 conversion zeroes a 1e-9 `exp_avg_sq`).
- `tests/test_distillation_config.py` (framework-free, same pattern) covers the distillation stage's config rules: LR = `lr0/10` unless `distillation.lr0`, `warmup_bias_lr` default 0, logged params are the effective ones, `temperature` and typos raise.
- `tests/test_pruning_fixes.py` is the one test that needs **venv-torch** (torch + ultralytics; builds yolov8n-pose from YAML, no download). Same both-ways pattern; regression for the `C2f_v2` exactness and for `run_predictions` not fusing the caller's model, each with a premise test proving the defect would show.
- `check_label_line()` / `check_label_dir()` in that module are reusable to audit a generated dataset, not just to test. Running `check_label_dir` over the existing GT labels surfaces **17 keypoints with `y` marginally > 1.0** (max 1.013, all `right_wrist` in exercise 13 / cam1): the converter clamps only the bbox, not keypoints. Pre-existing and marginal, but worth knowing before treating that check as a hard gate on the GT dataset.
- Other modules are deliberately framework-free to *be* unit-testable standalone (`src/inference/normalizer.py`, most of `src/data/uco_to_yolo.py`, the serialization half of `pseudo_label_vitpose.py`).
- **Two checks worth running before any long run**, since fail-fast makes them cheap: `configs/smoke_yolov8n.yaml` through each runner (must complete), and any config with a bogus `data:` and no `smoke_test` (must raise `PipelineError`). The second is what proves the escapes are actually gone — before, it finished green. Note the bogus-`data:` check must also hold with `cross_validation: false`, since that branch runs after the same guard.
