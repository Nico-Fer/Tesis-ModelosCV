---
name: preflight-long-run
description: "Use when: about to launch a multi-hour training run (any phase runner, CV or deploy config), or the user asks to 'run phase N', 'train', 'launch the deploy run', or 'preflight'. Validates the environment and pipeline cheaply before committing hours of GPU time."
---

# Preflight checklist for long training runs

Training costs ~4.1 min/epoch on the 3090 and a single INT8 eval pass costs ~53 min, so a
run that dies (or silently degrades) hours in is expensive. Run these checks first — fail-fast
makes them cheap. Full background: [AGENTS.md](../../../AGENTS.md) ("Running Experiments",
"Two TFLite export traps").

## 1. Environment

- **venv**: YOLO phases (1, 4, 5) and all TFLite export run in `venv-torch`. MoveNet/BlazePose
  pruning runs in `venv-tf`. Never mix.
- **CUDA**: `python -c "import torch; print(torch.cuda.is_available())"` must print `True`.
  Ultralytics tends to reinstall the CPU torch build; if `False`, run `make setup-torch-cuda`
  (constraints file is mandatory — see AGENTS.md).
- **MLflow server up** and `MLFLOW_TRACKING_URI` pointing at it (`http://127.0.0.1:5000`).
  A bare `file://./mlruns` run can read existing experiments but `log_artifact` will fail
  mid-run because artifact locations use the `mlflow-artifacts:` scheme.

## 2. Pipeline sanity (both must behave as stated)

```bash
# a) Smoke test through the target runner — MUST complete green
python -m src.training.<runner> --config configs/smoke_yolov8n.yaml

# b) Bogus data:, no smoke_test — MUST raise PipelineError (proves fail-fast escapes are gone)
#    Repeat with cross_validation: false in the config; it must still abort.
```

## 3. Restore the stock control after any smoke run

A smoke run with `export.format: tflite` **overwrites** `models/yolov8n-pose_saved_model/`,
clobbering the stock INT8 control used by the app comparison and `scripts/diagnose_kpt_ratio.py`.
Verify before trusting it:

```powershell
Get-FileHash models/yolov8n-pose_saved_model/yolov8n-pose_int8.tflite -Algorithm MD5
# Must be EE27DCF9F0AD0C25FCF37868E097AD68
```

If it differs, restore from `mlruns/545531124405737930/035c95c6c0f84077b29caa32ef48840a/artifacts/model/`
(INT8) and `models/exports/yolov8n-pose_saved_model/` (floats).

## 4. Config review before launch

- Deploy artifact → `*_deploy*.yaml` (`cross_validation: false`, separate `fase*-deploy`
  experiment). Thesis CV numbers → the plain config. Never mix the two experiments.
- If the config sets `lr0`, it must also set `optimizer:` explicitly — with Ultralytics'
  default `optimizer: auto`, `lr0` is silently ignored.
- `augmentation:` keys are validated (typo → `PipelineError`), but confirm they are the
  intended ones; defaults train silently otherwise.
- Do NOT launch Phase 4 or Phase 5's distillation stage for real results — the distillation
  loss is known-broken (see AGENTS.md "BROKEN — the distillation stage diverges").

## 5. Size the run

~233 img/s training at batch 24 (57.8k-image train split ⇒ ~4.1 min/epoch); `.pt` eval
12.8 ms/img on GPU; INT8 `.tflite` eval 196 ms/img on CPU (~53 min over the 16.1k val split).
State the expected wall-clock to the user before launching.
