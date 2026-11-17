---
name: gate-deploy-checkpoint
description: "Use when: a deployable model (.tflite or .pt) is ready and must be validated before shipping to the Android app, or the user asks to 'gate', 'validate the deploy checkpoint', 'check for keypoint collapse', or 'run the OOD gate'. In-domain OKS does NOT cover the deployment distribution — this gate caught a 0.931-OKS model that was broken in the app, and the five 0.964-OKS Phase 5 CV models fail it too."
---

# OOD acceptance gate for deployable checkpoints

Every deployable artifact must pass `scripts/diagnose_kpt_ratio.py --gt-set` **in addition to**
in-domain OKS. Rationale and the v2→v3 case study: [AGENTS.md](../../../AGENTS.md)
("RESOLVED — the deployed model's keypoints collapsed out of domain").

## Run it

```bash
# 1. The image set (once). ID lists are versioned in data/ood_gate/; the script needs the
#    official COCO annotations (annotations_trainval2017.zip, images.cocodataset.org).
python scripts/build_ood_gt_set.py --annotations-dir <dir with person_keypoints_*.json>

# 2. The gate (venv-torch, no MLflow needed)
python scripts/diagnose_kpt_ratio.py --model <path/to/model.tflite|.pt> \
    --gt-set data/processed/ood_gate_coco/train2017 [--save-vis <overlay_dir>] [--json-out f.json]
```

- **Reference set:** `data/processed/ood_gate_coco/train2017` (80 images, rebuilt 2026-10-16
  with explicit criteria; the original 79-image set of July 2026 was lost). Also run
  `.../val2017` (37 images, unseen by the stock and UCO-only models; **not** OOD for the v3
  lineage, whose fine-tune mixes COCO val2017).
- **Decode sanity check:** run the default matrix mode (no `--gt-set`) once; the stock-INT8
  control must reproduce its recorded values (x-ratio **0.564 on `bus.jpg`**, ~0.6–0.85 on UCO
  frames). A different value means the script/decode changed — fix that before judging.
- `--save-vis` overlays are the visual evidence; keep them for the thesis.
- To gate many models at once and get CSVs, adapt `scripts/run_ood_gate_fase5.py` (it calls the
  importable `gate_gt_set()`).

## Interpret the output

| Metric | Pass | Fail |
|---|---|---|
| keypoint error (frac. of GT diagonal) — **the decisive one** | ≤ ~0.05 | ≥ ~0.3 = skeleton off the body |
| person detected at conf ≥ 0.25 (the app threshold) | ~all images | a large fraction missing |
| collapsed images (span ratio < 0.6) | ~0–1 of 80 | dozens |
| span ratio pred/GT (mean) | ≥ ~0.9 | < 0.6 = collapse |

The gate's own decode picks the best detection **without** a confidence threshold, so a
model that "detects" at score 0.01 still counts as detected there; the conf ≥ 0.25 count
(`det025` in `run_ood_gate_fase5.py`) is what the user would actually see. The mean span
ratio alone is **not** enough: UCO-only models also produce over-wide skeletons, so a broken
model can average ~1.0 (Phase 5 deploy: 1.30 with error 0.43).

Calibrated reference points on the rebuilt train2017 set (80 images):

| Model | span ratio | collapsed | kpt error | detected @0.25 |
|---|---|---|---|---|
| stock COCO INT8 (ceiling) | 0.984 | 0/80 | 0.017 | 80/80 |
| v3 INT8 (shipped) | 0.945 | 0/80 | 0.039 | 78/80 |
| v2 INT8 (broken in app) | 0.759 | 33/73 | 0.393 | 47/80 |
| Phase 5 deploy INT8 (UCO-only) | 1.300 | 5/77 | 0.430 | 43/80 |
| Phase 5 CV INT8, 5 folds (UCO-only) | 0.82 ± 0.04 | 38 % | 0.42 ± 0.09 | 13–33/80 |

On the lost July set: v2 0.701 / 40/76 / 0.314, v3 0.913 / 1/79 / 0.049, stock 0.977 / 0/79 /
0.022 — same verdicts.

## Methodological rules (do not relax)

1. **Normalize spans by the GT bbox width, never the predicted box** — a broken model also
   predicts a bad box, which inflates the denominator and hides the collapse.
2. **`bus.jpg` is qualitative only** — even the stock control stays below 0.75 on it; never
   use it as a threshold.
3. A pass here plus in-domain OKS (target ≥ ~0.94 for the deploy lineage) is the ship
   criterion; either alone is insufficient. Every UCO-only training recipe measured so far
   (v2, Phase 5 deploy, the 5 CV folds) fails this gate; only the mixed UCO + COCO v3 passes.
