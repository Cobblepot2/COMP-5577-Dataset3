# COMP5577 — Dataset 3: Adversarial Robustness

This repository contains the supplied Dataset 3 files, fixed data preparation, baseline predictions and explicit FGSM/PGD attacks for the COMP5577 project. Adversarial training, strong defense evaluation and reproducible test inference are now implemented and measured.

**Continuing this project? Start with [HANDOFF.md](HANDOFF.md) and [AGENTS.md](AGENTS.md).** Steps 1–3 are complete. The user submitted the baseline; its actual Kaggle Public Score is **0.51194** ([evidence](docs/results/kaggle_baseline_public_score.png)). Steps 4–5 and local Step 6 inference are now complete on the stronger device; the user will manually upload defense CSVs for actual post-defense Kaggle scores. See [new-device results](docs/results_5070ti.md) and [training/inference commands](docs/training_and_defense.md).

## Supplied files

All original files are preserved under `data/dataset3/`:

- `train.csv`, `train_images/`: clean labeled training data.
- `test.csv`, `test_images/`: unlabeled test data, including adversarially perturbed images.
- `class_names.csv`: mapping of the ten labels to class names.
- `sample_submission.csv`: required prediction format (`id,label`).
- `dataset3_model.py`, `provided_model.pt`: the course-provided classifier and weights.

Images are 64×64 RGB. The provided model expects float inputs in [0, 1] and performs normalization internally; do not normalize a second time. Test perturbations are bounded by 8/255 per channel in the L-infinity norm.

## Integrity checks

`docs/dataset3_manifest.csv` records each original file's relative path, byte size and SHA-256 hash. `docs/dataset3_validation.json` records image counts, dimensions, class counts and validation results. The copied files were checked against their local originals before upload.

## Project constraints

Use only the supplied data. External datasets and pretrained weights are prohibited, except for this supplied Dataset 3 classifier. Implement the attack and defense algorithms explicitly. Evaluate Macro-F1 and record actual Kaggle public scores before and after defense. Local measured results are diagnostic and are reported separately from Kaggle scores.

Generated checkpoints, logs and predictions belong in `runs/`, `checkpoints/` and `submissions/`; these are excluded from source control by default.

Selected experiment summaries and figures are intentionally archived under `docs/results/`. The small, reproducible baseline submission is explicitly retained under `results/baseline/` so it can be downloaded on another device; raw run directories remain ignored.

## Steps 2–3: baseline and explicit attacks

```powershell
.venv/Scripts/python.exe -m dataset3.baseline
.venv/Scripts/python.exe -m dataset3.evaluate_attacks
```

The baseline command creates `submissions/submission_baseline.csv` and detailed local results in `runs/baseline/`. It does not submit to Kaggle. The [archived baseline CSV](results/baseline/submission_baseline.csv) contains all 3,925 test predictions in template order.

The attack command uses `configs/attacks.json` and writes metrics, predictions, the fixed mixed-condition mask and example figures to `runs/attacks/`. It evaluates all 1,894 diagnostic images with FGSM at 2/255, 4/255 and 8/255; PGD-10 at 4/255 and 8/255; and PGD-20 with three random starts at 8/255. PNG-style quantized adversarial inputs are measured separately. Attack functions are implemented directly in `dataset3/attacks.py` without a ready-made attack package.

See [measured results and methodology](docs/baseline_and_attacks.md) and the [Step 4 plan](docs/step4_plan.md). The clean diagnostic Macro-F1 is 0.999476; FGSM at 8/255 reduces it to 0.191187 and tested PGD at 8/255 to 0.000000. These are local attacks against the provided model, not Kaggle scores or a defense result.

## Step 1: environment and fixed split

Use Python 3.12 and run commands from the repository root. Windows PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/prepare_dataset.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/check_environment.py
```

The CUDA wheel follows the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v291). It installs the runtime required by PyTorch; this project does not require a separate CUDA toolkit build. The GPU check explicitly requires CUDA.

For the exact tested package set, install `requirements-lock.txt` after installing torch; it pins transitive packages as well. The environment is local and excluded from Git. Data preparation details and class counts are in [docs/preparation.md](docs/preparation.md).

`configs/experiment.json` fixes seed 5577 and a 20% per-class validation split. The committed `splits/train.csv` (7,575 rows) and `splits/validation.csv` (1,894 rows) preserve the original image IDs. Both refer to images inside **train_images/**. IDs in test.csv belong to a separate namespace, so an equal numeric ID across train/test does not mean an equal image.

The preparation entry point validates metadata once and reuses the completed image/hash import audit. It assumes the original dataset is unchanged. It does not re-open every image or repeat all file hashes during training. Training scripts should consume the committed split files directly. Split verification results are in `docs/split_report.json`.

The environment check loads the supplied weights, runs a batch of 64 real training images through forward/backward on the GPU, and confirms finite, nonzero input gradients. It performs **no optimizer step** and reports no accuracy score. Its measurements are in `docs/environment_report.json`.

### Evaluation limitation

The provided classifier has already been trained on the supplied training data. Therefore the held-out IDs are not unseen examples for that classifier or its fine-tuned descendants. Use this split for consistent diagnostics of those models, not an unbiased generalization claim. Newly initialized models must train only on `splits/train.csv` during model selection. Kaggle results must be reported separately.

### Coding conventions

- Validate external inputs at the entry point, then rely on the established contract internally.
- Avoid repeated null checks and blanket try/except blocks.
- Keep simple one-use logic inline; extract functions for actual reuse or clear responsibility boundaries.
- Ask before adding defensive machinery solely for low-probability edge cases.
