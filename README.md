# COMP5577 — Dataset 3: Adversarial Robustness

This repository contains the supplied Dataset 3 files for the COMP5577 project. Defense implementation and experiment results will be added separately.

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

Use only the supplied data. External datasets and pretrained weights are prohibited, except for this supplied Dataset 3 classifier. Implement the attack and defense algorithms explicitly. Evaluate Macro-F1 and record actual Kaggle public scores before and after defense. No performance results are claimed yet.

Generated checkpoints, logs and predictions belong in `runs/`, `checkpoints/` and `submissions/`; these are excluded from source control by default.
