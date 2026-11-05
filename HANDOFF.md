# Dataset 3 handoff — 2026-11-05

## Current state

Steps 1–5 are complete. Step 6 local inference/export is complete; **the user will manually upload the two defense CSVs and supply real Kaggle Public Scores**. The actual baseline Public Score is still **0.51194**. Do not substitute local validation F1 for a Kaggle score. Step 7 PPT waits for the user-supplied presentation skill, per AGENTS.md.

Current source implementation commit: 331ec5551dfdd33dc9c200a5ea01b38b31f08170. PR: https://github.com/Cobblepot2/COMP-5577-Dataset3/pull/4. Source and curated measurements are version-controlled; raw runs, optimizer checkpoints, transfer caches and submissions remain ignored. Compact inference checkpoints/CSVs are delivered in this chat's outputs bundle, with identities in docs/results/final_artifacts.json.

## Completed on the stronger device

- Original data: 9,469 labeled + 3,925 unlabeled, 64×64 RGB. All 13,400 original file sizes/SHA-256 match manifest; no supplied file changed. Fixed seed5577 split remains 7,575/1,894; train/validation IDs both refer to train_images, test IDs use a separate namespace.
- Environment: RTX5070Ti16GB sm120, Ultra9 285K24cores,64GBRAM; Python3.12.13 torch2.9.1+cu128 NumPy2.2.6 Pillow11.3.0. Official wheel SHA verified, real GPU backward and dependency checks pass. docs/results/environment_5070ti.json and device_import_verification.json record evidence; original RTX3060 environment report retained.
- Original8 tests passed before integration; full22 tests pass after integration. Synthetic GPU stop/resume versus uninterrupted training produced bitwise-identical parameters/scheduler. This is plumbing verification, not a score.
- New-machine baseline CSV exactly matches archived SHA acb82dd309a12d025c606b274e7f379a562898eb411235bf9b9937882a372c06; local clean F1 remains0.9994764075. Baseline Kaggle screenshot and actual0.51194 remain authoritative.
- Six actual training runs completed: clean/FGSM/PGD supplied-weight10epochs, randomPGD10epochs, suppliedPGD+augmentation40epochs, randomPGD+augmentation50epochs. Configs in configs/training*.json; curated actual summary docs/results/training_comparison.json. The short random trial was not a convergence result; long configurations change multiple parameters together.
- Seven models (baseline + six candidates) evaluated on all1,894 validation IDs, fixed947 attack mask, white-box and frozen-source transfer FGSM8/PGD10/PGD20x3, float and8-bit quantization. Bounds passed; no PGD10 float failure recovered under strongerPGD20x3 for any model. Full reports docs/results/defense_*.json and defense_comparison.json.
- Selected for worst configured white-box mixed F1: random50epoch model, bestepoch50, cleanF1.796880, whitePGD20x3F1.447401, mixedF1.630491. Main CSV submission_pgd_random50.csv. Supplied-weight augmented model bestepoch39/40 retained as tradeoff: clean.862639, transferPGD20x3.831076, transfermixed.848462; CSV submission_pgd_finetuned40.csv.
- Both compact inference artifacts match evaluated source weights bitwise, pass weights_only=True reload, and generate exactly3,925 template-ordered predictions. Two full inference passes per model have identical predictions/logits. Full identities, source hashes and CSV hashes in final_artifacts.json; inference_*.json records timing and label counts. Compact files omit optimizer/RNG and cannot resume training.

## Interpretation and remaining work

The supplied classifier saw all original training IDs; its descendants' local validation remains diagnostic. Random models excluded validation from gradients, but validation was used for checkpoint/hyperparameter selection. These are not independent final test estimates. A finite successful attack check is not a robustness certificate. FGSM training remained weak under PGD20x3; do not hide this failure.

Full measured table/methodology/failure classes: docs/results_5070ti.md. Commands, attack conventions and same-directory exact resume: docs/training_and_defense.md. Do not rebuild fixed splits merely for a new device, re-normalize inputs, introduce external data/pretrained weights, or use black-box attack/defense packages.

Next concrete task: receive the user's actual Public Scores for each CSV, update docs/results/kaggle_status.json and results documentation with evidence. Do not automatically choose the team's final Kaggle selection. PPT/report assembly then follows the user's presentation skill and team allocation; do not invent roster/contributions. Brief deadlines previously recorded: Kaggle Nov10 2026 23:59, Canvas Nov13 23:59, timezone to verify in those services.

Local runtime: ../ml-env/Scripts/python.exe relative to this repo. Full training runs in runs/, resume same output directory and original horizon using --resume runs/NAME/last.pt. Inference weights in checkpoints/ are local/ignored. User deliverables are in the surrounding outputs/ directory.
