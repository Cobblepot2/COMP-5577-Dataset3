# Curated project outputs

`baseline/submission_baseline.csv` is the intentionally versioned baseline prediction artifact. It contains 3,925 rows (plus header), with `id,label` in the sample submission's order. It has not yet been submitted to Kaggle.

Reproduce it from the repository root:

```powershell
.venv/Scripts/python.exe -m dataset3.baseline
```

The command writes `submissions/submission_baseline.csv`. For the tested environment and original checkpoint, its SHA-256 is `acb82dd309a12d025c606b274e7f379a562898eb411235bf9b9937882a372c06`. Two runs reproduced identical predictions and logits. This byte-level claim applies to the tested environment; record differences if moving to another software/hardware stack.

Full raw run directories, logits and intermediate prediction files remain ignored. Selected metric snapshots and figures are archived in `docs/results/`. After an actual Kaggle submission, update `docs/results/kaggle_status.json` with the observed public score and submission information. Never infer that score from local validation metrics.
