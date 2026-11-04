# Task 3 handoff

## Current state (2026-11-04)

**Step 1 is complete: data preparation, fixed split, and GPU/model interface verification.** No training, baseline inference over the test set, attack evaluation, Kaggle submission, or presentation has been completed. Continue with **Step 2: baseline**.

The repository is self-contained for the supplied data: all 13,400 original files were imported in commit `4201ec6`. The original local source was `D:/downl/comp-5577-dataset-3`; future contributors should use the repository-relative `data/dataset3/` instead.

## Completed and verified

- Data import: 9,469 labeled clean training images; 3,925 unlabeled test images; all 64×64 RGB. Full image/CSV correspondence and copied file hashes were checked at import. Evidence: `docs/dataset3_validation.json`, `docs/dataset3_manifest.csv`.
- Fixed split: seed 5577, per-class 80/20, 7,575 training and 1,894 validation records. Both split files reference `data/dataset3/train_images/`. Class counts: `docs/split_report.json`.
- Split checks: exhaustive/disjoint IDs, preserved labels, deterministic byte-identical regeneration. File-hash audit found zero exact duplicates within training and zero train/test shared file hashes. It is not a perceptual near-duplicate search.
- Environment: Python 3.12.14, torch 2.9.1+cu128, NumPy 2.2.6, Pillow 11.3.0; RTX 3060 Laptop GPU with 6,144 MiB. `pip check` passed. Package versions are pinned in `requirements-lock.txt`.
- Supplied model: 1,175,786 parameters; weights load strictly. A real batch of 64 images passed GPU forward/backward with finite logits and finite nonzero input gradients. Peak PyTorch allocated memory was 372.52 MiB for this check; it is not PGD training peak memory or total process GPU memory. Evidence: `docs/environment_report.json`.
- No optimizer step was run and supplied weights were not changed.

## Reproduce Step 1

From the repository root, use Python 3.12 and the setup commands in README. Then:

```powershell
.venv/Scripts/python.exe scripts/prepare_dataset.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/check_environment.py
.venv/Scripts/python.exe -m pip check
```

The preparation script validates metadata at its entry and reuses the original import audit. It assumes the supplied dataset remains unchanged. Downstream training should directly consume the committed split CSVs, without re-running preparation in every job or batch. `check_environment.py` deliberately requires CUDA rather than silently switching to CPU. Hardware-specific memory values can differ on other machines. The fixed split is version-controlled and should not change merely because hardware changes.

## Next concrete task — Step 2: baseline

1. Add a small shared image dataset loader for the committed split CSVs and test.csv. Images become CHW float32 values in `[0,1]` by dividing by 255. **ProvidedNet already normalizes internally.** Do not normalize twice.
2. Load `ProvidedNet` from `data/dataset3/dataset3_model.py` and the original weights; set eval mode. Run local diagnostic inference on `splits/validation.csv`, then test inference in `test.csv` order.
3. Report Macro-F1 across labels 0–9, per-class results/confusion matrix and inference time/memory. Document warm-up and GPU synchronization for timing. The user-facing report must label provided-model local scores as diagnostic.
4. Produce ignored `submissions/submission_baseline.csv` with exactly `id,label`, 3,925 rows, IDs in sample_submission order, labels 0–9. Save run configuration and metrics in ignored `runs/`, and commit a concise measured-results summary plus the generating code.
5. Record the **actual** Kaggle Public score once obtained; never replace it with a local validation score. Competition invite: https://www.kaggle.com/t/af6658aab6d1415e890535aa591ec2e2 . Account login/team entry may require the user. Do not claim a submission happened without evidence.
6. Update this document and create a PR with completed items and verification, then merge when the stage is complete.

## Later stages and constraints

- Step 3: explicit FGSM and random-start projected PGD implementations; enforce `L_inf <= 8/255` in pixel space and clip pixels to `[0,1]`. The actual test attack is not fully disclosed; FGSM is an example in the brief.
- Step 4: compare clean fine-tuning, FGSM adversarial training and PGD adversarial training; consider attacks transferred from the provided classifier. New models must start from random weights and use only supplied training data.
- Step 5: evaluate clean, attacked and approximately 50/50 mixed validation conditions; distinguish transferred attacks from attacks against the defense itself. Run ablations and stronger-attack checks. Do not treat gradient masking as proof of robustness.
- Step 6: reproducible final inference and actual pre/post-defense Kaggle Public scores. Final model choice and tuning should not rely solely on repeated public leaderboard probing.
- Step 7: create Task 3 process slides using the presentation skill the user will supply. The project brief's 5–10 slides and 10-minute talk + 5-minute Q&A apply to the whole team project; Task 3 slides should fit the team's allocation.
- The provided model was trained on the entire supplied training dataset. Fine-tuning it after splitting cannot undo this exposure. Local comparisons for it and descendants are diagnostic. Models trained from scratch must exclude validation IDs during model selection.
- Metric: Macro-F1. Approx. half the test images are perturbed; labels remain the actual object class. No test labels are provided.
- Per project brief: Kaggle predictions due Nov 10, 2026 at 23:59; Canvas report/slides/code due Nov 13 at 23:59. Confirm deadline timezone in Kaggle/Canvas. The brief allows 10 submissions per day and two final selected submissions per competition.
- Team name, full team roster, contribution percentages and final report assembly remain team-level inputs. Do not invent them.

## Local setup note

The CUDA wheel download through the index's CDN stalled during setup; downloading the same official `download.pytorch.org` wheel with a resumed byte range completed and its ZIP CRC check passed. This was a one-time environment issue. The wheel and local `.venv/` are not project deliverables. GitHub connector calls also failed in the original session; normal Git HTTPS with the OpenSSL backend and the user's existing credential manager worked outside the sandbox. Do not put credentials into repository files or logs.
