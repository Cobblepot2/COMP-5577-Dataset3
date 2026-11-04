# Task 3 handoff

## Current state (2026-11-04)

**Steps 1–3 are complete, including the actual Kaggle baseline score: 0.51194.** The user manually submitted the baseline and supplied a screenshot showing successful completion. Continue with **Step 4: adversarial training on the user's stronger device**. No training or presentation has been started.

The repository is self-contained for the supplied data: all 13,400 original files were imported in commit `4201ec6`. The original local source was `D:/downl/comp-5577-dataset-3`; future contributors should use the repository-relative `data/dataset3/` instead.

## Completed and verified

- Data import: 9,469 labeled clean training images; 3,925 unlabeled test images; all 64×64 RGB. Full image/CSV correspondence and copied file hashes were checked at import. Evidence: `docs/dataset3_validation.json`, `docs/dataset3_manifest.csv`.
- Fixed split: seed 5577, per-class 80/20, 7,575 training and 1,894 validation records. Both split files reference `data/dataset3/train_images/`. Class counts: `docs/split_report.json`.
- Split checks: exhaustive/disjoint IDs, preserved labels, deterministic byte-identical regeneration. File-hash audit found zero exact duplicates within training and zero train/test shared file hashes. It is not a perceptual near-duplicate search.
- Environment: Python 3.12.14, torch 2.9.1+cu128, NumPy 2.2.6, Pillow 11.3.0; RTX 3060 Laptop GPU with 6,144 MiB. `pip check` passed. Package versions are pinned in `requirements-lock.txt`.
- Supplied model: 1,175,786 parameters; weights load strictly. A real batch of 64 images passed GPU forward/backward with finite logits and finite nonzero input gradients. Peak PyTorch allocated memory was 372.52 MiB for this check; it is not PGD training peak memory or total process GPU memory. Evidence: `docs/environment_report.json`.
- No optimizer step was run and supplied weights were not changed.
- Baseline: 1,894 local diagnostic images, 1,893 correct, Macro-F1 0.9994764075. Generated and archived all 3,925 test predictions in `results/baseline/submission_baseline.csv`. IDs/order/labels verified against the sample; SHA-256 `acb82dd309a12d025c606b274e7f379a562898eb411235bf9b9937882a372c06`.
- Explicit attacks: FGSM budgets 2/255, 4/255, 8/255; random-start PGD-10 at 4/255 and 8/255; PGD-20 with 3 restarts at 8/255. All run on every diagnostic validation record. Float and 8-bit quantized perturbations remained within the configured budget (1e-6 numerical tolerance).
- FGSM at 8/255: diagnostic Macro-F1 0.1911868614. PGD-10 and PGD-20 at 8/255: 0.0, with 1,893/1,893 initially correct images successfully attacked. This is evidence of vulnerability, not an achieved defense. Quantization did not restore performance.
- Metrics include accuracy, Macro-F1 over all 10 labels, per-class precision/recall/F1, confusion matrices, attack success among initially correct records, a fixed 947-attacked/947-clean mixture, timing and GPU memory. Details: `docs/baseline_and_attacks.md`, machine-readable snapshots and figures: `docs/results/`.
- Reproducibility: two baseline runs produced identical CSV predictions and logits; two full attack sweeps produced identical float and quantized prediction CSVs. The unit suite has 8 passing tests covering input scaling, hand-calculated metrics, the Week 5 analytic FGSM example, PGD constraints/randomness and split properties.
- The strongest attack sweep used approximately 678 MiB peak PyTorch allocated memory on this machine. This is inference/attack evaluation, not adversarial training memory. Actual times are archived, with warm-up and I/O scope stated.

## Reproduce Step 1

From the repository root, use Python 3.12 and the setup commands in README. Then:

```powershell
.venv/Scripts/python.exe scripts/prepare_dataset.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/check_environment.py
.venv/Scripts/python.exe -m pip check
```

The preparation script validates metadata at its entry and reuses the original import audit. It assumes the supplied dataset remains unchanged. Downstream training should directly consume the committed split CSVs, without re-running preparation in every job or batch. `check_environment.py` deliberately requires CUDA rather than silently switching to CPU. Hardware-specific memory values can differ on other machines. The fixed split is version-controlled and should not change merely because hardware changes.

## Reproduce Steps 2–3

```powershell
.venv/Scripts/python.exe -m dataset3.baseline
.venv/Scripts/python.exe -m dataset3.evaluate_attacks
```

Both commands accept `--config` and `--output`; the baseline also accepts `--submission`, and the attack evaluator `--attacks`. Current measured settings are version-controlled. Baseline output defaults to `runs/baseline/`, attacks to `runs/attacks/`. The archived attack snapshot was taken from the verified second run (`runs/attacks_verified/`) with identical predictions. Repeating every old experiment is unnecessary just to begin Step 4; use the saved evidence and smoke-test the new device.

## Completed external item — actual Kaggle baseline score

The user manually uploaded `submission_baseline.csv`; the screenshot received on 2026-11-04 (Asia/Shanghai) shows **completed, Public Score 0.51194**, with description indicating the supplied classifier, original weights, and no defense. Current authoritative record: `docs/results/kaggle_status.json`; visual evidence: `docs/results/kaggle_baseline_public_score.png`. Numeric submission ID and exact submission timestamp are not visible and remain null. No API credentials are required to continue the local training work. Do not resubmit merely to obtain the already recorded score.

Use **0.51194** as the actual pre-defense Kaggle Public baseline for the report; keep it separate from local diagnostic Macro-F1 0.999476. Local run JSONs preserve the original pre-submission snapshots. The screenshot shows 0/2 final selections and the baseline unchecked; final submission selection remains a later team decision. No leaderboard selection was changed by the agent.

## Next concrete task — Step 4 on stronger hardware

Read `docs/step4_plan.md`. Implement a training entry point with checkpoint/resume and recorded configuration; run clean fine-tuning, FGSM adversarial training and PGD adversarial training comparisons. Start with a short sanity run before full experiments. The user deferred training to the stronger device, so no training command/checkpoint exists yet.

Important code reuse details: `common.load_provided_model()` freezes parameters for evaluation; training must restore parameter gradients. Current PGD candidate selection prioritizes misclassification for evaluation; choose/document a maximum-CE or final-iterate convention for training. Model mode must be handled explicitly around attack generation and outer training. Preserve original weights, fixed splits, and the user's coding rules. New random-initialized models provide a genuinely held-out local comparison; supplied-weight descendants do not.

## Later stages and constraints

- Step 3 (complete): explicit FGSM and random-start projected PGD implementations; enforce `L_inf <= 8/255` in pixel space and clip pixels to `[0,1]`. The actual test attack is not fully disclosed; FGSM is an example in the brief.
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
