# Training and defense workflow

Run these commands from the repository root with the project's CUDA-enabled Python environment active. They reuse the supplied `ProvidedNet` architecture, supplied data, seed 5577, and existing `splits/train.csv` / `splits/validation.csv`. Images enter the model in `[0,1]`; normalization remains inside `ProvidedNet`. No external data or pretrained weights are used.

## Comparable training runs

```powershell
python -m dataset3.training --config configs/experiment.json --training configs/training_clean.json --output runs/clean_provided
python -m dataset3.training --config configs/experiment.json --training configs/training_fgsm.json --output runs/fgsm_provided
python -m dataset3.training --config configs/experiment.json --training configs/training.json --output runs/pgd_provided
python -m dataset3.training --config configs/experiment.json --training configs/training_random_pgd.json --output runs/pgd_random
```

The first three configurations start from `provided_model.pt`; the fourth constructs a fresh, randomly initialized instance of the same architecture. All parameters are unfrozen for training.

The initial configurations use 10 epochs, batch size 128, SGD with learning rate 0.005, momentum 0.9, weight decay 0.0005, and cosine decay to zero. PGD training uses seven steps, step size `2/255`, one random start, and `epsilon=8/255`. FGSM uses a single step of `8/255` without a random start. Adversarial modes optimize `0.5 * clean_CE + 0.5 * adversarial_CE`; clean mode optimizes clean CE alone. Optional reflection-pad/crop/horizontal-flip augmentation is disabled initially and, when enabled, runs only on training images before generating attacks.

The random-initialization configuration is a preliminary comparison at this learning rate and horizon. Its result does not establish the best performance achievable from scratch; convergence and an appropriate training budget must be checked before drawing that conclusion.

## Checkpoints and resume

Each run writes `last.pt`, `best.pt`, and `metrics.json`. Checkpoints include model, optimizer, scheduler, completed epoch, Python/NumPy/Torch/CUDA random states, the shuffled-loader generator state, configuration, provenance hashes, history, and selection metadata. Load only trusted local training checkpoints because RNG restoration requires Python deserialization.

To interrupt after one completed epoch while retaining the configured ten-epoch cosine schedule:

```powershell
python -m dataset3.training --config configs/experiment.json --training configs/training.json --output runs/pgd_provided --stop-after-epoch 1
python -m dataset3.training --config configs/experiment.json --training configs/training.json --output runs/pgd_provided --resume runs/pgd_provided/last.pt
```

`--stop-after-epoch` is an absolute completed-epoch number. It does not shorten the scheduler horizon. Resume at an epoch boundary with the same output directory so an earlier `best.pt` stays available. Exact resume requires unchanged experiment/training configuration and provenance hashes, including the total epoch count. To investigate a longer horizon, choose it before starting a separate run rather than changing it halfway through an exact resume.

## Attack and selection conventions

Attack generation temporarily puts the model in evaluation mode, disabling dropout and batch-normalization updates. The outer clean/adversarial loss forwards and optimizer update use training mode. Input-gradient attacks do not accumulate parameter gradients.

Training PGD performs untargeted cross-entropy ascent and returns its **final iterate** after seven steps. It uses one optional random start and projects every step into both the `[0,1]` pixel range and the `Linf <= epsilon` ball. It does not retain earlier misclassifications or inject an FGSM candidate.

Evaluation PGD uses a different retention rule: for each image, prefer a misclassification, then larger CE within the same correctness group, across the clean input, every iterate, and every restart. This preserves failures discovered during the search. A finite attack result is an empirical stress test, not a robustness certificate.

Every training epoch evaluates the complete validation split with clean inference and PGD10 (`epsilon=8/255`, step size `2/255`, one restart). A fixed half of validation positions is attacked, selected with NumPy seed 5578. Mixed macro-F1 is recomputed from the resulting per-image mixture of clean and attacked predictions; it is not the average of two F1 values. `best.pt` maximizes this mixed macro-F1, uses clean macro-F1 as the tie-break, and retains the earliest exact tie. Epoch metrics record both losses and all three validation metric sets.

## Validation interpretation

The provided checkpoint may already have been pretrained on records in the fixed validation split. Results from that checkpoint and its fine-tuned descendants are local diagnostic results, not evidence of unseen-data performance.

A random-initialized candidate receives gradient updates only from training IDs. Its validation records are held out from fitting, but they still select the best checkpoint. This makes the comparison useful for separating the supplied-weight leakage issue; independent test data is still required for an unbiased final performance estimate. Test labels and Kaggle scores are not inferred from local validation.

## Defense evaluation

Use the quick suite for candidate comparison, then the full suite for selected candidates:

```powershell
python -m dataset3.evaluate_defense --config configs/experiment.json --checkpoint runs/pgd_provided/best.pt --suite quick --transfer-cache runs/transfer_cache --output runs/pgd_provided_quick
python -m dataset3.evaluate_defense --config configs/experiment.json --checkpoint runs/pgd_provided/best.pt --suite full --transfer-cache runs/transfer_cache --output runs/pgd_provided_full
```

The quick suite runs white-box FGSM8, white-box PGD10, and transfer PGD10. The full suite runs FGSM8, PGD10, and PGD20 with three restarts under both white-box and transfer threats. White-box attacks differentiate through the candidate; transfer attacks use the frozen supplied model as their source. Every attack uses `epsilon=8/255`, with `2/255` PGD steps. Transfer caches are tied to source weights, validation ordering/pixels, configuration, and attack/evaluator code; they can be reused across defenses.

The evaluator requires the fixed 1,894-record validation split. It writes clean, float-attack, and 8-bit-quantized prediction CSVs, the fixed mixed mask, and `metrics.json` with attack bounds, timing, memory, checkpoint identity, clean/attacked/mixed scores, and attack success rates conditional on initially correct predictions. Float and quantized results remain separate. The full suite also reports the PGD10 versus PGD20x3 strength comparison.

Omit `--checkpoint` to evaluate the original supplied baseline with the same evaluator. Replace the checkpoint/output paths to compare other trained candidates.

## Test inference

```powershell
python -m dataset3.predict_defense --config configs/experiment.json --checkpoint runs/pgd_provided/best.pt --output runs/final_inference
```

This writes `submission.csv` and `inference_metrics.json`. It requires exactly 3,925 unique IDs in the sample-submission order and one integer label in `0..9` per ID. It repeats the complete inference pass and requires identical predictions and bitwise-identical logits, then reloads and validates the saved CSV. The report records file/checkpoint hashes and timing. This command creates the artifact; it does not submit to Kaggle or produce a measured Kaggle score.

## Code verification

```powershell
python -m unittest discover -s tests -v
```

The implementation verification run passed 22 tests covering attacks, training updates and mode handling, epoch-boundary RNG/optimizer/scheduler/loader resume, actual mixed F1, transfer-cache reuse, and submission validation. Experiment measurements belong in the run reports and results documentation.

## Measured stronger-device continuation

Six training runs and seven full attack evaluations are complete; see [results](results_5070ti.md). Extended configurations are `configs/training_pgd_augmented.json` and `configs/training_random_convergence.json`. The latter is a longer convergence-oriented trial, not proof that optimization globally converged. Compact exported checkpoints preserve inference weights/metadata and omit resume state. Actual post-defense Kaggle scores remain pending manual upload.
