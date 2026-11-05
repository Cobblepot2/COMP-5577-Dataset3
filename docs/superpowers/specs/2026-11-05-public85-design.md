# Public-score improvement study

User objective: improve actual Kaggle Public Score from 0.80418 toward at least 0.85, with approximately 1–2 hours additional local compute. Actual new Public Scores require user manual upload; no local metric establishes success.

Existing constraints remain: supplied data/weights only; seed5577 and committed7575/1894 splits; normalization inside ProvidedNet; original files immutable; explicit algorithms; pixel epsilon<=8/255. Supplied-weight validation remains diagnostic. Preserve the old observed scores, models and source.

## Approaches and evidence

1. Existing-model probability ensembles, flip/shift TTA and mild blur were screened in a throwaway local probe. Error overlap is large; the best old-model flip gain is only about0.004 source-PGD mixed F1. Keep deterministic flip as an optional inference comparison, not the primary training strategy.
2. Implement TRADES clean CE + beta KL(p_clean||p_adv), beta0.3 and1.0, uniform random-start seven-step KL ascent with epsilon8/255 step2/255. Inner clean reference detached; outer both distributions retain parameter gradients. Same provided initialization,40epochs,batch128,LR.01/cosine,momentum.9,weightdecay.0005,training crop/flip.
3. Implement hybrid PGD training: .75clean CE + .25 adversarial CE, shuffled batches split approximately equally between current-model and frozen-provided-source PGD7. Generate perturbations after augmentation. Keep random50 absent from attack-source training for additional transfer checks.

Before any new training measurements, beta0.3/1.0 replaces the initial beta1/3 proposal to include a more clean-preserving arm; KL and adversarial CE gradients are not assumed equivalent. New architecture work is deferred until these controlled objectives have been measured. This is a bounded study, not a claim that the target is achievable.

## Selection and evaluation

Every epoch: full fixed validation, own PGD10, and cached original-source FGSM8/PGD20x3. Recompute947/947 mixed predictions. Prefer candidates meeting cleanF1>=0.857639 (old fine40 minus.005), then maximize the worse of the two transferred mixed F1s, with cleanF1 tiebreak. Preserve exact criterion and all failed experiments. Do not use Kaggle scores to sweep weights/settings.

Finalists: original full suite, float/8bit, plus PGD50x5 CE and a separately implemented logit-margin objective against the complete deployed predictor. Identity+flip probability averaging must remain differentiable for adaptive evaluation; no stochastic preprocessing/EOT needed. Evaluate transfer from the unused random50 source for every finalist. A below-clean-floor checkpoint remains diagnostic; if no new candidate qualifies, retain old fine40 rather than silently recommending an ineligible model. Freeze at most two new CSV candidates before the next user submissions.

After all model/config/TTA choices are frozen, optionally refit the winner on all9469 supplied labeled records for the selected epoch E using the original40epoch cosine horizon. Never alter committed splits, inspect test labels, or select a new epoch from the now-exposed validation data. Final refit skips validation/cache/selection, records selection-source hashes and full-data exposure, and exports last.pt as a fixed-epoch final fit with no held-out metric. This increases usable adaptation data after model selection without claiming a new local generalization score.

## Implementation boundaries

`dataset3/improve_training.py`: objectives, full-data CLI, source validation cache, criterion, complete checkpoint/resume using existing state helpers. Separate from the historical training entrypoint. New configs and meaningful tests.

`dataset3/tta.py`: deterministic probability averaging wrapper. Extend evaluation/inference CLIs with optional flip/stress flags; defaults preserve previous behavior. Explicit margin PGD and analytic/bound/gradient tests.

Curated commands, metrics, code/data hashes and actual timings remain reviewable; raw runs/checkpoints/caches ignored. Record Public results only from user evidence. Existing plan and this runtime allowance authorize the bounded implementation/experiments; no account or final-submission selection changes are part of this work.
