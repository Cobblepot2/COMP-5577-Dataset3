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

## Measured extension after the three controlled trials

The original objectives have now been measured: both final-epoch raw TRADES models fail the clean eligibility constraint; beta0.3 has an eligible early checkpoint at epoch3 but very poor transfer metrics, while hybrid bestepoch37 improves worst-source mixed F1 only from0.838496 to0.850292. Train-only BN calibration repairs some TRADES clean performance but not its attacked performance; hybrid calibration gains are negligible. Internal80/96pixel scaling lowers results and is rejected. Preserve these negatives.

Use the remaining budget for one random-initialized SmallResNet18(width32) experiment, trained only on7575IDs during selection. CIFAR-style3x3stem, four2-block residual stages32/64/128/256, internal given normalization, no external weights/data. Eightyepoch cosine LR.1,batch128,momentum.9,decay.0005; equalclean/source-adversarial CE in one joint BN forward. Source adversarial images use frozen provided FGSM8/PGD10 cached training inputs, equal per-image sampling, with shared crop/flip applied jointly to clean/adversarial pairs. This avoids per-step self-PGD overhead and tests architecture diversity; it is a transfer-focused recipe, not a claim of broad robustness.

Reuse the same clean eligibility and worst-source mixed selection. The attack teacher previously saw validation IDs, so the student validation is still diagnostic despite excluding validation inputs from its own updates. Every finalist receives the reserved random50 source and adaptive full-predictor stress evaluation. Freeze architecture/config/horizon/stopping epoch/inference settings before any new Public Score; at most two new CSVs. No tuning from a new Public Score within this batch.

In parallel, the earlier hybrid protocol is already frozen in runs/public85_frozen/hybrid_refit_protocol.json: restart from supplied weights on all9469records, original40epoch cosine horizon, stop37, identity+flip inference, no BN recalibration and no validation/selection during finalrefit. This is candidate1. Residual settings/selection procedure form candidate2 only if local evidence justifies it within the user's15:47CST budget boundary.

## Final diagnostic scope clarification

The conversation fixed the residual-only versus 50:50 residual/split-hybrid probability comparison with identity+flip before receiving candidate1’s Public result. There is no weight or temperature sweep based on Public feedback. If the fixed ensemble is selected locally, substitute only the already frozen full37 hybrid for deployment. The residual checkpoint follows the pre-existing eligible/worst-source selection rule over its80epochs.

Measured residual runtime is about18seconds/epoch. Within the same15:47CST compute boundary, retain full1894 FGSM/PGD10/PGD20x3; the extra adaptive CE50x5 and margin50x5 use a pre-prediction stratified200 sample (20/class,seed5577,original order). CE20x3 on that identical sample checks failure retention. This is explicitly sampled stress, not full1894 stress or certification. Reserved random50-source transfer uses full1894. Full-refit pipeline results are in-sample security checks, not further model selection.
