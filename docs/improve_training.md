# Controlled objective comparison

These runs use the existing 7,575/1,894 split and supplied initialization. The local supplied-weight validation remains diagnostic. The Public Score target is not established by these local metrics.

```powershell
python -m dataset3.improve_training --training configs/improve_trades_beta03.json --output runs/improve_trades_beta03 --transfer-cache runs/improve_validation_cache
python -m dataset3.improve_training --training configs/improve_trades_beta1.json --output runs/improve_trades_beta1 --transfer-cache runs/improve_validation_cache
python -m dataset3.improve_training --training configs/improve_hybrid.json --output runs/improve_hybrid --transfer-cache runs/improve_validation_cache
```

All three start with 40 epochs, learning rate .01 with cosine decay, batch size128, SGD momentum .9/weight decay .0005, and crop/flip augmentation. These TRADES beta values (.3 and1) were chosen before the runs. Historical training configurations and algorithms are unchanged.

TRADES uses clean CE + beta KL(p_clean || p_adversarial), summed over classes and averaged over records. The eval-mode inner reference is detached. Uniform-start KL-ascent PGD uses seven steps of2/255 at epsilon8/255 and returns the final iterate. Both outer distributions retain parameter gradients, and both outer forwards use training mode.

Hybrid uses .75 clean CE + .25 adversarial CE. On each shuffled, augmented batch, floor-half is attacked against the current model and the remainder against the frozen supplied source. Both attacks use random-start CE-ascent PGD7. The source stays in evaluation mode with frozen parameters; input gradients remain available.

Each epoch evaluates all validation records using own-model PGD10 and cached original-source FGSM8/PGD20x3. Source cache identities include the source weights/architecture, split/pixels, attack/cache code, seed, batch size and settings. Caches load tensor-only payloads with identity/bound checks at the CLI boundary. Dedicated cache loader/attack RNGs avoid altering the training random stream.

The checkpoint criterion first prefers clean macro-F1>=.857639, then maximizes the worse of the two source-attack mixed macro-F1s, then clean F1. All mixed scores are recomputed from fixed seed5578 half-clean/half-attacked predictions. Exact ties keep the earlier epoch. An ineligible `best.pt` is retained for diagnosis; it must not replace the prior eligible model automatically. Own-model PGD10 scores remain visible in every epoch report.

`last.pt` and `best.pt` use the complete existing RNG/optimizer/scheduler/loader checkpoint format. Resume in the same output directory using the same configuration, horizon, source/data/code hashes and protocol:

```powershell
python -m dataset3.improve_training --training configs/improve_trades_beta1.json --output runs/improve_trades_beta1 --stop-after-epoch 1
python -m dataset3.improve_training --training configs/improve_trades_beta1.json --output runs/improve_trades_beta1 --resume runs/improve_trades_beta1/last.pt
```

The stop epoch is absolute and does not change the cosine horizon.

## Full-data final refit

Only after the configuration, horizon and stopping epoch have been frozen from selection runs, a separate run may use all9,469 supplied training records:

```powershell
python -m dataset3.improve_training --training configs/improve_trades_beta1.json --output runs/final_refit_trades_beta1 --final-refit --stop-after-epoch 39
```

The stopping epoch above is an example, not a selected result. `--final-refit` reads `data/dataset3/train.csv` directly and never rewrites split files. It skips validation, transfer caches and checkpoint selection; it writes `last.pt` and training-only metrics with a full-data/no-held-out-metric scope. Its checkpoint records the refit flag and full-data CSV hash, and resume requires the same flag. The frozen source is still used for hybrid attacks. These runs do not supply a held-out estimate or a Kaggle score.
