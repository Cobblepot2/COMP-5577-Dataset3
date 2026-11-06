# One controlled residual-network experiment

This experiment trains a random `SmallResNet18` using only the supplied records and supplied classifier as a frozen attack source. It uses no external weights or data. Its input normalization remains inside the network.

```powershell
python -m dataset3.cached_source_training --training configs/public85_residual_source.json --output runs/public85_residual_source --train-cache runs/public85_bnprobe --transfer-cache runs/improve_validation_cache
```

The predeclared configuration is80epochs, batch128, SGD learning rate.1/momentum.9/weight decay.0005, cosine decay, clean CE weight.5 and adversarial CE weight.5. Each shuffled batch independently chooses FGSM8 or source-PGD10 for each whole adversarial image with probability.5. Teacher attacks use epsilon8/255 and PGD step2/255. They are generated once from the frozen, eval-mode supplied source and cached on CPU; the cache identity matches the earlier BN-probe training attacks when the training CSV/pixels, teacher weights/architecture, attack code/settings, seed5577 and source batch size64 match exactly.

Each item contains clean, FGSM and PGD float images plus an integer label. Crop/flip augmentation is applied jointly to the selected clean/adversarial pair by concatenating their six channels; identical spatial transforms preserve their pixel perturbation bound. A single forward on the concatenated clean+adversarial batch gives batch normalization equal counts from both domains. There is no current-student attack during training.

Every epoch evaluates the full1,894-record validation set under clean inference and cached supplied-source FGSM8/PGD20x3. It first prefers clean macro-F1>=.857639, then maximizes the worse of the two fixed947/947 mixed macro-F1 values, then clean F1. Earlier exact ties remain selected. Ineligible checkpoints remain diagnostic and must not replace the prior eligible model. Own-model adaptive attacks are deferred to finalist evaluation; source robustness does not establish white-box robustness.

Student gradient updates use only the7,575 training IDs. However, the attack teacher originally saw all supplied training IDs, including validation. These local results are therefore diagnostic rather than independent unseen-data estimates. A local result does not establish a Kaggle Public Score.

`last.pt`/`best.pt` preserve the standard complete checkpoint state, with `training_config.architecture="small_resnet18"` for inference. Exact resume requires the same output directory, configurations, total80epoch cosine horizon, source/data/code hashes and protocol:

```powershell
python -m dataset3.cached_source_training --output runs/public85_residual_source --stop-after-epoch 1
python -m dataset3.cached_source_training --output runs/public85_residual_source --resume runs/public85_residual_source/last.pt
```

After model/configuration/horizon/stopping epoch are frozen, a separate `--final-refit` run reads all9,469 supplied training records directly, generates appropriately identified teacher caches for those IDs, and skips validation and checkpoint selection. It writes `last.pt` with a full-data/no-held-out-metric scope. This never rewrites committed splits. Choose its absolute `--stop-after-epoch` from the preceding selection protocol; do not select an epoch by evaluating now-exposed validation records.
