# Public85 improvement implementation plan

Goal: produce at most two independently verified new candidate CSVs after a bounded1–2hour study; target actual PublicScore>=.85 remains subject to user submissions.

Spec: ../specs/2026-11-05-public85-design.md

- [x] Inspect existing data/results, quantify error overlap and run bounded old-model ensemble/TTA probe. Preserve diagnostics separately from actual Public Scores.
- [ ] Implement/test new TRADES and source/self hybrid objectives in improve_training.py, validation cache/criterion and exact resume; three configs use the same fixed data and initial checkpoint.
- [ ] Implement/test differentiable identity+flip averaging and explicit margin PGD; preserve old CLI defaults, add optional adaptive stress evaluation and provenance.
- [ ] Run unit suite and short real-GPU sanity, then three40epoch jobs sequentially. Each logs clean, ownPGD and both source conditions. Stop/extend only with measured justification and remaining runtime budget.
- [ ] Screen candidates without Kaggle labels; full/stress evaluation on frozen finalists and unused-source transfer check. Export complete reproducible inference settings, repeat3925-row inference, verify exact IDs/order/hash/weights.
- [ ] Independent source/result review, update HANDOFF/report/curated records, commit/push via PR under existing authorization. Deliver new CSVs without overwriting the .80418 baseline artifacts; receive actual new Public Scores from manual user upload.

Review focus: KL direction/gradient branches; attack train/eval and source immutability; full criterion and cached-source identities; checkpoint RNG/optimizer/scheduler/loader resume; adaptive gradients through complete averaging wrapper; no false .85 or certification claims.
