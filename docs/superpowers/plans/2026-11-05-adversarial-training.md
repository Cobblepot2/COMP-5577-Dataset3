# Step 4–6 continuation plan — 2026-11-05

Spec: existing `docs/step4_plan.md`, `HANDOFF.md`, and the user's instruction to continue on this stronger computer.

Goal: implement reproducible adversarial training, compare actual measured defenses, stress-test the selected checkpoint, and produce a validated Kaggle CSV. Retain the actual baseline Public Score 0.51194. No presentation until the user supplies their intended skill.

Architecture: reuse ProvidedNet, fixed CSV splits, image cache, explicit attacks and ten-class metrics. Add one training module with resume and separate evaluation/inference CLIs. All runs/checkpoints stay ignored; archive small measured summaries only.

Constraints: supplied data/weights only; seed5577; original data immutable; 7575/1894 split; pixel [0,1], epsilon<=8/255; preserve normalization inside model. Supplied-weight descendants have diagnostic local validation. Random initialization sees only training IDs.

Tasks:
- [ ] Recreate local Python3.12/CUDA environment, load data and run original tests/GPU smoke check.
- [ ] Test and implement clean/FGSM/PGD training, train-mode handling, checkpointing and exact epoch resume.
- [ ] Test and implement defense evaluation with transferred/white-box attacks, quantized attacks, fixed 50% mixture and final test inference.
- [ ] Run a short sanity experiment, then comparable clean, FGSM and PGD supplied-weight trials and a random-init trial. Record configurations/timings/selection. Extend the promising candidate if measurements justify it.
- [ ] Run full strong evaluation on selected candidates; compare PGD10 and PGD20x3, inspect bound/gradient and clean tradeoffs.
- [ ] Generate 3925-row inference CSV, repeat inference determinism, record checkpoint/CSV hashes.
- [ ] Independent code review, full tests, update handoff/results/README and package user deliverables. Publish completed source through PR according to repository handoff authorization if credentials available.

Review focus: clean/attack model mode; frozen evaluation parameters restored for training; actual per-image maximum CE training convention distinct from evaluation; resume preserves all random streams and optimizer/scheduler; mixed F1 recalculated, not averaged; no claims of unseen validation for supplied-weight descendants; Kaggle score never fabricated.

Decision: do not repeat the unchanged Step3 sweep solely because hardware changed. Old sweep already establishes baseline vulnerability. Use stronger attacks against new defenses, which answers the unresolved question.
