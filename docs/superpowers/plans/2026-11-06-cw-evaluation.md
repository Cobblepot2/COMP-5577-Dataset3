# C W evaluation implementation plan

Goal: explicit tested C&W-L2 and optional additional Linf-box evaluation of frozen models, with measured Chinese report update and no model training.

Spec: ../specs/2026-11-06-cw-evaluation.md

- [x] Write analytical behavioral tests and observe failure, implement tanh/Adam/constant-search attack.
- [x] Review attack source; run CPU suite and bounded GPU smoke/runtime estimate.
- [x] Evaluate standard and budget-constrained C&W on the fixed200 for three frozen pipelines; verify bounds, image IDs, and model/CSV identity.
- [x] Independently review measurements, document limited search and in-sample scope, compare same200 PGD.
- [x] Update/visually verify all seven Chinese chapter pages, archive evidence, and prepare HANDOFF/PR integration and deliverables.

Evidence: docs/cw_evaluation.md, docs/results/cw/summary.json, gpu_replay.json and document_qa.json. Integration is recorded in Git/PR history. The English final awaits user review of the Chinese chapter.
