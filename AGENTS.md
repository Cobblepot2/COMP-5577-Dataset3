# Instructions for future contributors and agents

Read `HANDOFF.md`, `README.md`, and `docs/preparation.md` before continuing.

- Work on COMP5577 Task 3 / Dataset 3 only. Preserve the supplied files in `data/dataset3/`.
- Use only supplied data. No external datasets or pretrained weights except `provided_model.pt`.
- Implement attack/defense algorithm details explicitly, not as ready-made black-box defenses.
- Validate external inputs once at the entry point; trust the established contract internally. Do not add repeated null checks or blanket try/except blocks.
- Keep simple one-use logic inline. Ask the user before adding defensive machinery solely for low-probability edge cases.
- Keep seed 5577 and the committed splits fixed for comparable experiments. The provided model has already seen the source training data: its local validation scores are diagnostic, not unseen-data estimates.
- Do not invent metrics, Kaggle scores, training results, or team information. Record commands and actual measurements.
- The user will provide a presentation skill later. Do not author the PPT before that step.
- For each completed stage, update `HANDOFF.md` with completed work, validation, limitations and the next concrete task. Describe these in a PR and merge completed work to `main` as requested by the user. Never force-push over others' changes.
- Do not commit environment directories, credentials, temporary downloads, or generated training artifacts by accident. Use the existing ignored directories for runs/checkpoints/submissions.
