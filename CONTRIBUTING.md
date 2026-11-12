# Contributing

1. `pip install -e ".[dev]"` and `pre-commit install`.
2. Write the test first; keep functions small and typed. `make all` must pass (ruff, ruff format, mypy, pytest,
   artifact validation).
3. **No hand-typed results.** A number in a table or document must come from a script and a prediction file.
   Manuscript-reported numbers live only in `reporting/manuscript_reported_values.json`, labelled as such.
4. **No patient data, images, DICOM, credentials or model weights** in commits or issues.
5. Never describe a newly generated split as the original one, a synthetic fixture as an experiment, or an
   implementation choice as a recovered detail; record such choices in `docs/known_discrepancies.md`.
6. Update `docs/manuscript_to_code_mapping.md` when behaviour tied to the manuscript changes, and re-run
   `make audit`.
