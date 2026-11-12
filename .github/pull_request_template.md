## Summary

## Checks
- [ ] `ruff check .`, `ruff format --check .`, `python -m mypy` pass
- [ ] `python -m pytest -m "not slow"` passes; new behaviour has tests
- [ ] No numbers were typed in by hand: results come from a script and a prediction file
- [ ] No credentials, patient data, images or model weights are included
- [ ] If the manuscript-derived behaviour changed, `docs/manuscript_to_code_mapping.md` and
      `docs/known_discrepancies.md` are updated
- [ ] Generated documents are regenerated (`make audit`)
