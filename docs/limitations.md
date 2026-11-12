# Limitations

## Of this repository
1. It is a **manuscript-derived reference implementation**. It is not the authors' original code and it is not
   known to be equivalent to it. It does not reproduce, and must not be cited as reproducing, any number in the
   manuscript.
2. **No trained weights, no exact historical split, no prediction files** are provided (see
   `reporting/artifact_inventory.csv`).
3. The backbone is a randomly initialised (or timm-initialised) ViT-B/16, **not** BioViL-T (discrepancy I).
4. Zone boundaries, graph edges, frequency-band geometry, ablation wiring, text-prompt encoding and the NLL
   variant are implementation choices (`known_discrepancies.md`).
5. **Not exercised:** multi-GPU training (no multi-GPU hardware), CUDA/AMP paths (the Python environment used
   had a CPU-only PyTorch build although an RTX 3050 6 GB was present), and the Docker image. The code paths exist; they are untested here.
   GitHub Actions (Linux, CPU, Python 3.10-3.12) ran green on commit `96c08ef`: lint, format, mypy, 264+ tests,
   coverage gate, the full-size test, gitleaks, pip-audit, bandit.
6. Near-duplicate detection uses 64-bit perceptual hashes. Chest radiographs are structurally similar, so a
   single-hash match is only a *candidate*; only two-hash agreement or an exact match fails an audit. A clean
   audit does not prove patient-level separation.
7. The synthetic fixture makes plumbing testable; it cannot test clinical validity.

## Of the study the manuscript describes (as the manuscript itself states)
Retrospective, image-level, custom split; patient-level separation unverified; label provenance incomplete;
external cohorts without a microbiological reference standard; no demographics, treatment stage or acquisition
data; binary endpoint mixes active, latent and combined TB; auxiliary pattern/severity outputs unvalidated; no
prospective, reader or workflow study. **Not a medical device; not for clinical use.**
