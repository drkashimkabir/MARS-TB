# Reproducibility status

Date of assessment: 2026-11-12. Environment: Windows 11, Python 3.10.11, PyTorch 2.13.0 **CPU** build (RTX 3050
6 GB present but unused by PyTorch), no multi-GPU. Categories: Verified original research artifact · Verified
source-dataset artifact · Manuscript-derived implementation · Independently reproduced result · Test fixture ·
Unavailable or unverified.

## Bottom line
**The published results are not reproducible from what has been supplied.** The code can be verified and a new
experiment can be run; the manuscript's numbers cannot be regenerated because the artifacts that produced them
are unavailable, and an audit of the printed numbers shows internal inconsistencies that need the authors'
records to resolve.

## What the editor asked for, and where it stands (Comment 10)

| Request | Status |
|---|---|
| Code | Manuscript-derived reference implementation provided (this repository). **Original code: unavailable.** |
| Trained models | **Unavailable.** No original checkpoint located; none trained here (no data, no CUDA build). |
| Exact data-split files | **Unavailable.** Only newly generated, labelled-as-such splits can be produced. `--strata label` reproduces Table 1's *counts* but not the assignments. |
| Report preprocessing | Done: `docs/preprocessing.md`, `src/mars_tb/data/preprocessing.py` (with two documented deviations). |
| Report hyperparameters | Done: `configs/`, `docs/training.md` (global-batch and sampler-weight discrepancies flagged). |
| Report software versions | Partly: manuscript's recorded versions listed; the original full environment is unrecorded. Each new run records its own (`environment.json`). |
| Reporting checklist (CLAIM / TRIPOD+AI) | Both completed as **assessments of the manuscript as submitted**: 9 of 44 CLAIM items and 8 of 52 TRIPOD+AI sub-items are fully reported; the rest list what is missing and the corrective action. Authors must review. |

## Verified in this environment
* 265 unit/integration/CLI tests pass (+1 full-size test run with -m slow); 97 % line coverage; ruff and mypy clean.
* Synthetic end-to-end run (CPU): manifest -> grouped split -> training -> checksummed checkpoint -> schema-valid
  predictions -> metrics, thresholds from validation only, tamper detection.
* Full-size model: LoRA parameter count 589,824 (exact), default model 103,002,283 parameters
  (16,567,723 trainable); measured 231.6 GFLOPs (2 x MAC) = 115.8 GMACs on CPU.
* Numerical parity of the backbone with timm's ViT; DeLong against the O(n^2) definition; AUC/AP against
  scikit-learn; Clopper-Pearson against `scipy.stats.binomtest`.

## CI
GitHub Actions (`ci`, `security`, `documentation`) ran on Linux/CPU, Python 3.10-3.12, green on commit `96c08ef`.
Getting there exposed three real portability bugs (numpy 2 `trapz`, Python 3.10 typing, a missing dev dependency).

## Not verified / not exercised
CUDA and AMP training, multi-GPU (`torchrun`), the Dockerfile, loading of any
BioViL-T/CXR-BERT weights, real TBX11K/PadChest/TB Portals data, DICOM from real scanners.

## Why the published numbers cannot currently be regenerated
1. No prediction files, checkpoints, logs or split manifests (`artifact_inventory.csv`).
2. The results tables were computed on a 2,221-image test set that contradicts the revised 2,238-image Table 1.
3. BioViL-T as described does not exist as a ViT-B/16 checkpoint.
4. Seed-level tables, Supplementary S6 and several intervals show regularities that need the underlying logs
   (`manuscript_consistency_audit.md`, C07-C08, C20-C23).

## Reproducibility statement that is safe to make
"A manuscript-derived reference implementation with tests is available at <URL>. The original code, trained
weights, exact split assignments and prediction files are not available to the preparers of this repository; the
repository therefore does not reproduce the published results."
