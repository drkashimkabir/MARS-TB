# Response to Editor Comment 10 — DRAFT for the authors

> **Do not send as is.** The repository has been built and tested locally but **has not been published**: nothing
> below may be described as "available" until the authors have pushed it, opened the URL logged out, and
> replaced every `[CONFIRM ...]` marker. The factual statements about what does *not* exist are current.

**Editor comment.** "Please make the code, trained models and the exact data-split files available in a public
repository (or explain why this is not possible), report preprocessing, hyperparameters and software versions,
and complete a reporting checklist suitable for AI studies in medical imaging (for example CLAIM or TRIPOD+AI)."

**Response (suggested text).**

We thank the editor for this request. We have taken the following steps and are transparent about what we are
unable to provide.

**1. Code.** We have prepared a public repository, [CONFIRM URL after publication: https://github.com/drkashimkabir/MARS-TB],
containing a documented, tested re-implementation of MARS-TB written from the manuscript (model, preprocessing,
split generation and verification, leakage audit, training, evaluation, statistics, explainability and
configuration files). **It is a reference implementation, not our original research code**, and we have not shown
that it is numerically equivalent to the code that produced the manuscript's results. [AUTHORS: if the original
code exists, state its location here or why it cannot be released.]

**2. Trained models.** [AUTHORS TO COMPLETE.] No trained checkpoint is included in the repository. The original
MARS-TB weights [CONFIRM: were not retained / cannot be shared because ...]. The repository provides the training
procedure and a checksummed checkpoint format so that a model can be retrained and verified.

**3. Exact data-split files.** The exact historical assignment of TBX11K images to the training, validation and
internal-test partitions is not available in the form of a file [CONFIRM]. The repository provides a seeded,
verifiable split generator; splits it produces are labelled "newly generated" and must not be taken as the
original partition. We have also corrected the dataset description: the 60/20/20 split is image-level and
patient-level separation cannot be verified because TBX11K has no patient identifiers.

**4. Preprocessing, hyperparameters, software.** Preprocessing is documented in `docs/preprocessing.md` and
implemented in `src/mars_tb/data/preprocessing.py`; the training configuration is in `configs/` and
`docs/training.md`. While preparing the repository we identified inconsistencies in the manuscript (for example
the test-set size reported in Table 1 versus Tables 4-5, the class-imbalance weight, the effective batch size, the
description of BioViL-T, and several supplementary-table details). They are listed in `docs/known_discrepancies.md`
and we will correct the manuscript [CONFIRM which corrections the authors will make]. The software versions
recorded in the original work were PyTorch 2.1.0, CUDA 12.1, OpenCV 4.8 and torchvision 0.16; the repository
documents a modern environment and records versions for every new run.

**5. Reporting checklists.** CLAIM 2024 and TRIPOD+AI checklists are provided in `reporting/`. Items that the
manuscript does not currently report are marked as such with a corrective action; we will revise the manuscript
accordingly [CONFIRM].

**6. What cannot be shared.** Original code, trained weights, exact split assignments and model-generated
prediction files [CONFIRM each: unavailable / withheld because ...]. The image datasets are not redistributed;
TBX11K, PadChest and TB Portals are available from their providers under their own terms (TB Portals requires a
data-use agreement).

---

### Preparer's notes (delete before sending)
* Everything marked `[CONFIRM]` or `[AUTHORS ...]` is a fact only the authors can supply.
* Do not cite Table 8, Supplementary Tables S3-S4 and S6, or the Table 4 intervals as verified results unless the
  underlying run logs can be produced (`reporting/manuscript_consistency_audit.md`, findings C07, C08, C20-C23).
* Files that will exist once pushed (all present locally): `README.md`, `src/mars_tb/`, `scripts/`, `configs/`,
  `tests/`, `docs/`, `reporting/CLAIM_2024_checklist.csv`, `reporting/TRIPOD_AI_2024_checklist.csv`,
  `reporting/artifact_inventory.csv`, `reporting/manuscript_consistency_audit.md`.
* Files that do **not** exist and are not promised: checkpoints, split-assignment files, prediction files,
  original code, a licence (see `docs/licensing_status.md`).
