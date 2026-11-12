# Dataset card: cohorts used by the manuscript

No image data is included in this repository. Counts are **manuscript-reported** (Table 1) and not verified
against the data.

| Cohort | Images | Positive | Negative | Prevalence | Role |
|---|---:|---:|---:|---:|---|
| TBX11K training | 6,714 | 714 | 6,000 | 10.63 % | model fitting |
| TBX11K validation | 2,238 | 238 | 2,000 | 10.63 % | early stopping, threshold |
| TBX11K internal test | 2,238 | 238 | 2,000 | 10.63 % | internal evaluation (Section 3.1 uses 2,221; discrepancy B) |
| PadChest-TB | 3,000 | 891 | 2,109 | 29.70 % | external only |
| TB Portals cohort | 2,800 | 1,847 | 953 | 65.96 % | external only |
| Total | 16,990 | 3,928 | 13,062 | 23.12 % | |

## Label definitions
* **TBX11K positive**: active TB (924), latent TB (212), or both (54); **negative**: healthy (5,000) or sick
  non-TB (5,000); 10 uncertain images excluded. "Negative" does not mean radiographically normal.
* **PadChest-TB**: positive = tuberculosis-labelled report; negative = *normal-labelled* report. Not
  microbiologically confirmed; narrower negative spectrum than TBX11K.
* **TB Portals**: positive/negative as designated by the authors; source and standard of the negatives not
  established; baseline vs treatment vs follow-up images not distinguished.

## Missing information (acknowledged by the manuscript)
Age, sex, HIV status, prior TB, drug resistance, acquisition (projection, device, portable vs fixed), treatment
timing, dates of collection, unique-patient counts and cross-dataset overlap, annotation procedure and
inter-rater agreement for the auxiliary pattern/severity labels.

## Splits
The 60/20/20 split is image-level and stratified; its seed and assignment are not documented. `generate_splits.py`
creates a *newly generated* split; `--strata label` reproduces Table 1's counts but not its assignments.

## Licences and access
TBX11K: see the dataset's own terms (Kaggle copy and original repository). PadChest: BIMCV terms. TB Portals:
NIAID data-use agreement. Do not redistribute images from this repository.
