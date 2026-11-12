# Statistics

| Quantity | Method | Code |
|---|---|---|
| AUC | Mann-Whitney (average ranks); checked against scikit-learn incl. ties | `evaluation/metrics.py` |
| AUC CI, paired AUC difference | DeLong (Sun & Xu fast algorithm); variance checked against the O(n^2) definition | `statistical_tests.py` |
| Sensitivity / specificity CI | exact Clopper-Pearson (checked against `scipy.stats.binomtest`) | `metrics.py` |
| Bootstrap CI | percentile, stratified; **patient-clustered** (whole patients resampled) when `patient_id` exists | `bootstrap_ci` |
| Multiple comparisons | Holm-Bonferroni over a pre-declared family | `holm` |
| Seed variability | mean, sample SD, t-interval over independent seeds, *conditional on the fixed test set* | `seed_summary` |
| Calibration | ECE (equal-width bins), Brier, NLL, reliability curve, temperature scaling on validation | `calibration.py` |
| Uncertainty | error-detection AUROC/AUPRC, risk-coverage, selective accuracy | `uncertainty.py` |

Caveats that apply to any analysis made with this code:
* Several radiographs per patient are **not independent**. Image-level DeLong/Clopper-Pearson intervals are then
  too narrow; use the clustered bootstrap when patient ids exist (tested: it is >1.3x wider on a correlated
  construction). TBX11K has no patient ids, so its intervals are image-level by necessity.
* Seed t-intervals describe training-run variability, not patient-sampling uncertainty or paired significance.
* Define the comparison family (and the Holm adjustment) before looking at results; `compare_models.py` adjusts
  over exactly the comparators you pass.
* Delta-AUC from the same cases uses the *paired* test; do not compare marginal intervals by eye.
