# Data provenance

Provenance classes used across the repository: **verified source-dataset artifact** (a public dataset, not
redistributed here), **manuscript-derived implementation**, **test fixture** (synthetic), **unavailable or
unverified**.

| Dataset | Role | Source | What is verified | What is NOT |
|---|---|---|---|---|
| TBX11K | train / val / internal test | Original: github.com/yun-liu/Tuberculosis. Study copy: kaggle.com/datasets/usmanshams/tbx-11 | Dataset exists and is cited by the manuscript | Image-level labels for all 11,200 images; patient identifiers (none documented); the authors' split |
| PadChest-TB | external only | PadChest, Hospital Universitario San Juan de Alicante | Source dataset exists | The 3,000-image selection rules; negatives are radiologically *normal*, not microbiologically TB-free |
| TB Portals cohort | external only | NIAID TB Portals (data-use agreement; API, March 2025) | Programme exists | Where the 953 negatives came from; whether positives are baseline/pre-treatment; diagnostic reference standard |
| Synthetic fixture | CI only | `mars_tb.testing.synthetic` | Generated deterministically | It is not data about anyone; results on it are not experiments |

## Rules enforced in code

* Raw images are **never** committed (`.gitignore`, `tests/test_security.py`).
* Every manifest row needs `label_provenance`; TB Portals negatives need an explicit source string or
  `prepare_tbportals.py` refuses to run.
* Patient-level separation is reported as **unverified** unless real `patient_id` values exist
  (`verify_splits.py` warns; the leakage audit says so in its verdict).
* A split is `newly_generated` unless the authors' actual file is supplied with a written attestation
  (`load_historical_split`).
* "Normal" is not "TB-free": PadChest-TB manifests carry
  `cohort_definition=proposed_reconstruction_not_the_manuscript_cohort`.

## Expected layout

```
<root>/TBX11K/imgs/*.png         -> scripts/prepare_tbx11k.py   -> manifest CSV
<root>/PadChest/<ImageID>        -> scripts/prepare_padchest.py -> manifest CSV
<root>/TBPortals/...             -> scripts/prepare_tbportals.py (needs your access-controlled table)
```

Manifest and prediction schemas: `data/schemas/*.schema.json`, examples in `data/templates/`.
