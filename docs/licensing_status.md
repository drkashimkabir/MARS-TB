# Licensing status

**No licence file is included yet, deliberately.** Publishing source code without a licence leaves it "all
rights reserved"; adding one is a decision for the people who hold the rights, and nobody has confirmed that
yet. Before the repository is made public the owner (the corresponding author, `drkashimkabir`, together with
any co-authors or institutions with a claim) should choose a licence and add `LICENSE` and the `license`
field in `pyproject.toml`.

A permissive licence (MIT or Apache-2.0) is compatible with every dependency listed below. Apache-2.0 adds an
explicit patent grant.

| Dependency | Licence (as published) | Note |
|---|---|---|
| PyTorch, torchvision | BSD-3-Clause | |
| NumPy, SciPy, pandas | BSD-3-Clause | |
| OpenCV (opencv-python-headless) | Apache-2.0 | |
| pydicom | MIT | |
| PyYAML | MIT | |
| jsonschema | MIT | |
| matplotlib | PSF-based (BSD-compatible) | |
| timm (optional, tests) | Apache-2.0 | |
| transformers (optional) | Apache-2.0 | |
| Microsoft BioViL-T / CXR-BERT model code | MIT (model card) | not bundled; fetched only when you run `precompute_text_embeddings.py --trust-remote-code` |

Dataset licences are separate and are **not** covered by any code licence (`dataset_card.md`).
Checklist texts under `reporting/official_checklist_text/` are quoted from CLAIM 2024 (item titles only) and
TRIPOD+AI (CC BY 4.0, BMJ) with attribution in the file headers.
