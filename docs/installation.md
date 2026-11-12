# Installation

Tested here: Windows 11, Python 3.10.11, PyTorch 2.13.0 (CPU build), torchvision 0.28.0, numpy 1.26.4,
OpenCV 4.10.0, pandas 2.3.3, scipy 1.15.3. This is a **modern** environment; the manuscript used PyTorch 2.1.0,
CUDA 12.1, OpenCV 4.8 and torchvision 0.16 on 2 x A100 (those are the only versions it records, and others are
not guessed here).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # library + test/lint tooling (+ timm for weight-loading tests)
pip install -e ".[text]"           # optional: transformers, only for precompute_text_embeddings.py
```

GPU: install the CUDA build of PyTorch that matches your driver from pytorch.org first, then
`pip install -e .`. Nothing in the repository downloads weights or data automatically.

Verify:
```bash
python -m pytest -m "not slow"          # 265 tests, ~2 min on CPU
python -m pytest -m slow                # full-size ViT-B/16 LoRA parameter-count test
python scripts/validate_artifacts.py
```

`requirements.txt` / `environment.yml` pin lower bounds only; for an exact record of a run use the
`environment.json` that `scripts/train.py` writes next to each checkpoint.
