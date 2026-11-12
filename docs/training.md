# Training

```bash
# 1. manifest -> split (newly generated, labelled as such)
python scripts/prepare_tbx11k.py --root <TBX11K> --out data/manifests/tbx11k.csv
python scripts/generate_splits.py --manifest data/manifests/tbx11k.csv --root <TBX11K> \
       --name tbx11k_seed0 --seed 0 --strata label
python scripts/verify_splits.py --split-csv data/splits/tbx11k_seed0.csv --root <TBX11K>
python scripts/audit_data_leakage.py --split-csv data/splits/tbx11k_seed0.csv --root <TBX11K>

# 2. train (single process)
python scripts/train.py --split-csv data/splits/tbx11k_seed0.csv --root <TBX11K> \
       --out-dir runs/mars_tb/seed42 --seed 42

# multi-GPU (NOT exercised in CI)
torchrun --nproc_per_node=2 scripts/train.py ...
```

Defaults reproduce Table 3: AdamW, LoRA lr 1e-4, new modules 1e-5, backbone 0 (frozen), weight decay 0.01,
betas (0.9, 0.999), 5 warm-up epochs then cosine to 1e-7, 100 epochs, per-GPU batch 16, accumulation 2, clip 1.0,
AMP on CUDA, early stopping patience 15 on validation AUC, KL weight `min(1, epoch/10)`, task weights
1.0 : 0.3 : 0.3.

* **Selection is validation-only.** The trainer receives the train and val loaders only; `train_from_split`
  never reads `test` rows (a test poisons the test paths and still passes). The checkpoint stores the Youden
  threshold computed on the validation set at the best epoch.
* **Effective batch size** = per-GPU x accumulation x processes (printed at start; discrepancy H).
* **Sampler weight** = n_neg / n_pos of the *training* labels (8.40 for 6,000/714); `sampler_weights: fixed:4.5`
  reproduces the manuscript's stale value deliberately.
* **Auxiliary tasks** train only on verified labels: put verified `pattern_label` / `severity_label` columns in
  the manifest; empty cells contribute nothing. Without them only the TB head learns.
* **Ablations / baselines:** `--ablation A6a_no_ddfm`, `--baseline ResNet-50` (CheXNet and TBNet are refused);
  `scripts/run_ablations.py` loops ablations x seeds (42, 0, 1, 2, 3).
* **Checkpoints** (`best.ckpt` + `best.ckpt.sha256`) hold weights, config, seed, epoch, validation AUC,
  threshold, preprocessing config, label mapping, library versions and git commit, and load with
  `weights_only=True`. 6 GB GPUs: set `backbone.gradient_checkpointing: true` and reduce the per-GPU batch.
* Smoke test with no data: `python scripts/make_synthetic_fixture.py --out-dir /tmp/syn` (synthetic, not clinical).
