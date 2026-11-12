# Evaluation

```bash
python scripts/evaluate.py --checkpoint runs/seed42/best.ckpt --manifest data/splits/tbx11k_seed0.csv \
       --root <TBX11K> --dataset tbx11k --split test --only-split test \
       --out-predictions results/metrics/tbx11k_test_predictions.csv --out-metrics results/metrics/tbx11k_test.json
python scripts/evaluate.py ... --manifest padchest_manifest.csv --dataset padchest_tb --split external ...
```

Reported: ROC AUC, PR AUC, accuracy, sensitivity, specificity, PPV, NPV, precision, recall, F1, MCC, confusion
counts, prevalence; Brier score, NLL, ECE; for evidential models error-detection AUROC/AUPRC, risk-coverage
(AURC) and selective accuracy.

**Thresholds.** `scripts/select_threshold.py` accepts validation predictions only and refuses test/external
(`ThresholdLeakageError`); `calibrate.py` fits temperature on validation only. Thresholds are saved with the
checkpoint and with `FrozenThreshold` (method, dataset, split, n, digest of the image ids) before any external run.
External data never influence thresholds, calibration, early stopping or model choice.

**Tables and figures from real predictions only**
```bash
python scripts/reproduce_tables.py --predictions MARS-TB=runs/s42/test.csv MARS-TB=runs/s0/test.csv ... \
       --out-dir results/tables           # same NAME repeated => seed aggregation
python scripts/compare_models.py --predictions a.csv b.csv --names MARS LoRA --reference MARS --out cmp.csv
python scripts/generate_figures.py --predictions runs/s42/test.csv
python scripts/analyze_uncertainty.py --predictions runs/s42/test.csv --out-json u.json
```
`reproduce_tables.py` reads nothing from the manuscript and exits 2 if a file is missing.

**Efficiency.** `scripts/benchmark_efficiency.py` reports measured parameters, FLOPs (2 x MAC; attention
counted), batch-1 latency (warm-up + CUDA sync), throughput, peak GPU memory, checkpoint size and the
environment. The manuscript's 18.3 M / 104.6 M / 116.3 G / 167 ms are printed as *reference values, not
measurements*. A CPU run is in `reporting/measured/benchmark_cpu_default_config.json`
(103.0 M total, 16.57 M trainable, 231.6 GFLOPs = 115.8 GMACs, 7.85 s per image on this CPU; not comparable
with the A100 latency).
