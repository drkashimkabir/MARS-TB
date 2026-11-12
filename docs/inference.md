# Inference and prediction export

```bash
python scripts/infer.py --checkpoint runs/seed42/best.ckpt --image xray.png --out pred.csv
python scripts/infer.py --checkpoint ... --dir some_folder/ --out preds.json
python scripts/infer.py --checkpoint ... --manifest m.csv --root <data root> --out preds.csv
python scripts/export_predictions.py --checkpoint ... --manifest split.csv --root <root> \
       --dataset tbx11k --split test --only-split test --out results/metrics/test_predictions.csv
```

Output columns (schema `data/schemas/predictions.schema.json`):
`dataset, split, image_id, patient_id, model, seed, actual_label, p_tb, predicted_label, threshold, uncertainty,
alpha_negative, alpha_positive, checkpoint_hash`.

* `actual_label` is **null** for unlabelled inputs, never a placeholder 0.
* `threshold` defaults to the validation-selected value stored in the checkpoint; override with `--threshold`.
* `p_tb = alpha_pos / (alpha_neg + alpha_pos)` and `uncertainty = 2 / (alpha_neg + alpha_pos)` are checked
  numerically on export (`validate_predictions`); a softmax (cross-entropy) model leaves alpha and uncertainty empty.
* Inference is deterministic (eval mode, `no_grad`) and batch-size independent (tested to 1e-5).
* Loading refuses a checkpoint whose SHA-256 does not match its sidecar. Paths are confined to `--root`.
* Research use only. The model is not validated for clinical use.
