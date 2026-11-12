# Explainability

```bash
python scripts/generate_explanations.py --checkpoint runs/seed42/best.ckpt --manifest split.csv --root <root> \
       --predictions results/metrics/test_predictions.csv --category FP --n 4 --out-dir results/figures/fp
python scripts/evaluate_localization.py --checkpoint ... --manifest split.csv --root <root> \
       --boxes-csv tb_boxes.csv --out results/metrics/localization.csv
```

| Method | Output | Notes |
|---|---|---|
| Grad-CAM++ | overlay PNG per image | on the layer-12 patch-token grid; target = log(alpha_pos / alpha_neg) |
| AGRN regional activations | `agrn_node_activations.csv`, zone overlay | unsupervised readout; **not lesion probabilities** |
| Graph attention | `explainability/graph_attention.py` | head-averaged (B, 3 layers, 12, 12) |
| Zone occlusion | `zone_occlusion.csv` | change in P(TB) when one zone is neutralised |
| Localisation | energy-in-box, IoU at 0.5, pointing game | TB-positive images with reference boxes only |

* **TP/TN/FP/FN selection** reads correctness from a prediction file's stored threshold and labels; it is never
  inferred from the label alone.
* Saliency is an attribution of one checkpoint's score. It is not validated lesion localisation unless compared
  with reference annotations (`evaluate_localization.py`), and a map from a newly trained model does **not**
  reproduce the manuscript's Figures 7-8 (those cases' probabilities and thresholds are not documented).
* Boxes passed to the localisation script are rescaled independently in x and y because the preprocessing resize
  does not preserve aspect ratio.
