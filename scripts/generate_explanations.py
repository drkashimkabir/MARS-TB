#!/usr/bin/env python
"""Grad-CAM++, AGRN zone activations and zone-occlusion sensitivity for selected images.

Selection by outcome (TP/TN/FP/FN) needs a prediction file from evaluate.py: correctness is read
from its stored threshold and reference labels, never inferred. Saliency is attribution of this
checkpoint's TB score, not validated lesion localisation. Maps from a freshly trained model do
not reproduce the manuscript's Figures 7-8.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from mars_tb.data.preprocessing import load_and_preprocess, normalize, preprocess_config_from_dict
from mars_tb.explainability.gradcam_pp import GradCAMPlusPlus
from mars_tb.explainability.graph_attention import node_table
from mars_tb.explainability.zone_perturbation import zone_occlusion
from mars_tb.models.anatomical_graph import load_graph_spec
from mars_tb.pipeline import pick_device
from mars_tb.training.checkpointing import restore_model
from mars_tb.utils.io import safe_join


def outcome(row: pd.Series) -> str:
    y, yhat = int(row["actual_label"]), int(row["predicted_label"])
    return {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(y, yhat)]


def select(preds: pd.DataFrame, category: str | None, n: int, ids: list[str] | None) -> pd.DataFrame:
    if ids:
        return preds[preds["image_id"].isin(ids)]
    preds = preds.dropna(subset=["actual_label"]).assign(outcome=lambda d: d.apply(outcome, axis=1))
    if category:
        preds = preds[preds["outcome"] == category]
    return preds.head(n)


def overlay(img_gray: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap((cv2.resize(cam, img_gray.shape[::-1]) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor((img_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 0.55, heat, 0.45, 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--predictions", type=Path, help="evaluate.py output (needed for --category)")
    ap.add_argument("--category", choices=["TP", "TN", "FP", "FN"])
    ap.add_argument("--image-ids", nargs="+")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    if args.category and not args.predictions:
        ap.error("--category requires --predictions (correctness comes from stored predictions)")

    manifest = pd.read_csv(args.manifest, dtype={"image_id": str}).set_index("image_id")
    if args.predictions:
        chosen = select(pd.read_csv(args.predictions, dtype={"image_id": str}), args.category, args.n, args.image_ids)
        chosen_ids = chosen["image_id"].tolist()
        labels = dict(zip(chosen["image_id"], chosen.get("outcome", [""] * len(chosen)), strict=False))
    else:
        chosen_ids, labels = (args.image_ids or manifest.index[: args.n].tolist()), {}
    if not chosen_ids:
        print("error: no images selected", file=sys.stderr)
        return 2

    model, blob, _ = restore_model(args.checkpoint)
    dev = pick_device(args.device)
    model.to(dev).eval()
    pp = preprocess_config_from_dict(blob["preprocess"])
    spec = load_graph_spec(model.cfg.agrn.graph_spec) if getattr(model, "agrn", None) is not None else None  # type: ignore[union-attr]
    layer = model.backbone.blocks[-1]  # type: ignore[union-attr]
    cam_fn = GradCAMPlusPlus(model, layer, has_cls_token=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    node_rows, occ_rows = [], []
    for image_id in chosen_ids:
        raw = load_and_preprocess(safe_join(args.root, manifest.loc[image_id, "path"]), pp)
        x = normalize(raw, pp).unsqueeze(0).to(dev)
        cam = cam_fn(x)[0].cpu().numpy()
        tag = labels.get(image_id, "")
        out_png = args.out_dir / f"{image_id.replace('/', '_')}{'_' + tag if tag else ''}_gradcampp.png"
        cv2.imwrite(str(out_png), overlay(raw[0].numpy(), cam))
        if spec is not None:
            with torch.no_grad():
                aux = model(x)["aux"]  # type: ignore[operator,index]
            nt = node_table(spec, aux["node_prob"][0].cpu().numpy()).assign(image_id=image_id)
            node_rows.append(nt)
            occ_rows.append(zone_occlusion(model, x, spec).assign(image_id=image_id))
    cam_fn.close()
    if node_rows:
        pd.concat(node_rows).to_csv(args.out_dir / "agrn_node_activations.csv", index=False)
        pd.concat(occ_rows).to_csv(args.out_dir / "zone_occlusion.csv", index=False)
    print(f"wrote {len(chosen_ids)} Grad-CAM++ overlays (+ CSVs) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
