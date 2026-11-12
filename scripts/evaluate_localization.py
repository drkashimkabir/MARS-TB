#!/usr/bin/env python
"""Compare Grad-CAM++ maps with reference TB-region boxes (TB-positive images only).

``--boxes-csv`` columns: image_id, x0, y0, x1, y1 in the ORIGINAL image's pixel coordinates, plus
orig_w, orig_h. Boxes are rescaled to the 512x512 model frame (the preprocessing resize does not
preserve aspect ratio, so x and y scale independently). Reference boxes exist only for TB-positive
TBX11K images; metrics are undefined for negatives and are not computed for them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mars_tb.data.preprocessing import load_and_preprocess, normalize, preprocess_config_from_dict
from mars_tb.explainability.gradcam_pp import GradCAMPlusPlus
from mars_tb.explainability.localization_metrics import energy_in_boxes, iou_at_threshold, pointing_game, resize_map
from mars_tb.pipeline import pick_device
from mars_tb.training.checkpointing import restore_model
from mars_tb.utils.io import safe_join


def scale_boxes(rows: pd.DataFrame, size: int) -> list[tuple[float, float, float, float]]:
    return [
        (r.x0 * size / r.orig_w, r.y0 * size / r.orig_h, r.x1 * size / r.orig_w, r.y1 * size / r.orig_h)
        for r in rows.itertuples()
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--boxes-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    manifest = pd.read_csv(args.manifest, dtype={"image_id": str}).set_index("image_id")
    boxes = pd.read_csv(args.boxes_csv, dtype={"image_id": str})
    need = {"image_id", "x0", "y0", "x1", "y1", "orig_w", "orig_h"}
    if not need <= set(boxes.columns):
        print(f"error: boxes CSV needs columns {sorted(need)}", file=sys.stderr)
        return 2
    model, blob, _ = restore_model(args.checkpoint)
    dev = pick_device(args.device)
    model.to(dev).eval()
    pp = preprocess_config_from_dict(blob["preprocess"])
    cam_fn = GradCAMPlusPlus(model, model.backbone.blocks[-1], has_cls_token=True)  # type: ignore[union-attr]
    rows = []
    for image_id, grp in boxes.groupby("image_id"):
        if image_id not in manifest.index or int(manifest.loc[image_id, "label_binary"]) != 1:
            continue
        raw = load_and_preprocess(safe_join(args.root, manifest.loc[image_id, "path"]), pp)
        cam = resize_map(cam_fn(normalize(raw, pp).unsqueeze(0).to(dev))[0].cpu().numpy(), (pp.size, pp.size))
        bx = scale_boxes(grp, pp.size)
        rows.append(
            {
                "image_id": image_id,
                "n_boxes": len(bx),
                "energy_in_boxes": energy_in_boxes(cam, bx),
                "iou_at_0.5": iou_at_threshold(cam, bx, 0.5),
                "pointing_game": pointing_game(cam, bx),
            }
        )
    cam_fn.close()
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, lineterminator="\n")
    if out.empty:
        print("no TB-positive images with boxes were found")
        return 1
    print(out.drop(columns="image_id").agg(["mean", "median"]).to_string())
    print(f"n={len(out)} images; maps are attributions, not validated lesion localisation")
    return 0 if np.isfinite(out["pointing_game"]).all() else 1


if __name__ == "__main__":
    sys.exit(main())
