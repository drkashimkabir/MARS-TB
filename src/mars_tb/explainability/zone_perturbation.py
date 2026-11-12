"""Zone-level occlusion: how much does P(TB) change when one lung zone is neutralised?"""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from mars_tb.models.anatomical_graph import GraphSpec


def zone_pixel_box(box: tuple[float, float, float, float], size: int) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = box
    return int(round(y0 * size)), int(round(y1 * size)), int(round(x0 * size)), int(round(x1 * size))


@torch.no_grad()
def zone_occlusion(model: nn.Module, image: torch.Tensor, spec: GraphSpec, fill: float = 0.0) -> pd.DataFrame:
    """Replace each zone of a normalised (1,3,H,W) image by ``fill`` (0 = ImageNet-mean pixel).

    Returns one row per zone with baseline and occluded P(TB) and their difference. This measures
    the model's sensitivity to a region, which need not equal clinical relevance.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("expected a single image of shape (1,3,H,W)")
    model.eval()
    size = image.shape[-1]
    base = float(model(image)["tb"]["p_tb"][0])
    rows = []
    for name, box in zip(spec.node_names, spec.boxes, strict=True):
        y0, y1, x0, x1 = zone_pixel_box(box, size)
        occluded = image.clone()
        occluded[..., y0:y1, x0:x1] = fill
        p = float(model(occluded)["tb"]["p_tb"][0])
        rows.append({"node": name, "p_baseline": base, "p_occluded": p, "delta_p": p - base})
    return pd.DataFrame(rows)
