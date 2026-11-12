"""Inspect AGRN: regional activations and graph-attention coefficients."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mars_tb.models.anatomical_graph import GraphSpec
from mars_tb.models.mars_tb import MARSTB


@torch.no_grad()
def graph_outputs(model: MARSTB, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Node activations (B,12) and head-averaged attention (B, layers, N, N)."""
    if model.agrn is None:
        raise ValueError("this model has no AGRN (ablated)")
    model.eval()
    aux = model(x)["aux"]
    return {"node_prob": aux["node_prob"], "attention": aux["graph_attention"].mean(dim=2)}


def node_table(spec: GraphSpec, node_prob: np.ndarray) -> pd.DataFrame:
    """Long table of regional activations. These are NOT calibrated lesion probabilities."""
    return pd.DataFrame({"node": spec.node_names, "activation": np.asarray(node_prob, dtype=float)})


def plot_zone_overlay(image: np.ndarray, spec: GraphSpec, values: np.ndarray, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    cmap = plt.get_cmap("magma")
    for name, (y0, y1, x0, x1), v in zip(spec.node_names, spec.boxes, values, strict=True):
        ax.add_patch(
            Rectangle(
                (x0 * w, y0 * h),
                (x1 - x0) * w,
                (y1 - y0) * h,
                fill=True,
                facecolor=cmap(float(v)),
                alpha=0.35,
                edgecolor="white",
                lw=1,
            )
        )
        ax.text(
            (x0 + x1) / 2 * w,
            (y0 + y1) / 2 * h,
            f"{name}\n{v:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=7,
        )
    ax.set_title("AGRN regional activations (not calibrated lesion probabilities)", fontsize=8)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
