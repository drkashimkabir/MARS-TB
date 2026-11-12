"""Run a model over a loader and export predictions in the documented schema."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from mars_tb.data.metadata import PREDICTION_COLUMNS


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device | str,
    *,
    dataset: str,
    split: str,
    model_name: str,
    seed: int | None,
    threshold: float,
    checkpoint_hash: str,
    extra_columns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Deterministic inference (eval mode, no grad). Values come only from the model's outputs;
    non-evidential models leave alpha_*/uncertainty empty rather than inventing them."""
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        out = model(batch["image"].to(device))["tb"]
        p = out["p_tb"].float().cpu().numpy()
        alpha = out["alpha"].float().cpu().numpy() if out["alpha"] is not None else None
        unc = out["uncertainty"].float().cpu().numpy() if out["uncertainty"] is not None else None
        for i, image_id in enumerate(batch["image_id"]):
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "image_id": image_id,
                    "patient_id": batch["patient_id"][i] or None,
                    "model": model_name,
                    "seed": seed,
                    "actual_label": None if int(batch["label"][i]) < 0 else int(batch["label"][i]),
                    "p_tb": float(p[i]),
                    "predicted_label": int(p[i] >= threshold),
                    "threshold": float(threshold),
                    "uncertainty": None if unc is None else float(unc[i]),
                    "alpha_negative": None if alpha is None else float(alpha[i, 0]),
                    "alpha_positive": None if alpha is None else float(alpha[i, 1]),
                    "checkpoint_hash": checkpoint_hash,
                }
            )
    df = pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))
    if extra_columns is not None:
        df = df.merge(extra_columns, on="image_id", how="left", validate="many_to_one")
    return df


def metrics_from_predictions(df: pd.DataFrame) -> dict[str, Any]:
    from mars_tb.evaluation.metrics import binary_report

    if df["threshold"].nunique() != 1:
        raise ValueError("predictions mix several thresholds; evaluate each threshold separately")
    return binary_report(
        df["actual_label"].astype(int).to_numpy(),
        df["p_tb"].to_numpy(dtype=float),
        float(df["threshold"].iloc[0]),
    )


def prediction_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df["actual_label"].astype(int).to_numpy(), df["p_tb"].to_numpy(dtype=float)
