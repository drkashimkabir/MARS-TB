"""Agreement between a saliency map and reference bounding boxes (when such boxes exist).

Boxes are (x0, y0, x1, y1) in the SAME coordinate frame as the (resized) heat-map: pass boxes
after applying the identical resize/flip used for the image. TBX11K provides TB region boxes for
TB-positive images only, so these metrics are undefined for TB-negative cases.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Box = tuple[float, float, float, float]


def resize_map(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of an (h, w) map to (H, W) = ``size``."""
    return cv2.resize(heatmap.astype(np.float32), (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def box_mask(boxes: Sequence[Box], shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=bool)
    for x0, y0, x1, y1 in boxes:
        mask[int(max(0, y0)) : int(min(shape[0], np.ceil(y1))), int(max(0, x0)) : int(min(shape[1], np.ceil(x1)))] = (
            True
        )
    return mask


def energy_in_boxes(heatmap: np.ndarray, boxes: Sequence[Box]) -> float:
    """Fraction of total (non-negative) map energy that falls inside the reference boxes."""
    total = float(np.clip(heatmap, 0, None).sum())
    if total == 0 or not boxes:
        return float("nan")
    return float(np.clip(heatmap, 0, None)[box_mask(boxes, heatmap.shape)].sum() / total)


def iou_at_threshold(heatmap: np.ndarray, boxes: Sequence[Box], rel_threshold: float = 0.5) -> float:
    """IoU between {map >= rel_threshold * max} and the union of reference boxes."""
    if not boxes or heatmap.max() <= 0:
        return float("nan")
    pred = heatmap >= rel_threshold * heatmap.max()
    ref = box_mask(boxes, heatmap.shape)
    union = np.logical_or(pred, ref).sum()
    return float(np.logical_and(pred, ref).sum() / union) if union else float("nan")


def pointing_game(heatmap: np.ndarray, boxes: Sequence[Box]) -> float:
    """1.0 if the map's maximum lies inside any reference box, else 0.0."""
    if not boxes:
        return float("nan")
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return float(box_mask(boxes, heatmap.shape)[y, x])
