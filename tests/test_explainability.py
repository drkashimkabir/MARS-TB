import math

import numpy as np
import pytest
import torch
from torch import nn

from mars_tb.explainability.gradcam_pp import GradCAMPlusPlus
from mars_tb.explainability.graph_attention import graph_outputs, node_table, plot_zone_overlay
from mars_tb.explainability.localization_metrics import (
    box_mask,
    energy_in_boxes,
    iou_at_threshold,
    pointing_game,
    resize_map,
)
from mars_tb.explainability.zone_perturbation import zone_occlusion, zone_pixel_box
from mars_tb.models.anatomical_graph import load_graph_spec
from mars_tb.models.mars_tb import MARSTB
from tests.conftest import tiny_config


class Toy(nn.Module):
    """Score depends only on a conv feature map: strong weights, so the CAM must follow the input."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1, bias=False)
        nn.init.constant_(self.conv.weight, 0.5)
        self.w = nn.Parameter(torch.tensor([1.0, 1.0]))

    def forward(self, x):
        a = torch.relu(self.conv(x))
        logit = (a.mean((2, 3)) * self.w).sum(1)
        zeros = torch.zeros_like(logit)
        logits = torch.stack([zeros, logit], dim=1)
        return {"tb": {"alpha": None, "logits": logits}}


def test_gradcam_pp_peaks_where_the_evidence_is():
    model = Toy()
    x = torch.zeros(1, 1, 16, 16)
    x[..., 2:6, 2:6] = 1.0  # evidence in the top-left
    cam = GradCAMPlusPlus(model, model.conv)(x)
    assert cam.shape == (1, 16, 16) and cam.min() >= 0 and cam.max() == pytest.approx(1.0)
    y, xx = np.unravel_index(int(cam[0].argmax()), cam[0].shape)
    assert 1 <= y <= 6 and 1 <= xx <= 6
    assert cam[0, 8:, 8:].max() < 0.2  # little attribution in the evidence-free quadrant


def test_gradcam_pp_on_marstb_vit_tokens_shape():
    model = MARSTB(tiny_config()).eval()
    cam_fn = GradCAMPlusPlus(model, model.backbone.blocks[11], has_cls_token=True)
    cam = cam_fn(torch.randn(2, 3, 256, 256))
    assert cam.shape == (2, 16, 16) and torch.isfinite(cam).all()
    assert (cam.flatten(1).max(1).values <= 1.0 + 1e-6).all()
    cam_fn.close()


def test_gradcam_restores_training_mode_and_requires_gradient_path():
    model = Toy().train()
    GradCAMPlusPlus(model, model.conv)(torch.rand(1, 1, 8, 8))
    assert model.training


def test_resize_and_box_helpers():
    m = resize_map(np.ones((4, 4)), (16, 16))
    assert m.shape == (16, 16)
    mask = box_mask([(2, 2, 6, 6)], (10, 10))
    assert mask.sum() == 16 and mask[2, 2] and not mask[6, 6]


def test_localization_metrics_known_values():
    h = np.zeros((10, 10))
    h[2:6, 2:6] = 1.0
    boxes = [(2, 2, 6, 6)]
    assert energy_in_boxes(h, boxes) == 1.0
    assert iou_at_threshold(h, boxes) == 1.0
    assert pointing_game(h, boxes) == 1.0
    assert energy_in_boxes(h, [(7, 7, 10, 10)]) == 0.0
    assert pointing_game(h, [(7, 7, 10, 10)]) == 0.0
    half = [(2, 2, 6, 4)]
    assert iou_at_threshold(h, half) == pytest.approx(0.5)
    assert all(math.isnan(f(h, [])) for f in (energy_in_boxes, iou_at_threshold, pointing_game))


def test_zone_occlusion_table_and_pixel_boxes():
    model = MARSTB(tiny_config()).eval()
    spec = load_graph_spec()
    df = zone_occlusion(model, torch.randn(1, 3, 256, 256), spec)
    assert list(df["node"]) == list(spec.node_names)
    assert np.allclose(df["delta_p"], df["p_occluded"] - df["p_baseline"])
    assert zone_pixel_box((0.1, 0.35, 0.08, 0.42), 512) == (51, 179, 41, 215)
    with pytest.raises(ValueError):
        zone_occlusion(model, torch.randn(2, 3, 256, 256), spec)


def test_zone_occlusion_with_identity_fill_changes_nothing():
    model = MARSTB(tiny_config()).eval()
    img = torch.zeros(1, 3, 256, 256)  # already equal to the fill value
    df = zone_occlusion(model, img, load_graph_spec(), fill=0.0)
    assert df["delta_p"].abs().max() < 1e-6


def test_graph_outputs_and_overlay(tmp_path):
    model = MARSTB(tiny_config()).eval()
    out = graph_outputs(model, torch.randn(2, 3, 256, 256))
    assert out["node_prob"].shape == (2, 12) and out["attention"].shape == (2, 3, 12, 12)
    spec = load_graph_spec()
    table = node_table(spec, out["node_prob"][0].numpy())
    assert len(table) == 12
    plot_zone_overlay(np.random.rand(64, 64), spec, out["node_prob"][0].numpy(), tmp_path / "o.png")
    assert (tmp_path / "o.png").stat().st_size > 0
    with pytest.raises(ValueError):
        graph_outputs(MARSTB(tiny_config(use_agrn=False)), torch.randn(1, 3, 256, 256))
