"""Comparison baselines that exist as faithful, public architectures.

Implemented (torchvision definitions, random init unless ``pretrained=True`` is passed, which
downloads ImageNet weights and is NOT done by default): ResNet-50, DenseNet-121, EfficientNet-B4,
ViT-B/16, Swin-B.

Deliberately NOT implemented, and refused rather than approximated under the same name:
  * CheXNet - DenseNet-121 *plus* ChestX-ray14 pretraining. The architecture alone is just
    DenseNet-121; without the CheXNet weights a model called "CheXNet" would be misleading.
  * TBNet - no authoritative public implementation was located; see docs/known_discrepancies.md.
BioViL-T (+full FT / +LoRA) baselines are configurations of the MARS-TB backbone
(configs/model/biovil_fullft.yaml and biovil_lora.yaml), not torchvision models.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torchvision
from torch import nn

UNAVAILABLE: dict[str, str] = {
    "CheXNet": "needs the CheXNet (ChestX-ray14) weights; DenseNet-121 alone is not CheXNet",
    "TBNet": "no authoritative public implementation located",
}


def _resnet50(pretrained: bool) -> nn.Module:
    m = torchvision.models.resnet50(weights="DEFAULT" if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def _densenet121(pretrained: bool) -> nn.Module:
    m = torchvision.models.densenet121(weights="DEFAULT" if pretrained else None)
    m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m


def _efficientnet_b4(pretrained: bool) -> nn.Module:
    m = torchvision.models.efficientnet_b4(weights="DEFAULT" if pretrained else None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def _vit_b16(pretrained: bool, image_size: int = 512) -> nn.Module:
    if pretrained:
        raise ValueError("pretrained ViT-B/16 weights are 224 px; load them through the MARS-TB backbone path")
    m = torchvision.models.vit_b_16(weights=None, image_size=image_size)
    m.heads.head = nn.Linear(m.heads.head.in_features, 2)
    return m


def _swin_b(pretrained: bool) -> nn.Module:
    m = torchvision.models.swin_b(weights="DEFAULT" if pretrained else None)
    m.head = nn.Linear(m.head.in_features, 2)
    return m


BASELINES: dict[str, Callable[..., nn.Module]] = {
    "ResNet-50": _resnet50,
    "DenseNet-121": _densenet121,
    "EfficientNet-B4": _efficientnet_b4,
    "ViT-B/16": _vit_b16,
    "Swin-B": _swin_b,
}


class BaselineWrapper(nn.Module):
    """Adapts a 2-logit classifier to the MARS-TB output dictionary (TB task only)."""

    def __init__(self, name: str, net: nn.Module):
        super().__init__()
        self.name, self.net = name, net

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        logits = self.net(x)
        prob = torch.softmax(logits.float(), dim=-1)
        return {
            "tb": {"logits": logits, "alpha": None, "prob": prob, "p_tb": prob[:, 1], "uncertainty": None},
            "features": logits,
            "aux": {},
        }

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"lora": [], "backbone": [p for p in self.parameters() if p.requires_grad], "new": []}


def build_baseline(name: str, pretrained: bool = False, **kwargs) -> BaselineWrapper:
    if name in UNAVAILABLE:
        raise NotImplementedError(f"{name} is not implemented: {UNAVAILABLE[name]}")
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINES)}")
    return BaselineWrapper(name, BASELINES[name](pretrained, **kwargs))
