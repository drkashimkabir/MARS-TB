"""MARS-TB assembled model (manuscript Figure 1).

    image -> ViT backbone (+LoRA) -+-> MS-VLE -> z_MS -> DDFM(image, z_MS) -> z_DDFM -+
                                   |                                                  +-> c = z_DDFM + z_AGRN
                                   +-> AGRN (layer-12 tokens) -> z_AGRN, p_node ------+      -> PRM -> MT-EDL head

Ablation switches (use_ddfm / use_agrn / use_prm) are defined here because the manuscript names
the ablations but not their wiring:
  * no DDFM : z_DDFM := z_MS                       * no AGRN : c := z_DDFM; PRM stage 2 sees no p_node
  * no PRM  : z_d := c
"""

from __future__ import annotations

import torch
from torch import nn

from mars_tb.config import ModelConfig, validate_model_config
from mars_tb.models.anatomical_graph import AGRN
from mars_tb.models.biovil_backbone import build_backbone
from mars_tb.models.evidential_head import MultiTaskHead
from mars_tb.models.frequency_fusion import DualDomainFusion
from mars_tb.models.multi_scale_encoder import MultiScaleEncoder
from mars_tb.models.progressive_reasoning import ProgressiveReasoning


class MARSTB(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        validate_model_config(cfg)
        self.cfg = cfg
        b = cfg.backbone
        grid = b.img_size // b.patch_size
        self.backbone = build_backbone(b, cfg.lora)
        self.msvle = MultiScaleEncoder(b.dim, grid, cfg.msvle, cfg.text)
        self.ddfm = DualDomainFusion(b.dim, b.img_size, cfg.ddfm) if cfg.use_ddfm else None
        self.agrn = AGRN(b.dim, grid, cfg.agrn) if cfg.use_agrn else None
        node_dim = self.agrn.n_nodes if self.agrn is not None else 0
        self.prm = ProgressiveReasoning(b.dim, node_dim, cfg.prm) if cfg.use_prm else None
        self.head = MultiTaskHead(b.dim, cfg.head)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        l_fine, l_mid, l_coarse = sorted(self.cfg.backbone.layers, reverse=True)
        feats = self.backbone(x, return_layers=(l_coarse, l_mid, l_fine))
        ms = self.msvle(feats[l_fine], feats[l_mid], feats[l_coarse])
        z_ms = ms["z_ms"]
        aux: dict[str, torch.Tensor] = {"scale_vectors": ms["scale_vectors"]}
        if self.ddfm is not None:
            d = self.ddfm(x, z_ms)
            z_ddfm = d["z_ddfm"]
            aux["band_features"] = d["band_features"]
        else:
            z_ddfm = z_ms
        p_node = None
        if self.agrn is not None:
            g = self.agrn(feats[l_fine])
            c = z_ddfm + g["z_agrn"]
            p_node = g["node_prob"]
            aux["node_prob"] = p_node
            aux["node_logit"] = g["node_logit"]
            aux["graph_attention"] = g["attention"]
        else:
            c = z_ddfm
        if self.prm is not None:
            z_d = self.prm(c, p_node)["z_d"]
        else:
            z_d = c
        out: dict[str, object] = dict(self.head(z_d))
        out["features"] = z_d
        out["aux"] = aux
        return out

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """LoRA / backbone / new-module parameter groups (trainable ones only)."""
        lora: list[nn.Parameter] = []
        backbone: list[nn.Parameter] = []
        new: list[nn.Parameter] = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                (lora if "lora_" in name else backbone).append(p)
            else:
                new.append(p)
        return {"lora": lora, "backbone": backbone, "new": new}


def build_mars_tb(cfg: ModelConfig) -> MARSTB:
    return MARSTB(cfg)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
