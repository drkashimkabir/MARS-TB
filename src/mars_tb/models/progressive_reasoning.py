"""Progressive Reasoning Module (PRM), manuscript Section 2.4.5, Eq. 11.

Four staged feature transformations: Screen -> Localize -> Characterize -> Diagnose.
Stage 2 additionally concatenates the 12 AGRN regional activations (768 + 12 = 780 -> 768).
Eq. 11 reuses the symbol W_1 in stages 1 and 2 and W_3 in stages 3 and 4; the implementation uses
one independent weight matrix per stage. "Reasoning" here means feature transformation only.
"""

from __future__ import annotations

import torch
from torch import nn

from mars_tb.config import PRMConfig


class Stage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_gelu: bool, residual: bool):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU() if use_gelu else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.residual = residual and in_dim == out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.linear(x))
        return self.norm(x + y if self.residual else y)


class ProgressiveReasoning(nn.Module):
    def __init__(self, dim: int, node_dim: int, cfg: PRMConfig):
        super().__init__()
        self.node_dim = node_dim
        self.screen = Stage(dim, dim, cfg.use_gelu, cfg.residual)
        self.localize = Stage(dim + node_dim, dim, cfg.use_gelu, False)
        self.characterize = Stage(dim, dim, cfg.use_gelu, cfg.residual)
        self.diagnose = Stage(dim, dim, cfg.use_gelu, cfg.residual)

    def forward(self, c: torch.Tensor, p_node: torch.Tensor | None) -> dict[str, torch.Tensor]:
        z1 = self.screen(c)
        if self.node_dim:
            if p_node is None or p_node.shape[-1] != self.node_dim:
                raise ValueError(f"stage 2 needs node activations of width {self.node_dim}")
            z1_in = torch.cat([z1, p_node.to(z1.dtype)], dim=-1)
        else:
            z1_in = z1
        z2 = self.localize(z1_in)
        z3 = self.characterize(z2)
        zd = self.diagnose(z3)
        return {"z_d": zd, "stages": torch.stack([z1, z2, z3, zd], dim=1)}
