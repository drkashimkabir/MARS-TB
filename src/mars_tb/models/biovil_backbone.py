"""ViT-B/16 backbone with intermediate-layer extraction, as specified in the manuscript.

IMPORTANT. The manuscript describes BioViL-T as "a ViT-B/16-based radiology image-text model" with
12 layers, width 768 and ~86 M parameters. The official BioViL-T image encoder (Microsoft model
card, ``microsoft/BiomedVLP-BioViL-T``) is a *hybrid ResNet-50 + Vision-Transformer* whose
transformer only aggregates features across time points. Its checkpoint therefore cannot be
loaded into the 12-layer ViT-B/16 implemented here (known_discrepancies.md, discrepancy I).

This module implements the architecture the manuscript *describes*. Weights are random unless
you supply a timm-format ViT-B/16 state dict. Nothing is downloaded automatically.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from mars_tb.config import BackboneConfig, LoRAConfig
from mars_tb.models.lora_adapter import inject_lora

_LN_EPS = 1e-6  # timm ViT convention


class Attention(nn.Module):
    """Multi-head self-attention with separate q/k/v/o projections (so q and v can take LoRA)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.heads

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, n, h, c // h).transpose(1, 2)

        out = F.scaled_dot_product_attention(split(self.q(x)), split(self.k(x)), split(self.v(x)))
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=_LN_EPS)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=_LN_EPS)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def interpolate_pos_embed(pos: torch.Tensor, old_grid: int, new_grid: int) -> torch.Tensor:
    """Bicubic resize of a (1, 1+g*g, D) position embedding; the CLS entry is kept as is."""
    if old_grid == new_grid:
        return pos
    cls_pos, grid_pos = pos[:, :1], pos[:, 1:]
    if grid_pos.shape[1] != old_grid * old_grid:
        raise ValueError(f"pos_embed has {grid_pos.shape[1]} patch entries, expected {old_grid}^2")
    d = pos.shape[-1]
    grid_pos = grid_pos.reshape(1, old_grid, old_grid, d).permute(0, 3, 1, 2)
    grid_pos = F.interpolate(grid_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, d)
    return torch.cat([cls_pos, grid_pos], dim=1)


class ViTBackbone(nn.Module):
    """Plain ViT returning hidden states after selected blocks (CLS excluded).

    ``Z^(l)`` in the manuscript is the output of transformer block ``l`` (1-based) *before* any
    final LayerNorm; that is what :meth:`forward` returns for each requested layer.
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = cfg.img_size // cfg.patch_size
        self.num_patches = self.grid * self.grid
        self.patch_embed = nn.Conv2d(3, cfg.dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, cfg.dim))
        self.blocks = nn.ModuleList(Block(cfg.dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.dim, eps=_LN_EPS)  # kept for weight loading / parity tests
        self._reset()

    def _reset(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_layers: Sequence[int] | None = None) -> dict[int, torch.Tensor]:
        layers = tuple(return_layers if return_layers is not None else self.cfg.layers)
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (B,3,H,W) input, got {tuple(x.shape)}")
        if x.shape[-2:] != (self.cfg.img_size, self.cfg.img_size):
            raise ValueError(f"expected {self.cfg.img_size}x{self.cfg.img_size} input, got {tuple(x.shape[-2:])}")
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        tokens = tokens + self.pos_embed
        wanted = set(layers)
        out: dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.blocks, start=1):
            if self.cfg.gradient_checkpointing and self.training and tokens.requires_grad:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if i in wanted:
                out[i] = tokens[:, 1:]
            if i >= max(wanted) and len(out) == len(wanted):
                break
        return out


def load_timm_vit_state_dict(
    model: ViTBackbone, state: Mapping[str, torch.Tensor], pretrained_grid: int | None = None
) -> list[str]:
    """Map a timm ``VisionTransformer`` state dict (fused qkv) onto :class:`ViTBackbone`.

    Returns the list of source keys that were not consumed (e.g. the classification head).
    Position embeddings are bicubically resized from ``pretrained_grid`` to the model grid.
    """
    cfg = model.cfg
    new: dict[str, torch.Tensor] = {}
    used: set[str] = set()

    def take(src: str, dst: str) -> None:
        if src not in state:
            raise KeyError(f"missing key in state dict: {src}")
        new[dst] = state[src]
        used.add(src)

    take("cls_token", "cls_token")
    take("patch_embed.proj.weight", "patch_embed.weight")
    take("patch_embed.proj.bias", "patch_embed.bias")
    pos = state["pos_embed"]
    used.add("pos_embed")
    old_grid = pretrained_grid or int(math.isqrt(pos.shape[1] - 1))
    new["pos_embed"] = interpolate_pos_embed(pos, old_grid, model.grid)
    d = cfg.dim
    for i in range(cfg.depth):
        p = f"blocks.{i}."
        for part in ("norm1", "norm2"):
            for kind in ("weight", "bias"):
                take(f"{p}{part}.{kind}", f"{p}{part}.{kind}")
        for kind in ("weight", "bias"):
            qkv_key = f"{p}attn.qkv.{kind}"
            if qkv_key not in state:
                raise KeyError(f"missing key in state dict: {qkv_key}")
            q, k, v = state[qkv_key].split(d, dim=0)
            new[f"{p}attn.q.{kind}"], new[f"{p}attn.k.{kind}"], new[f"{p}attn.v.{kind}"] = q, k, v
            used.add(qkv_key)
            take(f"{p}attn.proj.{kind}", f"{p}attn.proj.{kind}")
            take(f"{p}mlp.fc1.{kind}", f"{p}mlp.fc1.{kind}")
            take(f"{p}mlp.fc2.{kind}", f"{p}mlp.fc2.{kind}")
    take("norm.weight", "norm.weight")
    take("norm.bias", "norm.bias")
    model.load_state_dict(new, strict=True)
    return sorted(set(state) - used)


def build_backbone(cfg: BackboneConfig, lora: LoRAConfig) -> ViTBackbone:
    """Build the backbone, optionally load weights, inject LoRA, then apply freezing."""
    model = ViTBackbone(cfg)
    if cfg.pretrained == "timm_state_dict":
        if not cfg.pretrained_path:
            raise ValueError("backbone.pretrained='timm_state_dict' requires pretrained_path")
        state = torch.load(cfg.pretrained_path, map_location="cpu", weights_only=True)
        load_timm_vit_state_dict(model, state, cfg.pretrained_grid)
    elif cfg.pretrained != "none":
        raise ValueError(f"unknown backbone.pretrained: {cfg.pretrained}")
    if lora.enabled:
        inject_lora(model, lora)
    if cfg.freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = "lora_" in name
    return model
