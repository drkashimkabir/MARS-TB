"""Multi-Scale Vision-Language Encoder (MS-VLE), manuscript Section 2.4.2, Eqs. 4-6.

Three branches read three *different* backbone layers:
    scale 1  layer 12  full 32x32 grid (1024 tokens)
    scale 2  layer  8  2x2 average pooled   -> 16x16 grid (256 tokens)
    scale 3  layer  4  4x4 average pooled   ->  8x8  grid ( 64 tokens)
Each is conditioned on fixed text embeddings by residual cross-attention (visual tokens = query,
text = key/value), globally pooled, concatenated (3x768 = 2304) and projected back to 768.

Text conditioning uses FIXED PROMPTS (scale_prompts.yaml), never patient reports, so inference
does not need a report. Embeddings are either learned (default, no language prior) or loaded from
a file produced by a frozen text encoder - for a frozen encoder and fixed prompts, precomputing
the embeddings is exactly equivalent to running the encoder.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from mars_tb.config import MSVLEConfig, TextConfig

SCALE_KEYS = ("scale1_fine", "scale2_lobe", "scale3_whole_lung")


def load_prompts(source: str = "builtin") -> dict[str, list[str]]:
    if source == "builtin":
        text = resources.files("mars_tb.resources").joinpath("scale_prompts.yaml").read_text("utf-8")
        data = yaml.safe_load(text)
    else:
        data = yaml.safe_load(Path(source).read_text("utf-8"))
    missing = [k for k in SCALE_KEYS if k not in data or not data[k]]
    if missing:
        raise ValueError(f"prompt file missing scales: {missing}")
    return {k: list(data[k]) for k in SCALE_KEYS}


class CrossAttention(nn.Module):
    """Bias-free multi-head cross attention with an explicit attention width.

    Defaults (attn_dim=128, 8 heads, no bias) give 4*768*128 = 393,216 parameters, the figure
    listed for each text-guidance block in Supplementary Table 1.
    """

    def __init__(self, dim: int, attn_dim: int, heads: int, kv_dim: int | None = None, bias: bool = False):
        super().__init__()
        if attn_dim % heads:
            raise ValueError("attn_dim must be divisible by heads")
        kv_dim = kv_dim or dim
        self.heads = heads
        self.attn_dim = attn_dim
        self.q = nn.Linear(dim, attn_dim, bias=bias)
        self.k = nn.Linear(kv_dim, attn_dim, bias=bias)
        self.v = nn.Linear(kv_dim, attn_dim, bias=bias)
        self.o = nn.Linear(attn_dim, dim, bias=bias)
        self.last_attention: torch.Tensor | None = None
        self.store_attention = False

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, nq, _ = query.shape
        nk = kv.shape[1]
        h, d = self.heads, self.attn_dim // self.heads
        q = self.q(query).view(b, nq, h, d).transpose(1, 2)
        k = self.k(kv).view(b, nk, h, d).transpose(1, 2)
        v = self.v(kv).view(b, nk, h, d).transpose(1, 2)
        if self.store_attention:
            att = torch.softmax(q @ k.transpose(-1, -2) / d**0.5, dim=-1)
            self.last_attention = att.detach()
            out = att @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(b, nq, self.attn_dim))


class PromptBank(nn.Module):
    """Per-scale text embeddings, shape (K_s, dim). Learned or loaded from a file."""

    def __init__(self, dim: int, cfg: TextConfig):
        super().__init__()
        prompts = load_prompts(cfg.prompts)
        self.prompt_text = prompts
        self.source = cfg.source
        counts = [len(prompts[k]) for k in SCALE_KEYS]
        if cfg.source == "learned":
            self.embeddings = nn.ParameterList(nn.Parameter(torch.randn(c, dim) * 0.02) for c in counts)
        else:
            assert cfg.path is not None
            blob = torch.load(cfg.path, map_location="cpu", weights_only=True)
            tensors = []
            for key, c in zip(SCALE_KEYS, counts, strict=True):
                t = blob[key]
                if t.shape != (c, dim):
                    raise ValueError(f"{key}: expected {(c, dim)}, got {tuple(t.shape)}")
                tensors.append(t.float())
            for i, t in enumerate(tensors):
                self.register_buffer(f"fixed_{i}", t)
            self.embeddings = nn.ParameterList()

    def get(self, scale: int, batch: int) -> torch.Tensor:
        if self.source == "learned":
            emb = self.embeddings[scale]
        else:
            emb = getattr(self, f"fixed_{scale}")
        return emb.unsqueeze(0).expand(batch, -1, -1)


def pool_tokens(tokens: torch.Tensor, grid: int, kernel: int) -> torch.Tensor:
    """Average-pool a (B, grid*grid, D) token sequence spatially; returns (B, (grid/k)^2, D)."""
    if kernel == 1:
        return tokens
    b, n, d = tokens.shape
    if n != grid * grid:
        raise ValueError(f"token count {n} != grid^2 ({grid * grid})")
    x = tokens.transpose(1, 2).reshape(b, d, grid, grid)
    x = F.avg_pool2d(x, kernel_size=kernel, stride=kernel)
    return x.flatten(2).transpose(1, 2)


class MultiScaleEncoder(nn.Module):
    def __init__(self, dim: int, grid: int, cfg: MSVLEConfig, text_cfg: TextConfig):
        super().__init__()
        self.dim, self.grid, self.cfg = dim, grid, cfg
        self.kernels = (1, cfg.pool_kernels[0], cfg.pool_kernels[1])
        self.prompts = PromptBank(dim, text_cfg)
        self.guidance = nn.ModuleList(CrossAttention(dim, cfg.attn_dim, cfg.heads) for _ in range(3))
        self.fuse = nn.Linear(3 * dim, dim, bias=False)  # Supplementary Table 1: no bias
        self.norm = nn.LayerNorm(dim)

    def forward(self, z12: torch.Tensor, z8: torch.Tensor, z4: torch.Tensor) -> dict[str, torch.Tensor]:
        b = z12.shape[0]
        vecs = []
        for s, (z, kern) in enumerate(zip((z12, z8, z4), self.kernels, strict=True)):
            zs = pool_tokens(z, self.grid, kern)
            text = self.prompts.get(s, b).to(zs.dtype)
            zs = zs + self.guidance[s](zs, text)  # Eq. 5, residual cross-attention
            vecs.append(zs.mean(dim=1))  # global average pool (Eq. 6)
        z_ms = self.norm(self.fuse(torch.cat(vecs, dim=-1)))
        return {"z_ms": z_ms, "scale_vectors": torch.stack(vecs, dim=1)}
