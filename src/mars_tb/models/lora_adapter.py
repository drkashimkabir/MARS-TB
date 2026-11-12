"""Low-rank adaptation (manuscript Eq. 3): h' = h W0 + (alpha/r) h A B."""

from __future__ import annotations

import torch
from torch import nn

from mars_tb.config import LoRAConfig


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank update.

    ``A`` is (in, r) ~ N(0, init_std^2); ``B`` is (r, out) = 0, so the adapted layer initially
    equals the frozen layer exactly. For r=16 and a 768x768 projection this adds
    768*16 + 16*768 = 24,576 parameters (manuscript Section 2.4.1).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float, init_std: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(base.in_features, rank) * init_std)
        self.lora_B = nn.Parameter(torch.zeros(rank, base.out_features))
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        return out + (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scale

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the update into the base weight for inference (idempotent)."""
        if self.merged:
            return
        delta = (self.lora_A @ self.lora_B).t() * self.scale  # (out, in)
        self.base.weight.add_(delta)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        delta = (self.lora_A @ self.lora_B).t() * self.scale
        self.base.weight.sub_(delta)
        self.merged = False


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Replace ``attn.q`` / ``attn.v`` (per ``cfg.targets``) of every block with LoRA layers.

    Returns the dotted names of the wrapped projections. Raises if nothing matched so a
    silently-empty adaptation cannot be reported as "LoRA".
    """
    wrapped: list[str] = []
    for block_name, block in model.named_modules():
        attn = getattr(block, "attn", None)
        if attn is None or not hasattr(attn, "q"):
            continue
        for target in cfg.targets:
            lin = getattr(attn, target, None)
            if not isinstance(lin, nn.Linear):
                raise ValueError(f"{block_name}.attn.{target} is not an nn.Linear")
            setattr(attn, target, LoRALinear(lin, cfg.rank, cfg.alpha, cfg.dropout, cfg.init_std))
            wrapped.append(f"{block_name}.attn.{target}")
    if not wrapped:
        raise ValueError("no attention projections found to adapt with LoRA")
    return wrapped


def merge_lora(model: nn.Module) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
            n += 1
    return n


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    out: list[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            out.extend([m.lora_A, m.lora_B])
    return out
