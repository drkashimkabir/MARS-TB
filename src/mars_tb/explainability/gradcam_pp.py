"""Grad-CAM++ (Chattopadhyay et al., 2018) for conv feature maps and ViT patch tokens.

For activations A^k and gradients g = dY/dA^k, with the exponential-score closed form,
    alpha^k_ij = g^2 / (2 g^2 + sum_ab A^k_ab g^3),   w_k = sum_ij alpha^k_ij ReLU(g^k_ij),
    CAM = ReLU(sum_k w_k A^k).
For a ViT the "feature map" is the patch-token grid of the chosen block (CLS dropped).

A saliency map is an attribution of one model's score, not validated lesion localisation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

ScoreFn = Callable[[dict], torch.Tensor]


def tb_logit_score(output: dict) -> torch.Tensor:
    """log(alpha_pos / alpha_neg): monotone in p_tb and not saturating like p_tb itself."""
    alpha = output["tb"]["alpha"]
    if alpha is not None:
        return torch.log(alpha[:, 1]) - torch.log(alpha[:, 0])
    logits = output["tb"]["logits"]
    return logits[:, 1] - logits[:, 0]


class GradCAMPlusPlus:
    def __init__(
        self, model: nn.Module, target_layer: nn.Module, has_cls_token: bool = False, score_fn: ScoreFn = tb_logit_score
    ):
        self.model, self.layer, self.has_cls = model, target_layer, has_cls_token
        self.score_fn = score_fn
        self._act: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        self._act = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grad", g))

    def close(self) -> None:
        self._handle.remove()

    def _to_map(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 4:
            return t
        if t.ndim == 3:
            tokens = t[:, 1:] if self.has_cls else t
            b, n, d = tokens.shape
            g = int(round(n**0.5))
            if g * g != n:
                raise ValueError(f"token count {n} is not a square grid")
            return tokens.transpose(1, 2).reshape(b, d, g, g)
        raise ValueError(f"unsupported activation rank {t.ndim}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, h, w) maps normalised to [0, 1] per image."""
        was_training = self.model.training
        self.model.eval()
        self._act = self._grad = None
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            out = self.model(x)
            score = self.score_fn(out).sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()
        self.model.train(was_training)
        if self._act is None or self._grad is None:
            raise RuntimeError("target layer produced no gradient; is it on the output path?")
        a, g = self._to_map(self._act.detach()), self._to_map(self._grad.detach())
        g2, g3 = g**2, g**3
        denom = 2 * g2 + (a * g3).sum(dim=(2, 3), keepdim=True)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        weights = (alpha * torch.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * a).sum(dim=1))
        flat = cam.flatten(1)
        mx = flat.max(dim=1, keepdim=True).values
        return (flat / mx.clamp_min(1e-12)).view_as(cam)
