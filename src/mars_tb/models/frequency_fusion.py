"""Dual-Domain Fusion Module (DDFM), manuscript Section 2.4.3, Eqs. 7-8.

Frequency bands. The manuscript gives band limits "0-16 / 17-64 / >64" but not the index
geometry. This implementation uses the *centred* 2-D spectrum (fftshift) and a configurable
distance: ``radial`` (Euclidean, default) or ``square`` (Chebyshev). The three masks partition the
spectrum exactly (every frequency belongs to exactly one band), so the three band-limited images
sum back to the input. This is an implementation choice, not a recovered detail.

Cross-attention. Taken literally, Eq. 8 attends from ONE query vector (Z_MS) to ONE key/value
vector (the concatenated 768-d band feature). A softmax over a single key is identically 1, so the
"attention" collapses to a residual linear map of the frequency feature. ``kv_mode="concat_single"``
reproduces that literal reading; the default ``band_tokens`` instead exposes the three band
vectors as three key/value tokens so the attention weights are a genuine, inspectable
low/mid/high-band weighting (known_discrepancies.md, discrepancy J).
"""

from __future__ import annotations

import torch
from torch import nn

from mars_tb.config import DDFMConfig
from mars_tb.models.multi_scale_encoder import CrossAttention

BAND_NAMES = ("low", "mid", "high")


def frequency_distance(h: int, w: int, geometry: str) -> torch.Tensor:
    """Distance of each centred-spectrum bin from the DC bin, in bin units."""
    u = torch.arange(h, dtype=torch.float32) - h // 2
    v = torch.arange(w, dtype=torch.float32) - w // 2
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    if geometry == "radial":
        return torch.sqrt(uu**2 + vv**2)
    if geometry == "square":
        return torch.maximum(uu.abs(), vv.abs())
    raise ValueError(f"unknown geometry: {geometry}")


def band_masks(h: int, w: int, edges: tuple[int, int], geometry: str = "radial") -> torch.Tensor:
    """(3, H, W) {0,1} masks: low = d<=e1, mid = e1<d<=e2, high = d>e2. Exactly partition."""
    lo, hi = edges
    dist = frequency_distance(h, w, geometry)
    masks = torch.stack([dist <= lo, (dist > lo) & (dist <= hi), dist > hi]).float()
    empty = [BAND_NAMES[i] for i in range(3) if masks[i].sum() == 0]
    if empty:
        raise ValueError(
            f"band(s) {empty} are empty for a {h}x{w} input with edges {edges}; scale ddfm.band_edges to the image size"
        )
    return masks


def split_bands(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Band-limited spatial images, (B, 3, H, W), via mask in the centred FFT domain."""
    spec = torch.fft.fftshift(torch.fft.fft2(x.float(), norm="ortho"), dim=(-2, -1))
    banded = spec.unsqueeze(1) * masks.unsqueeze(0).unsqueeze(2)  # (B,3,1,H,W)
    banded = torch.fft.ifftshift(banded, dim=(-2, -1))
    return torch.fft.ifft2(banded, norm="ortho").real.squeeze(2)


class FreqBandCNN(nn.Module):
    """Three-layer CNN: 3 x (conv-BN-ReLU, stride 2) -> global pool -> Linear."""

    def __init__(self, channels: tuple[int, int, int], out_dim: int):
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(c3, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class DualDomainFusion(nn.Module):
    masks: torch.Tensor

    def __init__(self, dim: int, img_size: int, cfg: DDFMConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.register_buffer("masks", band_masks(img_size, img_size, cfg.band_edges, cfg.geometry), persistent=False)
        self.encoders = nn.ModuleList(FreqBandCNN(cfg.cnn_channels, cfg.feat_dim) for _ in range(3))
        if cfg.kv_mode == "band_tokens":
            self.attn = CrossAttention(dim, cfg.attn_dim, cfg.heads, kv_dim=cfg.feat_dim, bias=False)
        else:
            self.attn = CrossAttention(dim, cfg.attn_dim, cfg.heads, kv_dim=3 * cfg.feat_dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def band_features(self, image: torch.Tensor) -> torch.Tensor:
        """(B, 3, feat_dim) per-band feature vectors from the raw radiograph."""
        gray = image.mean(dim=1, keepdim=True)  # channels are replicated copies of one image
        bands = split_bands(gray, self.masks)  # (B,3,H,W)
        if self.cfg.band_input == "log_magnitude":
            spec = torch.fft.fftshift(torch.fft.fft2(gray.float(), norm="ortho"), dim=(-2, -1))
            bands = torch.log1p((spec * self.masks.unsqueeze(0)).abs())
        feats = [enc(bands[:, i : i + 1].to(gray.dtype)) for i, enc in enumerate(self.encoders)]
        return torch.stack(feats, dim=1)

    def forward(self, image: torch.Tensor, z_ms: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.band_features(image)  # (B,3,F)
        kv = feats if self.cfg.kv_mode == "band_tokens" else feats.flatten(1).unsqueeze(1)
        attended = self.attn(z_ms.unsqueeze(1), kv).squeeze(1)  # Eq. 8
        z = self.norm(z_ms + attended)
        return {"z_ddfm": z, "band_features": feats}
