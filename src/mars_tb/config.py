"""Typed, immutable configuration objects with YAML loading and validation."""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar, Union

import yaml

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class LoRAConfig:
    enabled: bool = True
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    targets: tuple[str, ...] = ("q", "v")
    init_std: float = 0.01

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


@dataclasses.dataclass(frozen=True)
class BackboneConfig:
    img_size: int = 512
    patch_size: int = 16
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    layers: tuple[int, ...] = (4, 8, 12)
    freeze_backbone: bool = True
    # none | timm_state_dict. "none" = random init (the default; nothing is downloaded).
    pretrained: str = "none"
    pretrained_path: str | None = None
    pretrained_grid: int | None = None
    gradient_checkpointing: bool = False


@dataclasses.dataclass(frozen=True)
class TextConfig:
    # learned = trainable prompt embeddings (no language prior); file = precomputed frozen
    # text-encoder embeddings (see scripts/precompute_text_embeddings.py).
    source: str = "learned"
    path: str | None = None
    prompts: str = "builtin"


@dataclasses.dataclass(frozen=True)
class MSVLEConfig:
    attn_dim: int = 128
    heads: int = 8
    pool_kernels: tuple[int, int] = (2, 4)  # scale 2 / scale 3 spatial pooling kernels


@dataclasses.dataclass(frozen=True)
class DDFMConfig:
    band_edges: tuple[int, int] = (16, 64)
    geometry: str = "radial"  # radial | square
    band_input: str = "ifft"  # ifft | log_magnitude
    cnn_channels: tuple[int, int, int] = (64, 128, 256)
    feat_dim: int = 256
    attn_dim: int = 256
    heads: int = 8
    kv_mode: str = "band_tokens"  # band_tokens | concat_single (manuscript-literal, degenerate)


@dataclasses.dataclass(frozen=True)
class AGRNConfig:
    graph_spec: str = "builtin"
    heads: int = 8
    hidden: int = 512
    layers: int = 3
    dropout: float = 0.0


@dataclasses.dataclass(frozen=True)
class PRMConfig:
    use_gelu: bool = False
    residual: bool = False


@dataclasses.dataclass(frozen=True)
class HeadConfig:
    type: str = "evidential"  # evidential | softmax_ce
    pattern_classes: int = 4
    severity_classes: int = 4


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    backbone: BackboneConfig = dataclasses.field(default_factory=BackboneConfig)
    lora: LoRAConfig = dataclasses.field(default_factory=LoRAConfig)
    text: TextConfig = dataclasses.field(default_factory=TextConfig)
    msvle: MSVLEConfig = dataclasses.field(default_factory=MSVLEConfig)
    ddfm: DDFMConfig = dataclasses.field(default_factory=DDFMConfig)
    agrn: AGRNConfig = dataclasses.field(default_factory=AGRNConfig)
    prm: PRMConfig = dataclasses.field(default_factory=PRMConfig)
    head: HeadConfig = dataclasses.field(default_factory=HeadConfig)
    use_ddfm: bool = True
    use_agrn: bool = True
    use_prm: bool = True


@dataclasses.dataclass(frozen=True)
class LossConfig:
    weights: tuple[float, float, float] = (1.0, 0.3, 0.3)
    kl_anneal_epochs: int = 10
    nll_type: str = "log"  # log | digamma | mse


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    optimizer: str = "adamw"
    lr_lora: float = 1e-4
    lr_new: float = 1e-5
    lr_backbone: float = 0.0
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_epochs: int = 5
    epochs: int = 100
    min_lr: float = 1e-7
    batch_size_per_gpu: int = 16
    grad_accum_steps: int = 2
    grad_clip: float = 1.0
    amp: bool = True
    early_stopping_patience: int = 15
    seed: int = 42
    num_workers: int = 4
    # neg_over_pos = n_neg / n_pos derived from the *training labels*; or "fixed:<value>".
    sampler_weights: str = "neg_over_pos"
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)


def _unwrap_optional(tp: Any) -> Any:
    origin = typing.get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def from_dict(cls: type[T], data: Mapping[str, Any] | None) -> T:
    """Recursively build a (frozen) dataclass from a mapping, rejecting unknown keys."""
    if data is None:
        return cls()
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        tp = _unwrap_optional(hints[key])
        if dataclasses.is_dataclass(tp) and isinstance(tp, type):
            kwargs[key] = from_dict(tp, value)
        elif typing.get_origin(tp) is tuple and isinstance(value, list | tuple):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_model_config(*paths: str | Path, overrides: Mapping[str, Any] | None = None) -> ModelConfig:
    merged: dict[str, Any] = {}
    for p in paths:
        merged = deep_merge(merged, load_yaml(p))
    if overrides:
        merged = deep_merge(merged, overrides)
    cfg = from_dict(ModelConfig, merged)
    validate_model_config(cfg)
    return cfg


def load_train_config(*paths: str | Path) -> TrainConfig:
    merged: dict[str, Any] = {}
    for p in paths:
        merged = deep_merge(merged, load_yaml(p))
    cfg = from_dict(TrainConfig, merged)
    validate_train_config(cfg)
    return cfg


def validate_model_config(cfg: ModelConfig) -> None:
    b = cfg.backbone
    if b.img_size % b.patch_size:
        raise ValueError("backbone.img_size must be divisible by backbone.patch_size")
    grid = b.img_size // b.patch_size
    k2, k3 = cfg.msvle.pool_kernels
    if grid % k2 or grid % k3:
        raise ValueError(f"token grid {grid} must be divisible by pool kernels {cfg.msvle.pool_kernels}")
    if b.dim % b.heads:
        raise ValueError("backbone.dim must be divisible by backbone.heads")
    if any(layer < 1 or layer > b.depth for layer in b.layers) or len(b.layers) != 3:
        raise ValueError("backbone.layers must be three indices in [1, depth]")
    if cfg.lora.enabled and not cfg.backbone.freeze_backbone:
        raise ValueError("LoRA with an unfrozen backbone is not a defined configuration")
    if cfg.text.source not in {"learned", "file"}:
        raise ValueError("text.source must be 'learned' or 'file'")
    if cfg.text.source == "file" and not cfg.text.path:
        raise ValueError("text.source='file' requires text.path")
    if cfg.ddfm.geometry not in {"radial", "square"}:
        raise ValueError("ddfm.geometry must be radial|square")
    if cfg.ddfm.kv_mode not in {"band_tokens", "concat_single"}:
        raise ValueError("ddfm.kv_mode must be band_tokens|concat_single")
    if cfg.ddfm.band_input not in {"ifft", "log_magnitude"}:
        raise ValueError("ddfm.band_input must be ifft|log_magnitude")
    if cfg.head.type not in {"evidential", "softmax_ce"}:
        raise ValueError("head.type must be evidential|softmax_ce")
    lo, hi = cfg.ddfm.band_edges
    if not 0 < lo < hi:
        raise ValueError("ddfm.band_edges must satisfy 0 < low < high")


def validate_train_config(cfg: TrainConfig) -> None:
    if cfg.grad_accum_steps < 1 or cfg.batch_size_per_gpu < 1:
        raise ValueError("batch size and accumulation steps must be >= 1")
    if cfg.warmup_epochs >= cfg.epochs:
        raise ValueError("warmup_epochs must be < epochs")
    if cfg.sampler_weights != "neg_over_pos" and not cfg.sampler_weights.startswith("fixed:"):
        raise ValueError("sampler_weights must be 'neg_over_pos' or 'fixed:<float>'")


def effective_batch_size(cfg: TrainConfig, world_size: int = 1) -> int:
    """Global optimiser-step batch size = per-GPU batch x accumulation x number of GPUs.

    The manuscript quotes 32 for a 2-GPU run (16 x 2); that equals the *single-process* value.
    With two data-parallel processes the global value is 64 (known_discrepancies.md, H).
    """
    return cfg.batch_size_per_gpu * cfg.grad_accum_steps * world_size
