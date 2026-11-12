#!/usr/bin/env python
"""Measure parameters, FLOPs, latency, throughput and memory. Only what actually runs is reported.

Definitions (stated so numbers are comparable):
  * parameters: sum of numel over all / trainable parameters.
  * FLOPs: torch.utils.flop_counter.FlopCounterMode on one forward pass (batch 1). It counts
    matmul/conv/attention operations at 2 x MACs and ignores elementwise ops, so it is NOT the same
    convention as papers that report MACs as "FLOPs".
  * latency: mean wall-clock of N forward passes at batch 1 after warm-up, torch.no_grad, with
    device synchronisation (CUDA) before reading the clock.
The manuscript's numbers (18.3 M trainable, 104.6 M total, 116.3 G FLOPs, 167 ms on an A100 40 GB)
are printed as REFERENCE VALUES, never as measurements of this run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from mars_tb.config import ModelConfig, deep_merge, from_dict, load_yaml, validate_model_config
from mars_tb.models.ablations import get_ablation
from mars_tb.models.baselines import build_baseline
from mars_tb.models.mars_tb import MARSTB, count_parameters
from mars_tb.training.reproducibility import environment_record

MANUSCRIPT_REFERENCE = {
    "trainable_params_M": 18.3,
    "total_params_M": 104.6,
    "flops_G": 116.3,
    "latency_ms_A100_40GB": 167,
    "status": "manuscript-reported, NOT measured here",
}
DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "model" / "mars_tb.yaml"


def measure_flops(model: torch.nn.Module, x: torch.Tensor) -> float:
    # FlopCounterMode's module tracker needs a grad_fn on parameters, which torch.no_grad() removes;
    # freezing the parameters temporarily avoids building an autograd graph without that problem.
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        counter = FlopCounterMode(display=False)
        # The fused SDPA kernels are not all registered with the counter (CPU flash attention is
        # not); the math backend decomposes attention into matmuls so QK^T and AV are counted.
        with sdpa_kernel(SDPBackend.MATH), counter:
            model(x)
        return float(counter.get_total_flops())
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def measure_latency(model: torch.nn.Module, x: torch.Tensor, warmup: int, trials: int) -> dict[str, float]:
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()
        for _ in range(trials):
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return {
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std(unbiased=True)) if len(times) > 1 else 0.0,
        "median_ms": float(t.median()),
        "trials": trials,
        "warmup": warmup,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--ablation", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--checkpoint", type=Path, help="to report the checkpoint file size")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--throughput-batch", type=int, default=8)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    dev = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if args.baseline:
        model: torch.nn.Module = build_baseline(args.baseline)
        size = 512
    else:
        raw = load_yaml(args.model_config)
        if args.ablation:
            raw = deep_merge(raw, dict(get_ablation(args.ablation)))
        cfg = from_dict(ModelConfig, raw)
        validate_model_config(cfg)
        model, size = MARSTB(cfg), cfg.backbone.img_size
    model.to(dev).eval()
    x1 = torch.randn(1, 3, size, size, device=dev)
    report: dict = {
        "model": args.baseline or args.ablation or "MARS-TB",
        "input": [1, 3, size, size],
        "parameters": count_parameters(model),
        "flops_G_counter_2xMAC": measure_flops(model, x1) / 1e9,
        "latency_batch1": measure_latency(model, x1, args.warmup, args.trials),
        "environment": environment_record(),
        "manuscript_reference": MANUSCRIPT_REFERENCE,
    }
    xb = torch.randn(args.throughput_batch, 3, size, size, device=dev)
    lat = measure_latency(model, xb, max(1, args.warmup // 10), max(5, args.trials // 20))
    report["throughput_images_per_s"] = args.throughput_batch / (lat["mean_ms"] / 1000)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(xb)
        report["peak_gpu_memory_MB"] = torch.cuda.max_memory_allocated() / 2**20
    else:
        report["peak_gpu_memory_MB"] = None
    if args.checkpoint:
        report["checkpoint_size_MB"] = args.checkpoint.stat().st_size / 2**20
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
