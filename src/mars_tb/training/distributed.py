"""Minimal torch.distributed helpers (launch with ``torchrun``).

NOT EXERCISED IN CI: this environment had no multi-GPU hardware. The single-process code path
(world_size == 1) is what the tests cover; see reporting/reproducibility_status.md.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def world_info() -> tuple[int, int, int]:
    """(rank, world_size, local_rank) from the torchrun environment; (0, 1, 0) otherwise."""
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


def setup() -> tuple[int, int, torch.device]:
    rank, world, local = world_info()
    if world > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    return rank, world, device


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return world_info()[0] == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
