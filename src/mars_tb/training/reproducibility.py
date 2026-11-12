"""Environment, version and code-revision capture for run provenance."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

_TRACKED = (
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "pandas",
    "opencv-python-headless",
    "pillow",
    "pyyaml",
    "pydicom",
    "jsonschema",
    "matplotlib",
)


def _git() -> str | None:
    return shutil.which("git")


def git_commit(repo: str | Path | None = None) -> str | None:
    exe = _git()
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            [exe, "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_is_dirty(repo: str | Path | None = None) -> bool | None:
    exe = _git()
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            [exe, "status", "--porcelain"], cwd=repo, capture_output=True, text=True, timeout=10, check=True
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_record() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_count": torch.cuda.device_count() if cuda else 0,
        "libraries": library_versions(),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
    }
