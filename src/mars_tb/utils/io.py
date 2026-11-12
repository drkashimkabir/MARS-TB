"""Safe file IO: path confinement, hashing, atomic JSON writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


class UnsafePathError(ValueError):
    """Raised when a manifest or CLI path escapes its permitted root."""


def safe_join(root: str | Path, relative: str | Path) -> Path:
    """Join ``relative`` onto ``root`` and refuse absolute paths or traversal outside ``root``."""
    rel = Path(str(relative).replace("\\", "/"))
    if rel.is_absolute() or (rel.drive and rel.drive != ""):
        raise UnsafePathError(f"absolute path not allowed: {relative!s}")
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / rel).resolve()
    if root_resolved != candidate and root_resolved not in candidate.parents:
        raise UnsafePathError(f"path escapes root: {relative!s}")
    return candidate


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: str | Path, obj: Any) -> None:
    """Write JSON via a temp file + rename so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
