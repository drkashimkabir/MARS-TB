"""Diagnostic-category -> binary TB label mapping (TBX11K).

TBX11K image-level classes: healthy, sick-but-non-TB, active TB, latent TB, active+latent TB, and
"uncertain" TB. The manuscript's binary endpoint is TB-positive (active, latent, both) versus
TB-negative (healthy + sick non-TB), with the 10 uncertain images excluded.

Filename prefixes (``h``/``s``/``tb``) identify only the coarse class. A ``tb`` prefix does NOT tell
you active vs latent vs uncertain; that needs the dataset's annotation files. This module refuses
to guess: :func:`category_from_filename` returns ``"tb_unspecified"`` for TB-prefixed names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

CATEGORIES = (
    "healthy",
    "sick_non_tb",
    "active_tb",
    "latent_tb",
    "active_and_latent_tb",
    "uncertain_tb",
    "tb_unspecified",
)
POSITIVE = frozenset({"active_tb", "latent_tb", "active_and_latent_tb", "tb_unspecified"})
NEGATIVE = frozenset({"healthy", "sick_non_tb"})
EXCLUDED = frozenset({"uncertain_tb"})

PATTERN_CLASSES = ("cavitary", "infiltrative", "miliary", "pleural")
SEVERITY_CLASSES = ("none", "mild", "moderate", "severe")
MISSING_AUX = -1

_PREFIX = re.compile(r"^(tb|h|s)\d+", re.IGNORECASE)


def category_from_filename(name: str) -> str:
    m = _PREFIX.match(name.split("/")[-1].split("\\")[-1])
    if not m:
        raise ValueError(f"filename {name!r} does not start with a TBX11K prefix (h|s|tb + digits)")
    return {"h": "healthy", "s": "sick_non_tb", "tb": "tb_unspecified"}[m.group(1).lower()]


def to_binary(category: str) -> int | None:
    """1 = TB-positive, 0 = TB-negative, None = excluded from the binary cohort."""
    if category in POSITIVE:
        return 1
    if category in NEGATIVE:
        return 0
    if category in EXCLUDED:
        return None
    raise ValueError(f"unknown diagnostic category {category!r}; known: {CATEGORIES}")


def encode_aux(values: Iterable[str | None], classes: tuple[str, ...]) -> list[int]:
    """Map verified auxiliary labels to indices; None/''/NaN -> -1 (ignored by the loss)."""
    out: list[int] = []
    for v in values:
        if v is None or (isinstance(v, float) and v != v) or str(v).strip() == "":
            out.append(MISSING_AUX)
        elif v in classes:
            out.append(classes.index(v))
        else:
            raise ValueError(f"unknown auxiliary label {v!r}; expected one of {classes}")
    return out
