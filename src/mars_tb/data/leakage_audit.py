"""Leakage audit: exact and perceptual near-duplicates, and cross-partition overlap.

Verdicts: OVERLAP_FOUND (exact duplicate across partitions), NEAR_DUPLICATES_FOUND_REVIEW (both hashes
agree), SINGLE_HASH_CANDIDATES_FOR_REVIEW (one hash only; informational), NO_OVERLAP_FOUND[...].

Exact duplicates are found by file SHA-256 and by SHA-256 of the decoded 8-bit pixels (so the same
radiograph re-encoded as PNG vs JPEG is caught only by the perceptual pass, while a re-saved PNG
is caught by the pixel hash). Near-duplicates use 64-bit dHash and pHash with a Hamming radius;
candidate pairs come from 4x16-bit bucketing (pigeonhole: Hamming <= 3 forces one identical
16-bit chunk), then are verified exactly.

An audit can only demonstrate overlap, never its absence: patient-level separation additionally
needs real patient identifiers, which this module reports as missing rather than inventing.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from mars_tb.data.preprocessing import read_image, to_uint8
from mars_tb.utils.io import safe_join, sha256_file


def dhash(img_u8: np.ndarray) -> int:
    small = cv2.resize(img_u8, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def phash(img_u8: np.ndarray) -> int:
    small = cv2.resize(img_u8, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    med = np.median(dct[1:])  # exclude the DC term
    bits = dct > med
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def near_duplicate_pairs(hashes: dict[str, int], max_distance: int = 3) -> list[tuple[str, str, int]]:
    """All pairs of keys whose 64-bit hashes differ in <= max_distance bits (<=3 guaranteed exact)."""
    if max_distance > 3:
        raise ValueError("bucketed search is only exhaustive for max_distance <= 3")
    keys = list(hashes)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        h = hashes[k]
        for chunk in range(4):
            buckets[(chunk, (h >> (16 * chunk)) & 0xFFFF)].append(i)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[str, str, int]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                seen.add(pair)
                d = hamming(hashes[keys[pair[0]]], hashes[keys[pair[1]]])
                if d <= max_distance:
                    out.append((keys[pair[0]], keys[pair[1]], d))
    return sorted(out)


def _groups(values: dict[str, str]) -> list[list[str]]:
    inv: dict[str, list[str]] = defaultdict(list)
    for k, v in values.items():
        inv[v].append(k)
    return [sorted(g) for g in inv.values() if len(g) > 1]


def audit_manifest(
    manifest: pd.DataFrame,
    root: str | Path,
    max_hamming: int = 3,
    compute_perceptual: bool = True,
) -> dict[str, Any]:
    """Audit images listed in ``manifest`` (needs image_id, path, split)."""
    for col in ("image_id", "path", "split"):
        if col not in manifest.columns:
            raise ValueError(f"manifest lacks {col!r}")
    split_of = dict(zip(manifest["image_id"], manifest["split"], strict=True))
    file_hash: dict[str, str] = {}
    pixel_hash: dict[str, str] = {}
    dh: dict[str, int] = {}
    ph: dict[str, int] = {}
    unreadable: list[str] = []
    for row in manifest.itertuples():
        image_id = str(row.image_id)
        path = safe_join(root, str(row.path))
        try:
            file_hash[image_id] = sha256_file(path)
            u8 = to_uint8(read_image(path))
        except (OSError, ValueError, FileNotFoundError):
            unreadable.append(image_id)
            continue
        pixel_hash[image_id] = hashlib.sha256(u8.tobytes() + str(u8.shape).encode()).hexdigest()
        if compute_perceptual:
            dh[image_id] = dhash(u8)
            ph[image_id] = phash(u8)

    def crossing(ids: list[str]) -> bool:
        return len({split_of[i] for i in ids}) > 1

    exact = _groups(file_hash) + [g for g in _groups(pixel_hash) if g not in _groups(file_hash)]
    near: list[tuple[str, str, int, str]] = []
    if compute_perceptual:
        for name, table in (("dhash", dh), ("phash", ph)):
            near += [(a, b, d, name) for a, b, d in near_duplicate_pairs(table, max_hamming)]
    cross_exact = [g for g in exact if crossing(g)]
    names: dict[tuple[str, str], set[str]] = defaultdict(set)
    for a, b, _d, name in near:
        names[(a, b)].add(name)
    cross_near = [t for t in near if split_of[t[0]] != split_of[t[1]]]
    # A pair flagged by BOTH independent hashes is a confirmed near-duplicate; a pair flagged by one
    # hash only is a candidate for manual review (chest radiographs are structurally similar, so a
    # single 64-bit hash produces coincidental matches).
    confirmed = sorted({(a, b) for (a, b), n in names.items() if len(n) == 2 and split_of[a] != split_of[b]})
    has_patient = "patient_id" in manifest.columns and manifest["patient_id"].notna().any()
    if cross_exact:
        verdict = "OVERLAP_FOUND"
    elif confirmed:
        verdict = "NEAR_DUPLICATES_FOUND_REVIEW"
    elif cross_near:
        verdict = "SINGLE_HASH_CANDIDATES_FOR_REVIEW"
    else:
        verdict = "NO_OVERLAP_FOUND" if has_patient else "NO_OVERLAP_FOUND_BUT_PATIENT_LEVEL_UNVERIFIED"
    return {
        "n_images": len(manifest),
        "unreadable": unreadable,
        "exact_duplicate_groups": exact,
        "near_duplicate_pairs": near,
        "cross_partition_exact": cross_exact,
        "cross_partition_near": cross_near,
        "cross_partition_near_confirmed": confirmed,
        "patient_ids_available": bool(has_patient),
        "max_hamming": max_hamming,
        "verdict": verdict,
    }


FAILING_VERDICTS = frozenset({"OVERLAP_FOUND", "NEAR_DUPLICATES_FOUND_REVIEW"})
