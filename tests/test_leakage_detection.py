import cv2
import numpy as np
import pandas as pd

from mars_tb.data.leakage_audit import audit_manifest, dhash, hamming, near_duplicate_pairs, phash


def _scene(seed: int, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.uint8)
    for _ in range(12):
        cv2.circle(
            img,
            tuple(int(v) for v in rng.integers(10, size - 10, 2)),
            int(rng.integers(5, 25)),
            int(rng.integers(60, 255)),
            -1,
        )
    return cv2.GaussianBlur(img, (0, 0), 2)


def _manifest(tmp_path, images: dict[str, np.ndarray], splits: dict[str, str]):
    rows = []
    for name, arr in images.items():
        cv2.imwrite(str(tmp_path / f"{name}.png"), arr)
        rows.append({"image_id": name, "path": f"{name}.png", "split": splits[name]})
    return pd.DataFrame(rows)


def test_bucketed_search_equals_brute_force():
    rng = np.random.default_rng(0)
    hashes = {f"k{i}": int(rng.integers(0, 2**63)) for i in range(150)}
    base = hashes["k0"]
    for j, bits in enumerate([[], [3], [5, 40], [1, 2, 63 - 1]], start=1):  # distances 0..3
        h = base
        for b in bits:
            h ^= 1 << b
        hashes[f"planted{j}"] = h
    fast = {(a, b) for a, b, _ in near_duplicate_pairs(hashes, 3)}
    keys = list(hashes)
    brute = {
        (keys[i], keys[j])
        for i in range(len(keys))
        for j in range(i + 1, len(keys))
        if hamming(hashes[keys[i]], hashes[keys[j]]) <= 3
    }
    norm = lambda s: {tuple(sorted(p)) for p in s}  # noqa: E731
    assert norm(fast) == norm(brute)
    assert len(norm(fast)) >= 4


def test_exact_duplicate_across_partitions_is_flagged(tmp_path):
    a = _scene(1)
    df = _manifest(
        tmp_path, {"orig": a, "copy": a.copy(), "other": _scene(2)}, {"orig": "train", "copy": "test", "other": "val"}
    )
    rep = audit_manifest(df, tmp_path)
    assert rep["cross_partition_exact"] == [["copy", "orig"]]
    assert rep["verdict"] == "OVERLAP_FOUND"


def test_recompressed_and_resized_copy_is_caught_as_near_duplicate(tmp_path):
    a = _scene(3)
    jpg = cv2.imdecode(cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(cv2.resize(a, (96, 96)), (128, 128))
    df = _manifest(tmp_path, {"a": a, "jpg": jpg, "small": resized}, {"a": "train", "jpg": "test", "small": "val"})
    rep = audit_manifest(df, tmp_path, max_hamming=3)
    assert ("a", "jpg") in {tuple(sorted(p)) for p in rep["cross_partition_near_confirmed"]}
    assert rep["verdict"] in {"NEAR_DUPLICATES_FOUND_REVIEW", "OVERLAP_FOUND"}


def test_unrelated_images_are_not_flagged_and_verdict_states_patient_caveat(tmp_path):
    imgs = {f"i{k}": _scene(100 + k) for k in range(6)}
    df = _manifest(tmp_path, imgs, {k: ("train", "val", "test")[i % 3] for i, k in enumerate(imgs)})
    rep = audit_manifest(df, tmp_path)
    assert rep["cross_partition_exact"] == [] and rep["cross_partition_near_confirmed"] == []
    assert rep["verdict"] in {"NO_OVERLAP_FOUND_BUT_PATIENT_LEVEL_UNVERIFIED", "SINGLE_HASH_CANDIDATES_FOR_REVIEW"}
    assert rep["patient_ids_available"] is False


def test_exact_duplicate_verdict_fails_and_single_hash_coincidence_does_not(tmp_path):
    from mars_tb.data.leakage_audit import FAILING_VERDICTS

    a = _scene(11)
    df = _manifest(tmp_path, {"p": a, "q": a.copy()}, {"p": "train", "q": "test"})
    assert audit_manifest(df, tmp_path)["verdict"] in FAILING_VERDICTS
    assert "SINGLE_HASH_CANDIDATES_FOR_REVIEW" not in FAILING_VERDICTS


def test_unreadable_files_are_reported_not_ignored(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"junk")
    df = pd.DataFrame([{"image_id": "bad", "path": "bad.png", "split": "train"}])
    assert audit_manifest(df, tmp_path)["unreadable"] == ["bad"]


def test_hashes_are_stable_and_distinguish_images():
    a, b = _scene(5), _scene(6)
    assert dhash(a) == dhash(a.copy()) and phash(a) == phash(a.copy())
    assert hamming(phash(a), phash(b)) > 3
