#!/usr/bin/env python
"""Check that the repository's claims about artifacts match what is actually on disk.

Checks (exit 1 on any failure):
  * reporting/artifact_inventory.csv: every row with status 'available' points at an existing file whose
    SHA-256 matches; 'unavailable' rows must not name a file.
  * no checkpoint / weight / DICOM files are present in the tree.
  * CLAIM (44 items) and TRIPOD+AI (52 sub-items) checklists are complete and have no empty answers.
  * relative Markdown links in README.md and docs/ resolve to files.
  * the committed manuscript audit is current.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".dcm", ".dicom", ".pkl", ".npz", ".h5"}
SKIP = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "mars_tb.egg-info",
    ".venv",
    "build",
    "dist",
}
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inventory(problems: list[str]) -> None:
    path = ROOT / "reporting" / "artifact_inventory.csv"
    if not path.is_file():
        problems.append("reporting/artifact_inventory.csv is missing")
        return
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            status, file, digest = row["status"], row.get("path", ""), row.get("sha256", "")
            if status == "available":
                target = ROOT / file
                if not file or not target.is_file():
                    problems.append(f"inventory '{row['artifact']}' is 'available' but {file!r} does not exist")
                elif digest and sha256(target) != digest:
                    problems.append(f"inventory '{row['artifact']}': sha256 mismatch for {file}")
                elif not digest and row["category"].startswith("Verified"):
                    problems.append(f"inventory '{row['artifact']}': released original artifact has no sha256")
            elif status in {"unavailable", "not_verified", "not_implemented"} and file:
                problems.append(f"inventory '{row['artifact']}' is {status} but names a file")
            elif status not in {"available", "unavailable", "not_verified", "not_implemented"}:
                problems.append(f"inventory '{row['artifact']}': unknown status {status!r}")


def check_banned_files(problems: list[str]) -> None:
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP) and p.suffix.lower() in BANNED_SUFFIXES:
            problems.append(f"binary artifact present: {p.relative_to(ROOT)}")


def check_checklists(problems: list[str]) -> None:
    for name, expected in (("CLAIM_2024_checklist.csv", 44), ("TRIPOD_AI_2024_checklist.csv", 52)):
        path = ROOT / "reporting" / name
        if not path.is_file():
            problems.append(f"reporting/{name} is missing")
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != expected:
            problems.append(f"{name}: expected {expected} items, found {len(rows)}")
        for r in rows:
            if r["response"] not in {"Yes", "No", "Not applicable", "Partial"}:
                problems.append(f"{name}: item {r['item']} has invalid response {r['response']!r}")
            if r["response"] in {"No", "Partial"} and not r["corrective_action"].strip():
                problems.append(f"{name}: item {r['item']} is {r['response']} without a corrective action")
            if r["response"] == "Yes" and not r["evidence"].strip():
                problems.append(f"{name}: item {r['item']} is Yes without evidence")


def check_links(problems: list[str]) -> None:
    files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted((ROOT / "reporting").glob("*.md"))]
    for f in files:
        if not f.is_file():
            continue
        for target in LINK.findall(f.read_text(encoding="utf-8")):
            if re.match(r"[a-z]+://|mailto:", target):
                continue
            if not (f.parent / target).resolve().exists():
                problems.append(f"{f.relative_to(ROOT)}: broken link {target!r}")


def check_audit_current(problems: list[str]) -> None:
    import json

    from mars_tb.audit import run_audit, to_markdown

    values = json.loads((ROOT / "reporting" / "manuscript_reported_values.json").read_text(encoding="utf-8"))
    audit = ROOT / "reporting" / "manuscript_consistency_audit.md"
    if not audit.is_file() or audit.read_text(encoding="utf-8") != to_markdown(run_audit(values)):
        problems.append("reporting/manuscript_consistency_audit.md is stale; run scripts/audit_manuscript.py")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-links", action="store_true")
    args = ap.parse_args(argv)
    problems: list[str] = []
    check_inventory(problems)
    check_banned_files(problems)
    check_checklists(problems)
    check_audit_current(problems)
    if not args.skip_links:
        check_links(problems)
    for p in problems:
        print(f"FAIL: {p}")
    print("artifact validation " + ("FAILED" if problems else "passed"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
