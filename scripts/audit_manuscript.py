#!/usr/bin/env python
"""Recompute the arithmetic of the manuscript's reported numbers and write the consistency audit.

Reads reporting/manuscript_reported_values.json (a transcription), writes
reporting/manuscript_consistency_audit.md and .json. Exit code 1 if any FAIL is present, so the
audit can gate a release; use --allow-fail to always exit 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mars_tb.audit import run_audit, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--values", type=Path, default=ROOT / "reporting" / "manuscript_reported_values.json")
    ap.add_argument("--out-md", type=Path, default=ROOT / "reporting" / "manuscript_consistency_audit.md")
    ap.add_argument("--out-json", type=Path, default=ROOT / "reporting" / "manuscript_consistency_audit.json")
    ap.add_argument("--allow-fail", action="store_true")
    args = ap.parse_args(argv)
    values = json.loads(args.values.read_text(encoding="utf-8"))
    findings = run_audit(values)
    args.out_md.write_text(to_markdown(findings), encoding="utf-8")
    args.out_json.write_text(json.dumps([f.to_dict() for f in findings], indent=2), encoding="utf-8")
    counts = {s: sum(f.status == s for f in findings) for s in ("PASS", "FAIL", "FLAG", "INFO")}
    print(counts)
    for f in findings:
        if f.status in {"FAIL", "FLAG"}:
            print(f"[{f.status}] {f.id}: {f.title}")
    return 0 if (args.allow_fail or counts["FAIL"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
