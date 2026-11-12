import json
from pathlib import Path

import pytest

from mars_tb.audit import hanley_mcneil_se, run_audit, to_markdown

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def values():
    return json.loads((ROOT / "reporting" / "manuscript_reported_values.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def findings(values):
    return {f.id: f for f in run_audit(values)}


def test_transcription_passes_the_sums_that_expose_typing_errors(findings):
    # If one of these fails, the JSON transcription (not the manuscript) is probably wrong.
    for fid in ("C01", "C02", "C06", "C11", "C12", "C15", "C17", "C19", "C22", "C24"):
        assert findings[fid].status == "PASS", (fid, findings[fid].detail)


def test_known_contradictions_are_reported_as_failures(findings):
    for fid in ("C03", "C04", "C13", "C25"):
        assert findings[fid].status == "FAIL"


def test_table4_intervals_identify_the_older_2221_image_test_set(findings):
    assert "404/1817" in findings["C07"].detail and findings["C07"].status == "FAIL"


def test_table8_template_pattern_is_detected(findings):
    assert findings["C20"].status == "FLAG" and findings["C21"].status == "FLAG"


def test_s6_grid_signature_detected(findings):
    assert findings["C23"].status == "FLAG"


def test_stale_sampler_weight_documented(findings):
    assert "4.4998" in findings["C05"].detail and "8.403" in findings["C05"].detail


def test_hanley_mcneil_known_value():
    # AUC 0.5 with equal classes of 100: SE = sqrt((0.25 + 99*(1/3-0.25)*2)/10000) ~ 0.0408
    assert hanley_mcneil_se(0.5, 100, 100) == pytest.approx(0.0408, abs=2e-4)


def test_markdown_lists_every_finding(findings):
    md = to_markdown(list(findings.values()))
    assert all(f"### {fid}" in md for fid in findings) and "Nothing here asserts intent" in md


def test_committed_audit_is_up_to_date(values):
    committed = (ROOT / "reporting" / "manuscript_consistency_audit.md").read_text(encoding="utf-8")
    assert committed == to_markdown(run_audit(values)), "re-run scripts/audit_manuscript.py"
