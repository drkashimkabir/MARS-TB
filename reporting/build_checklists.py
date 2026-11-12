#!/usr/bin/env python
"""Build CLAIM_2024_checklist.csv and TRIPOD_AI_2024_checklist.csv.

Item wording comes from official_checklist_text/ (snapshots of the published checklists). The
RESPONSES below are an assessment of the MANUSCRIPT AS SUBMITTED (TB Paper_finial-1_up5.pdf) plus
what this repository adds. They are the preparers' judgement and must be reviewed by the authors.
Rule applied: an item is 'Yes' only if the manuscript text itself reports it; explanatory
documentation in this repository does not turn a missing manuscript item into a completed one.

status: complete -> response Yes | partial/missing -> response No | n_a -> Not applicable
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = "TB Paper_finial-1_up5.pdf"

# item: (status, manuscript location, repository documentation, evidence, missing information, corrective action)
CLAIM: dict[int, tuple[str, str, str, str, str, str]] = {
    1: (
        "partial",
        "Title, Abstract, Keywords",
        "README.md",
        "Keywords list 'Deep Learning'; abstract names the framework components.",
        "Title/abstract do not state the technology category (deep learning).",
        "Add 'deep learning' to the title or abstract.",
    ),
    2: (
        "partial",
        "Abstract",
        "-",
        "Abstract summarises design, methods, results, conclusion.",
        "Single unstructured paragraph; AUCs given without confidence intervals; test-set size not stated.",
        "Restructure the abstract and add CIs and N.",
    ),
    3: (
        "partial",
        "Sec. 1",
        "docs/limitations.md",
        "Intro describes screening/triage role and the gap.",
        "Current practice and workflow it would replace or support are not described.",
        "Add a paragraph on the current reading/triage workflow and the intended point of use.",
    ),
    4: (
        "partial",
        "Sec. 1 (last paragraph)",
        "-",
        "'we propose MARS-TB' addresses a stated gap.",
        "No explicit aims or hypotheses.",
        "State objectives and hypotheses explicitly.",
    ),
    5: ("complete", "Sec. 2.1, 2.3", "-", "'This retrospective study ...'.", "", ""),
    6: (
        "complete",
        "Sec. 2.3",
        "-",
        "Model creation and retrospective evaluation, 'automated binary discrimination'.",
        "",
        "",
    ),
    7: (
        "partial",
        "Sec. 2.1, Data Availability",
        "docs/data_provenance.md; configs/data/",
        "TBX11K (Kaggle link), PadChest (HUSJ Alicante), NIAID TB Portals (API, March 2025).",
        "PadChest-TB and TB Portals cohort selection is incompletely documented; no manifests released.",
        "Release verified selection manifests (image ids + provenance) for both external cohorts.",
    ),
    8: (
        "partial",
        "Sec. 2.1, 2.2",
        "docs/data_provenance.md",
        "10 uncertain TBX11K images excluded; quality filter (edge >=256 px, aspect 0.5-2.0).",
        "External-cohort inclusion/exclusion criteria are 'incompletely documented' (manuscript's words); filter exclusion counts not reported.",
        "Document criteria and report how many images each filter removed.",
    ),
    9: (
        "complete",
        "Sec. 2.2, Table 2",
        "docs/preprocessing.md; src/mars_tb/data/preprocessing.py",
        "Eight-step pipeline with parameters; implemented and unit-tested here.",
        "DICOM-to-16-bit-PNG detail is ambiguous (see known_discrepancies).",
        "",
    ),
    10: (
        "missing",
        "Sec. 2.1",
        "docs/data_provenance.md",
        "-",
        "How the 3,000 PadChest and 2,800 TB Portals images were chosen from the full collections is not reported.",
        "Report the selection procedure and the size of the source collections.",
    ),
    11: (
        "missing",
        "Data Availability",
        "docs/data_provenance.md",
        "States no patient-identifiable data were used.",
        "No description of de-identification by the dataset providers or of DICOM header handling.",
        "State provider de-identification and confirm no headers were retained.",
    ),
    12: (
        "partial",
        "Sec. 2.1.1, Sec. 4",
        "src/mars_tb/models/losses.py (missing-label masks)",
        "Missing/ambiguous auxiliary-label handling acknowledged as undocumented.",
        "No procedure for missing auxiliary labels or missing demographics.",
        "Document handling; repository masks unavailable auxiliary labels instead of imputing.",
    ),
    13: (
        "missing",
        "Sec. 4",
        "docs/limitations.md",
        "Manuscript states scanner, projection and equipment information were unavailable.",
        "Acquisition protocol not reported.",
        "Report whatever acquisition metadata the source datasets provide.",
    ),
    14: (
        "partial",
        "Sec. 2.1",
        "docs/data_provenance.md",
        "TBX11K categories described; PadChest negatives = radiologically normal.",
        "Source of the image-level labels for the whole custom cohort and for TB Portals negatives 'insufficiently documented'/'not established'.",
        "Document label provenance per image (manifest column label_provenance).",
    ),
    15: (
        "missing",
        "-",
        "docs/data_provenance.md",
        "-",
        "No rationale for using dataset labels / normal-report negatives as the reference standard.",
        "Add rationale and discuss non-microbiological labels.",
    ),
    16: (
        "missing",
        "Sec. 2.1, 2.1.1",
        "-",
        "-",
        "Annotator identity and qualifications not reported; auxiliary-label procedures undocumented.",
        "Report or remove the auxiliary tasks.",
    ),
    17: (
        "missing",
        "Sec. 2.1",
        "-",
        "-",
        "Test-set annotation procedure not described; official test labels were withheld.",
        "Document how the custom test labels were derived.",
    ),
    18: (
        "missing",
        "Sec. 2.1.1",
        "-",
        "'Inter-rater agreement were not reported'.",
        "No inter-/intra-rater data.",
        "Provide agreement statistics or state none exist.",
    ),
    19: (
        "partial",
        "Sec. 2.1, Table 1",
        "docs/data_provenance.md; scripts/generate_splits.py",
        "Custom 60/20/20 stratified by diagnostic category; counts in Table 1.",
        "Seed and exact assignment files not documented; strata not reproducible from counts alone.",
        "Publish the original split manifests (see reporting/artifact_inventory.csv).",
    ),
    20: (
        "missing",
        "Sec. 2.1, 4",
        "src/mars_tb/data/leakage_audit.py",
        "'patient-grouped partitioning were not documented' - split is image-level.",
        "Patient-level disjointness not established; TBX11K has no patient ids.",
        "Audit duplicates; state clearly that partitions are image-level.",
    ),
    21: (
        "missing",
        "-",
        "-",
        "-",
        "No sample-size justification for the test sets.",
        "Add a precision-based justification or state none was done.",
    ),
    22: (
        "partial",
        "Sec. 2.4, Figs. 1-5, Eqs. 1-12, Supp. Table 1",
        "docs/architecture.md; docs/manuscript_to_code_mapping.md",
        "Full architecture, dimensions, equations; re-implemented and tested here.",
        "Main text and Supp. Table 1 disagree (CrossScaleAttention, frequency projection); Eq. 8 is degenerate; BioViL-T is described as ViT-B/16 (it is not).",
        "Resolve discrepancies A-K in docs/known_discrepancies.md.",
    ),
    23: (
        "partial",
        "Sec. 2.2, 2.5",
        "pyproject.toml; requirements*.txt; reporting/reproducibility_status.md",
        "PyTorch 2.1.0, CUDA 12.1, OpenCV 4.8, torchvision 0.16.",
        "Other libraries, the LoRA implementation and the BioViL-T code revision are not versioned.",
        "List all versions from the original environment.",
    ),
    24: (
        "partial",
        "Sec. 2.4.1",
        "-",
        "LoRA A ~ N(0, 0.01^2), B = 0.",
        "Source/checkpoint of the pretrained backbone and initialisation of all new modules not stated.",
        "State pretrained weights source and the init of new layers.",
    ),
    25: (
        "partial",
        "Sec. 2.5, Table 3",
        "docs/training.md; configs/training/default.yaml",
        "Optimizer, LRs, schedule, batch, clipping, AMP, early stopping, hardware.",
        "Sampling weight 4.50 is stale for the revised cohort; global batch ambiguous (32 vs 64 on 2 GPUs).",
        "State the weight and batch size actually used.",
    ),
    26: (
        "complete",
        "Sec. 2.5",
        "src/mars_tb/training/trainer.py",
        "Early stopping on validation AUC; 'No test data were used for early stopping or model selection'.",
        "",
        "",
    ),
    27: ("n_a", "-", "-", "Single model per run; seeds are independent runs, not an ensemble.", "", ""),
    28: (
        "complete",
        "Sec. 3.1",
        "docs/evaluation.md",
        "AUC, sensitivity, specificity, MCC (+ operating-point thresholds in Supp. Table S7).",
        "",
        "",
    ),
    29: (
        "partial",
        "Sec. 3.1, 3.3, 3.5",
        "docs/statistics.md; reporting/manuscript_consistency_audit.md",
        "DeLong, Clopper-Pearson, Holm, 5-seed t-intervals.",
        "Audit finds Table 4 intervals computed on a different test set and narrower than expected; seed results unverified.",
        "Recompute all intervals from released prediction files.",
    ),
    30: (
        "partial",
        "Sec. 3.3, 3.5",
        "scripts/run_ablations.py",
        "Five seeds, ablations.",
        "No sensitivity to threshold, prevalence, or preprocessing choices.",
        "Add prevalence/threshold sensitivity.",
    ),
    31: (
        "partial",
        "Sec. 3.6, Figs. 7-8",
        "docs/explainability.md; src/mars_tb/explainability/",
        "Grad-CAM++ on six selected cases; limits acknowledged.",
        "No quantitative localisation, no failure cases, predictions/thresholds for the cases not shown.",
        "Run scripts/evaluate_localization.py and show TP/TN/FP/FN.",
    ),
    32: (
        "partial",
        "Sec. 3.1, Table 4",
        "reporting/manuscript_consistency_audit.md",
        "Internal test performance reported.",
        "Test-set size contradicts Table 1 (2,221 vs 2,238).",
        "Reconcile and recompute on one defined test set.",
    ),
    33: (
        "complete",
        "Sec. 3.2, Table 5",
        "docs/data_provenance.md",
        "Two external cohorts, never used for training or selection.",
        "Reference standards of the external cohorts are weak (acknowledged).",
        "",
    ),
    34: ("n_a", "Declarations", "-", "'Clinical Trial Number: Not applicable'.", "", ""),
    35: (
        "partial",
        "Sec. 2.1",
        "-",
        "11,200 -> 11,190 (10 excluded) for TBX11K.",
        "Flow for external cohorts and quality-filter exclusions not reported.",
        "Add a flow diagram.",
    ),
    36: (
        "missing",
        "Sec. 4",
        "-",
        "Manuscript: age, sex, HIV status, prior TB, drug resistance unavailable.",
        "Demographic/clinical characteristics not reported.",
        "Report what the sources provide; state absence.",
    ),
    37: (
        "partial",
        "Sec. 3.1-3.2",
        "docs/statistics.md",
        "Metrics with intervals on three data partitions.",
        "No subgroup performance.",
        "Add site/device subgroup analysis where metadata exist.",
    ),
    38: (
        "partial",
        "Sec. 3.1-3.2",
        "src/mars_tb/evaluation/",
        "AUC with CI, Se/Sp with CI.",
        "No PPV/NPV at the very different external prevalences (29.7 %, 66.0 %), no calibration.",
        "Report PPV/NPV and calibration per cohort.",
    ),
    39: (
        "missing",
        "Sec. 3.6, 4",
        "-",
        "'verified failure cases' listed as future work.",
        "No failure analysis.",
        "Analyse false positives/negatives.",
    ),
    40: ("complete", "Sec. 4", "docs/limitations.md", "Extensive limitations section.", "", ""),
    41: (
        "complete",
        "Sec. 4, 5",
        "docs/model_card.md",
        "Screening/triage support stated as proposed, not validated.",
        "",
        "",
    ),
    42: (
        "partial",
        "Supplementary tables",
        "docs/manuscript_to_code_mapping.md",
        "Supplementary tables S1-S8.",
        "No protocol; supplement cites S1 for category allocations but S1 is the parameter table.",
        "Release the missing allocation table and a protocol/technical appendix.",
    ),
    43: (
        "partial",
        "Data Availability",
        "README.md; reporting/editor_comment_10_response.md; reporting/artifact_inventory.csv",
        "TBX11K link given; this repository provides a manuscript-derived reference implementation.",
        "Original code, trained weights and exact split files are not available.",
        "Release them if they exist; otherwise state why not.",
    ),
    44: ("complete", "Declarations", "-", "'received no external funding'.", "", ""),
}

CLAIM_SECTION = {
    **{i: "Title and abstract" for i in (1, 2)},
    **{i: "Introduction" for i in (3, 4)},
    **{i: "Methods" for i in range(5, 35)},
    **{i: "Results" for i in range(35, 40)},
    **{i: "Discussion" for i in (40, 41)},
    **{i: "Other information" for i in (42, 43, 44)},
}

TRIPOD: dict[str, tuple[str, str, str, str, str, str]] = {
    "1": (
        "partial",
        "Title",
        "-",
        "Title identifies the task, not the study type.",
        "Does not say development/evaluation, population or outcome.",
        "Revise the title.",
    ),
    "2": (
        "partial",
        "Abstract",
        "-",
        "Key results present.",
        "Not in TRIPOD+AI-for-abstracts form (no sample sizes, CIs).",
        "Rewrite abstract per the TRIPOD+AI abstract checklist.",
    ),
    "3a": ("complete", "Sec. 1", "-", "Healthcare context and rationale with references.", "", ""),
    "3b": (
        "partial",
        "Sec. 1, 4",
        "docs/model_card.md",
        "Screening/triage support proposed.",
        "Target population, care pathway and users not defined.",
        "Define population, pathway, users.",
    ),
    "3c": ("missing", "-", "-", "-", "Health inequalities not discussed.", "Add or state not assessed."),
    "4": (
        "partial",
        "Sec. 1",
        "-",
        "Development and external validation implied.",
        "Objectives not stated explicitly.",
        "State objectives.",
    ),
    "5a": (
        "partial",
        "Sec. 2.1",
        "docs/data_provenance.md",
        "Three sources named.",
        "Representativeness of sources not discussed.",
        "Discuss representativeness.",
    ),
    "5b": (
        "missing",
        "-",
        "-",
        "TB Portals accessed March 2025 only.",
        "Dates of data collection not given.",
        "Report accrual dates per source.",
    ),
    "6a": (
        "partial",
        "Sec. 2.1",
        "-",
        "PadChest single hospital; TB Portals multi-institutional.",
        "TBX11K centres and numbers of centres not given.",
        "Report setting and centres.",
    ),
    "6b": (
        "partial",
        "Sec. 2.1, 2.2",
        "-",
        "Quality filter and exclusion of uncertain cases.",
        "External eligibility incompletely documented.",
        "Document eligibility.",
    ),
    "6c": (
        "missing",
        "Sec. 4",
        "-",
        "Treatment stage unavailable (acknowledged).",
        "Treatments and their handling not reported.",
        "State absence and impact.",
    ),
    "7": (
        "partial",
        "Sec. 2.2",
        "docs/preprocessing.md",
        "Detailed preprocessing/QC.",
        "Not examined across sociodemographic groups.",
        "State that this could not be assessed.",
    ),
    "8a": (
        "partial",
        "Sec. 2.1",
        "docs/data_provenance.md",
        "Binary TB-positive vs TB-negative defined.",
        "Reference standard/assessment method undefined for external cohorts.",
        "Define outcome per cohort.",
    ),
    "8b": ("missing", "-", "-", "-", "Assessor qualifications not reported.", "Report or state unavailable."),
    "8c": ("missing", "-", "-", "-", "Blinding not reported.", "State whether labels were blinded to predictors."),
    "9a": ("n_a", "-", "-", "Predictors are image pixels; no variable pre-selection.", "", ""),
    "9b": (
        "partial",
        "Sec. 2.2",
        "docs/preprocessing.md",
        "Radiographs as input.",
        "Acquisition timing/protocol unknown.",
        "Report available acquisition metadata.",
    ),
    "9c": ("n_a", "-", "-", "No subjectively interpreted predictors.", "", ""),
    "10": ("missing", "-", "-", "-", "No sample-size reasoning.", "Add justification."),
    "11": (
        "partial",
        "Sec. 2.1.1, 4",
        "-",
        "Missing auxiliary labels/demographics acknowledged.",
        "No handling described.",
        "Document handling.",
    ),
    "12a": (
        "complete",
        "Sec. 2.1, Table 1",
        "docs/data_provenance.md",
        "Train/val/internal-test split and external-only cohorts.",
        "",
        "",
    ),
    "12b": (
        "complete",
        "Sec. 2.2",
        "docs/preprocessing.md",
        "Rescaling, normalisation, resizing, augmentation.",
        "",
        "",
    ),
    "12c": (
        "partial",
        "Sec. 2.3-2.5",
        "docs/training.md",
        "Model type, training, early stopping.",
        "Hyperparameter tuning procedure not described.",
        "Describe how hyperparameters were chosen.",
    ),
    "12d": (
        "missing",
        "-",
        "-",
        "-",
        "No heterogeneity across sites/clusters analysed.",
        "Add per-site analysis or state limitation.",
    ),
    "12e": (
        "partial",
        "Sec. 3.1-3.4",
        "docs/evaluation.md",
        "Discrimination, comparison tests.",
        "No calibration or clinical-utility analysis.",
        "Add calibration and decision-curve analysis.",
    ),
    "12f": ("n_a", "-", "-", "No model updating performed.", "", ""),
    "12g": (
        "partial",
        "Sec. 2.5",
        "src/mars_tb/; scripts/infer.py",
        "Training/evaluation described.",
        "Original prediction code unavailable; reference implementation provided.",
        "Release original code if available.",
    ),
    "13": (
        "partial",
        "Sec. 2.5",
        "src/mars_tb/data/datasets.py",
        "Weighted random sampler.",
        "Reported weight (4.50) does not match the revised cohort (8.40).",
        "State actual weight.",
    ),
    "14": ("missing", "-", "-", "-", "No fairness approach.", "Add or state not assessed."),
    "15": (
        "complete",
        "Sec. 2.4.5, 2.5, Supp. Table S7",
        "docs/evaluation.md",
        "Probability and uncertainty output; Youden threshold on validation.",
        "",
        "",
    ),
    "16": (
        "complete",
        "Sec. 4",
        "docs/limitations.md",
        "Differences in labels, spectrum and prevalence discussed.",
        "",
        "",
    ),
    "17": ("n_a", "Declarations", "-", "Public datasets; 'Ethics declaration: Not applicable'.", "", ""),
    "18a": ("complete", "Declarations", "-", "No external funding.", "", ""),
    "18b": (
        "missing",
        "-",
        "-",
        "-",
        "No conflict-of-interest declaration in the manuscript.",
        "Add competing-interests statement.",
    ),
    "18c": ("missing", "-", "-", "-", "Protocol availability not stated.", "State no protocol was prepared."),
    "18d": ("missing", "-", "-", "-", "Registration not stated.", "State not registered."),
    "18e": (
        "partial",
        "Data Availability",
        "docs/data_provenance.md; data/templates/",
        "TBX11K link.",
        "External cohorts and manifests not available.",
        "Provide access instructions and manifests.",
    ),
    "18f": (
        "partial",
        "-",
        "README.md",
        "This repository.",
        "Original code unavailable; repository is a reference implementation.",
        "Release original code if it exists.",
    ),
    "19": ("missing", "-", "-", "-", "PPI not mentioned.", "State no involvement."),
    "20a": ("partial", "Sec. 2.1", "-", "Counts in text and Table 1.", "No flow diagram.", "Add diagram."),
    "20b": (
        "partial",
        "Sec. 2.1, Supp. S3-S5",
        "-",
        "Counts and prevalence per dataset.",
        "No demographics/dates; patient counts for external cohorts 'not established' in text yet given in S3.",
        "Reconcile and report.",
    ),
    "20c": (
        "missing",
        "-",
        "-",
        "-",
        "No development-vs-evaluation distribution comparison.",
        "Add comparison of available characteristics.",
    ),
    "21": (
        "partial",
        "Table 1",
        "reporting/manuscript_consistency_audit.md",
        "Per-analysis counts.",
        "Internal test N inconsistent (2,238 vs 2,221).",
        "Reconcile.",
    ),
    "22": (
        "partial",
        "Sec. 2.4",
        "docs/architecture.md; src/mars_tb/",
        "Architecture described; reference implementation here.",
        "Weights unavailable; original code unavailable.",
        "Release weights/code or explain.",
    ),
    "23a": (
        "partial",
        "Sec. 3.1-3.2",
        "docs/statistics.md",
        "CIs for AUC/Se/Sp.",
        "No subgroups; interval consistency issues.",
        "Recompute; add subgroups.",
    ),
    "23b": ("missing", "-", "-", "-", "No cluster heterogeneity.", "Add or state limitation."),
    "24": ("n_a", "-", "-", "No updating.", "", ""),
    "25": (
        "partial",
        "Sec. 3-5",
        "docs/limitations.md",
        "Interpretation with caveats.",
        "No fairness discussion.",
        "Add.",
    ),
    "26": ("complete", "Sec. 4", "docs/limitations.md", "Detailed limitations.", "", ""),
    "27a": (
        "partial",
        "Sec. 2.2",
        "src/mars_tb/data/preprocessing.py",
        "Quality filter defined.",
        "Deployment handling of poor inputs not discussed.",
        "Add.",
    ),
    "27b": ("missing", "-", "-", "-", "User interaction/expertise not discussed.", "Add."),
    "27c": ("complete", "Sec. 4, 5", "docs/limitations.md", "Future work listed.", "", ""),
}

RESPONSE = {"complete": "Yes", "partial": "No", "missing": "No", "n_a": "Not applicable"}
COLS = [
    "item",
    "section",
    "requirement",
    "response",
    "status",
    "manuscript_section",
    "repository_documentation",
    "evidence",
    "missing_information",
    "corrective_action",
]


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    claim_titles = json.loads(
        (HERE / "official_checklist_text" / "claim_2024_item_titles.json").read_text(encoding="utf-8")
    )["items"]
    tripod = json.loads((HERE / "official_checklist_text" / "tripod_ai_2024_items.json").read_text(encoding="utf-8"))[
        "items"
    ]
    assert set(map(int, claim_titles)) == set(CLAIM) == set(range(1, 45))
    assert [t["item"] for t in tripod] == list(TRIPOD)
    crows = []
    for i in range(1, 45):
        st, loc, doc, ev, miss, fix = CLAIM[i]
        crows.append(
            {
                "item": str(i),
                "section": CLAIM_SECTION[i],
                "requirement": claim_titles[str(i)],
                "response": RESPONSE[st],
                "status": st,
                "manuscript_section": loc,
                "repository_documentation": doc,
                "evidence": ev,
                "missing_information": miss,
                "corrective_action": fix,
            }
        )
    write(HERE / "CLAIM_2024_checklist.csv", crows)
    trows = []
    for t in tripod:
        st, loc, doc, ev, miss, fix = TRIPOD[t["item"]]
        trows.append(
            {
                "item": t["item"],
                "section": t["section"],
                "requirement": t["text"],
                "response": RESPONSE[st],
                "status": st,
                "manuscript_section": loc,
                "repository_documentation": doc,
                "evidence": ev,
                "missing_information": miss,
                "corrective_action": fix,
            }
        )
    write(HERE / "TRIPOD_AI_2024_checklist.csv", trows)
    for name, rows in (("CLAIM 2024", crows), ("TRIPOD+AI", trows)):
        counts = {s: sum(r["status"] == s for r in rows) for s in ("complete", "partial", "missing", "n_a")}
        print(name, len(rows), counts)


if __name__ == "__main__":
    main()
