"""Arithmetic and consistency audit of the numbers REPORTED in the manuscript.

Everything here is computed from the transcription in ``reporting/manuscript_reported_values.json``.
Statuses:
    PASS  the reported numbers agree with a recomputation
    FAIL  two places in the manuscript contradict each other (or a recomputation contradicts it)
    FLAG  not provably wrong, but needs an explanation or source records before it is relied on
    INFO  a computed observation reviewers may ask about
No finding here is evidence of intent; they are things the authors must be able to document.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats

from mars_tb.evaluation.metrics import clopper_pearson, mcc_from_rates
from mars_tb.evaluation.statistical_tests import holm


@dataclass(frozen=True)
class Finding:
    id: str
    status: str
    title: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return math.sqrt(var)


def check_table1(v: dict[str, Any]) -> list[Finding]:
    rows, total = v["table1"]["rows"], v["table1"]["total"]
    out: list[Finding] = []
    tbx = [r for r in rows if r["dataset"] == "TBX11K"]
    comp = v["tbx11k_composition"]
    ok = (
        sum(r["images"] for r in tbx) == comp["analysed"]
        and sum(r["pos"] for r in tbx) == comp["positives"]
        and comp["active"] + comp["latent"] + comp["both"] == comp["positives"]
        and comp["positives"] + comp["healthy"] + comp["sick_non_tb"] == comp["analysed"]
        and comp["analysed"] + comp["excluded_uncertain"] == comp["original_total"]
    )
    out.append(
        Finding(
            "C01",
            "PASS" if ok else "FAIL",
            "TBX11K composition and partition sums",
            f"partitions sum to {sum(r['images'] for r in tbx)} (analysed {comp['analysed']}); positives "
            f"{sum(r['pos'] for r in tbx)}={comp['active']}+{comp['latent']}+{comp['both']}; "
            f"{comp['analysed']}+{comp['excluded_uncertain']}={comp['original_total']}",
        )
    )
    pct_bad = [
        r
        for r in rows
        if abs(100 * r["pos"] / r["images"] - r["pos_pct"]) > 0.005 or r["pos"] + r["neg"] != r["images"]
    ]
    tot_ok = (
        sum(r["images"] for r in rows) == total["images"]
        and sum(r["pos"] for r in rows) == total["pos"]
        and sum(r["neg"] for r in rows) == total["neg"]
        and abs(100 * total["pos"] / total["images"] - total["pos_pct"]) < 0.005
    )
    out.append(
        Finding(
            "C02",
            "PASS" if not pct_bad and tot_ok else "FAIL",
            "Table 1 row and total arithmetic",
            f"all rows' pos+neg=images and prevalence reproduce; totals {total['images']}/{total['pos']}/{total['neg']} "
            f"({100 * total['pos'] / total['images']:.2f} %)",
        )
    )
    t = v["text_counts"]
    t1_test = next(r for r in tbx if r["partition"] == "Internal test")
    conflict = (t["internal_test_n_section_3_1"], t["internal_test_pos_section_3_1"]) != (
        t1_test["images"],
        t1_test["pos"],
    )
    out.append(
        Finding(
            "C03",
            "FAIL" if conflict else "PASS",
            "Internal test-set size: Table 1 vs Section 3.1 / Table 4",
            f"Table 1: N={t1_test['images']} ({t1_test['pos']} positive / {t1_test['neg']} negative). Section 3.1 and Table 4: "
            f"N={t['internal_test_n_section_3_1']} ({t['internal_test_pos_section_3_1']} / {t['internal_test_neg_section_3_1']}). "
            "The results tables were evidently computed on a different test set than the revised dataset table describes.",
        )
    )
    t1_train = next(r for r in tbx if r["partition"] == "Training")
    out.append(
        Finding(
            "C04",
            "FAIL" if t["training_images_section_3_2"] != t1_train["images"] else "PASS",
            "Training-set size: Table 1 vs Section 3.2",
            f"Table 1: {t1_train['images']}; Section 3.2: {t['training_images_section_3_2']}.",
        )
    )
    a, b = t["sampler_weight_formula"]
    revised = t1_train["neg"] / t1_train["pos"]
    out.append(
        Finding(
            "C05",
            "FLAG",
            "Class-imbalance sampling weight is stale",
            f"Reported 4.50 = {a}/{b} = {a / b:.4f} (the earlier cohort). For the revised training split "
            f"{t1_train['neg']}/{t1_train['pos']} = {revised:.3f}. Which value was used in the reported experiments is unrecorded.",
        )
    )
    return out


def _cp_fit(se_pct: float, n: int, ci: list[float]) -> float:
    k = round(se_pct / 100 * n)
    lo, hi = clopper_pearson(k, n)
    return max(abs(100 * lo - ci[0]), abs(100 * hi - ci[1]))


def check_table4(v: dict[str, Any]) -> list[Finding]:
    rows = v["table4_internal"]
    n_pos, n_neg = v["text_counts"]["internal_test_pos_section_3_1"], v["text_counts"]["internal_test_neg_section_3_1"]
    t1 = next(r for r in v["table1"]["rows"] if r["partition"] == "Internal test")
    out: list[Finding] = []
    err_a = max(abs(mcc_from_rates(r["se"] / 100, r["sp"] / 100, n_pos, n_neg) - r["mcc"]) for r in rows)
    err_b = max(abs(mcc_from_rates(r["se"] / 100, r["sp"] / 100, t1["pos"], t1["neg"]) - r["mcc"]) for r in rows)
    out.append(
        Finding(
            "C06",
            "PASS" if err_a < 0.0015 else "FAIL",
            "Table 4 MCC recomputed from Se, Sp and class counts",
            f"max |MCC_reported - MCC_recomputed| = {err_a:.4f} using N={n_pos + n_neg} ({n_pos}/{n_neg}); "
            f"{err_b:.4f} using Table 1's N={t1['images']} ({t1['pos']}/{t1['neg']}).",
        )
    )
    fit_a = max(max(_cp_fit(r["se"], n_pos, r["se_ci"]), _cp_fit(r["sp"], n_neg, r["sp_ci"])) for r in rows)
    fit_b = max(max(_cp_fit(r["se"], t1["pos"], r["se_ci"]), _cp_fit(r["sp"], t1["neg"], r["sp_ci"])) for r in rows)
    out.append(
        Finding(
            "C07",
            "FAIL" if fit_a < fit_b else "INFO",
            "Which test-set size Table 4's exact binomial CIs correspond to",
            f"max deviation (percentage points) of reported Clopper-Pearson bounds from recomputed ones: {fit_a:.2f} with "
            f"{n_pos}/{n_neg} positives/negatives vs {fit_b:.2f} with Table 1's {t1['pos']}/{t1['neg']}. "
            "The smaller value identifies the test set the intervals were computed on.",
        )
    )
    ratios = []
    for r in rows:
        half = (r["auc_ci"][1] - r["auc_ci"][0]) / 2
        ratios.append(half / (1.96 * hanley_mcneil_se(r["auc"], n_pos, n_neg)))
    out.append(
        Finding(
            "C08",
            "INFO",
            "Table 4 AUC interval width vs the Hanley-McNeil approximation",
            f"reported half-width / (1.96 x HM standard error) ranges {min(ratios):.2f}-{max(ratios):.2f} (mean {np.mean(ratios):.2f}). "
            "DeLong and Hanley-McNeil usually agree within ~10-15 %; values far below 1 suggest the intervals are narrower than "
            "the stated test-set size supports. Needs the prediction file to settle.",
        )
    )
    keys = ("auc", "se", "sp", "mcc")
    mono = all(all(rows[i][k] < rows[i + 1][k] for i in range(len(rows) - 1)) for k in keys)
    out.append(
        Finding(
            "C09",
            "INFO" if mono else "PASS",
            "Ordering of the ten models is identical on AUC, sensitivity, specificity and MCC",
            f"{'All four metrics are strictly increasing down the same ten-row ordering.' if mono else 'Orderings differ across metrics.'} "
            "A reviewer may ask whether each row is an independently trained model evaluated on the same cases.",
        )
    )
    return out


def check_table5(v: dict[str, Any]) -> list[Finding]:
    out: list[Finding] = []
    for name, d in v["table5_external"].items():
        rows = d["rows"]
        mono = all(
            rows[i]["auc"] < rows[i + 1]["auc"]
            and rows[i]["se"] < rows[i + 1]["se"]
            and rows[i]["sp"] < rows[i + 1]["sp"]
            for i in range(len(rows) - 1)
        )
        ratios = [
            ((r["auc_ci"][1] - r["auc_ci"][0]) / 2) / (1.96 * hanley_mcneil_se(r["auc"], d["n_pos"], d["n_neg"]))
            for r in rows
        ]
        out.append(
            Finding(
                f"C10-{name}",
                "INFO",
                f"Table 5 ({name}): ordering and interval width",
                f"AUC/Se/Sp strictly increasing over {len(rows)} rows: {mono}; AUC half-width / HM expectation "
                f"{min(ratios):.2f}-{max(ratios):.2f}.",
            )
        )
    return out


def check_table6(v: dict[str, Any]) -> list[Finding]:
    rows = v["table6_ablation"]
    full = next(r for r in rows if r["id"] == "Full")["mean"]
    out: list[Finding] = []
    bad_delta = [r["id"] for r in rows if abs((r["mean"] - full) - r["delta"]) > 0.0006]
    out.append(
        Finding(
            "C11",
            "PASS" if not bad_delta else "FAIL",
            "Table 6 delta-AUC arithmetic",
            "every delta equals mean - 0.971 to within rounding" if not bad_delta else f"mismatch in {bad_delta}",
        )
    )
    tcrit = stats.t.ppf(0.975, 4)
    worst = max(abs(r["ci"][1] - r["mean"] - tcrit * r["sd"] / math.sqrt(5)) for r in rows)
    out.append(
        Finding(
            "C12",
            "PASS" if worst < 0.0012 else "FAIL",
            "Table 6 interval = mean +/- t(0.975, 4) x SD / sqrt(5)",
            f"max deviation of the upper bound from the formula: {worst:.4f} (SD is rounded to 3 dp).",
        )
    )
    t4 = {r["model"]: r for r in v["table4_internal"]}
    a1, a3, a6c = (next(r for r in rows if r["id"] == i) for i in ("A1", "A3", "A6c"))
    out.append(
        Finding(
            "C13",
            "FAIL",
            "Same model, different AUC across Tables 4 and 6",
            f"ResNet-50: Table 4 AUC {t4['ResNet-50']['auc']} vs Table 6 A1 {a1['mean']} (same test set, same model). "
            f"BioViL-T+LoRA: Table 4 {t4['BioViL-T (+LoRA)']['auc']} vs Table 6 A3 (A2 with LoRA r16 a32) {a3['mean']}. "
            "Either the configurations differ in an undocumented way or the numbers come from different runs.",
        )
    )
    out.append(
        Finding(
            "C14",
            "FLAG",
            "'Without PRM' equals 'BioViL-T + LoRA'",
            f"Table 6 A6c (DDFM + AGRN, no PRM) AUC {a6c['mean']} equals the plain BioViL-T+LoRA baseline "
            f"{t4['BioViL-T (+LoRA)']['auc']} (and Table 8 repeats all five per-seed values). A model with two extra modules "
            "scoring exactly the baseline is possible but needs the run logs.",
        )
    )
    return out


def check_table7(v: dict[str, Any]) -> list[Finding]:
    rows = {r["model"]: r for r in v["table7_efficiency"]}
    m, b = rows["MARS-TB"], rows["BioViL-T (+LoRA)"]
    c = v["table7_text_claims"]
    got = (
        100 * (m["total_M"] / b["total_M"] - 1),
        100 * (m["flops_G"] / b["flops_G"] - 1),
        100 * (m["ms"] / b["ms"] - 1),
    )
    ok = all(
        abs(g - e) < 0.06
        for g, e in zip(got, (c["param_increase_pct"], c["flops_increase_pct"], c["latency_increase_pct"]), strict=True)
    )
    out = [
        Finding(
            "C15",
            "PASS" if ok else "FAIL",
            "Relative-cost percentages in Section 3.5",
            f"recomputed +{got[0]:.1f} % params, +{got[1]:.1f} % FLOPs, +{got[2]:.1f} % latency",
        )
    ]
    n, d, ff = 1025, 768, 4 * 768
    macs_block = 4 * n * d * d + 2 * n * n * d + 2 * n * d * ff
    macs = 12 * macs_block + 1024 * 3 * 16 * 16 * d
    out.append(
        Finding(
            "C16",
            "FLAG",
            "Reported ViT-B/16 cost at 512x512 vs an analytic count",
            f"Analytic multiply-accumulates for a 12-layer ViT-B/16 on 1,025 tokens: {macs / 1e9:.1f} G MACs "
            f"(= {2 * macs / 1e9:.1f} GFLOPs at 2 FLOPs per MAC). The manuscript reports {rows['ViT-B/16']['flops_G']} G 'FLOPs' "
            "(obtained by scaling the 224-px figure). The quantity measured is undefined; scripts/benchmark_efficiency.py "
            "states its convention and measures it.",
        )
    )
    s1 = v["supp_s1_parameters"]
    trainable = sum(s1["components"].values())
    total = trainable + s1["frozen_backbone"]
    out.append(
        Finding(
            "C17",
            "PASS" if round(trainable / 1e6, 1) == 18.3 and round(total / 1e6, 1) == 104.6 else "FAIL",
            "Supplementary Table 1 parameter totals",
            f"component rows sum to {trainable:,} trainable (18.3 M) and {total:,} in total (104.6 M).",
        )
    )
    out.append(
        Finding(
            "C18",
            "FLAG",
            "Supplementary Table 1 lists modules absent from the main text",
            "It includes CrossScaleAttention (786,432 params) and per-band Linear(256->768) frequency projections (591,360), neither "
            "described in Section 2.4 / Figures 2-3, and shows the DDFM cross-attention input as (B,1025,768) whereas the text feeds "
            "a single 768-d vector. The 18.3 M figure depends on those modules.",
        )
    )
    return out


def check_table8(v: dict[str, Any]) -> list[Finding]:
    rows = v["table8_seeds"]["rows"]
    out: list[Finding] = []
    tcrit = stats.t.ppf(0.975, 4)
    worst_mean = worst_sd = worst_ci = 0.0
    for r in rows:
        x = np.array(r["values"])
        worst_mean = max(worst_mean, abs(x.mean() - r["mean"]))
        worst_sd = max(worst_sd, abs(x.std(ddof=1) - r["sd"]))
        worst_ci = max(worst_ci, abs(x.mean() + tcrit * x.std(ddof=1) / math.sqrt(5) - r["ci"][1]))
    out.append(
        Finding(
            "C19",
            "PASS" if worst_mean < 6e-4 and worst_sd < 6e-4 and worst_ci < 1.2e-3 else "FAIL",
            "Table 8 summaries follow from the printed per-seed values",
            f"max |mean diff| {worst_mean:.5f}, |SD diff| {worst_sd:.5f}, |upper CI diff| {worst_ci:.5f}",
        )
    )
    s42_eq_mean = all(abs(r["values"][0] - np.mean(r["values"])) <= 5e-4 for r in rows)
    devs = {r["model"]: np.array(r["values"][1:]) - r["values"][0] for r in rows}
    ref = devs["MARS-TB"]
    cos = {m: float(d @ ref / (np.linalg.norm(d) * np.linalg.norm(ref))) for m, d in devs.items()}
    template = all(c > 0.998 for c in cos.values())
    scale = {m: float(np.linalg.norm(d) / np.linalg.norm(ref)) for m, d in devs.items()}
    out.append(
        Finding(
            "C20",
            "FLAG" if template and s42_eq_mean else "INFO",
            "Table 8 per-seed deviations follow a single template",
            f"seed 42 equals the five-seed mean to 3 dp in all {len(rows)} rows: {s42_eq_mean}. Deviations of seeds 0,1,2,3 from seed 42 "
            f"are, in every row, a scalar multiple of one pattern ({', '.join(f'{x:+.4f}' for x in ref)}); cosine similarity with "
            f"MARS-TB's pattern: min {min(cos.values()):.6f}; scale factors {', '.join(f'{k}:{s:.2f}' for k, s in scale.items())}. "
            "Independent training runs do not normally produce deviations that are exactly proportional across eight models. "
            "These values should not be cited as measured results without the underlying run logs.",
        )
    )
    lora = np.array(next(r for r in rows if r["model"] == "BioViL-T + LoRA")["values"])
    mars = np.array(rows[0]["values"])
    noprm = np.array(next(r for r in rows if r["model"] == "MARS-TB without PRM")["values"])
    out.append(
        Finding(
            "C21",
            "FLAG",
            "Table 8: exact constant offset and exact duplicate rows",
            f"'BioViL-T + LoRA' = 'MARS-TB' - {float(np.ptp(mars - lora)):.4f} spread around {float(np.mean(mars - lora)):.4f} "
            f"(constant offset: {bool(np.ptp(mars - lora) < 1e-9)}); 'without PRM' is identical to 'BioViL-T + LoRA' in all five seeds: "
            f"{bool(np.array_equal(lora, noprm))}.",
        )
    )
    return out


def check_supplement(v: dict[str, Any]) -> list[Finding]:
    s = v["supp_s3_s5"]
    p, t = s["s3"]["padchest"], s["s3"]["tbportals"]
    s4 = s["s4"]
    ok = (
        s4["padchest"]["pos_patients"] + s4["padchest"]["neg_patients"] == p["patients"]
        and s4["tbportals"]["pos_patients"] + s4["tbportals"]["neg_patients"] == t["patients"]
        and s4["padchest"]["pos_extra"] + s4["padchest"]["neg_extra"] == s4["padchest"]["extra"]
        and s4["tbportals"]["pos_extra"] + s4["tbportals"]["neg_extra"] == s4["tbportals"]["extra"]
        and p["patients"] + t["patients"] == s4["patient_sum"]
        and s["s5"]["pos"] == 891 + 1847
        and s["s5"]["neg"] == 2109 + 953
        and abs(100 * s["s5"]["pos"] / s["s5"]["total"] - s["s5"]["pos_pct"]) < 0.01
    )
    out = [
        Finding(
            "C22",
            "PASS" if ok else "FAIL",
            "Supplementary Tables S3-S5 arithmetic",
            "patient and 'additional radiograph' counts add up. Internal consistency does not show that the counts were observed: "
            "the manuscript's own text says patient counts for these cohorts were 'not established' (Section 2.1) while S3 reports "
            "2,850 and 1,950 unique patients.",
        )
    ]
    rows = v["supp_s6"]
    ident = max(
        max(
            abs(r["alpha_pos"] / (r["alpha_neg"] + r["alpha_pos"]) - r["p"]),
            abs(2 / (r["alpha_neg"] + r["alpha_pos"]) - r["u"]),
        )
        for r in rows
    )
    grid = all(abs(round(r["u"] / 0.02) * 0.02 - r["u"]) < 1e-9 for r in rows)
    u_list = ", ".join(f"{r['u']:.2f}" for r in rows)
    out.append(
        Finding(
            "C23",
            "FLAG" if grid else "PASS",
            "Supplementary Table S6: records satisfy the identities exactly and U sits on a 0.02 grid",
            f"max identity error {ident:.4f}. All {len(rows)} uncertainty values are exact multiples of 0.02 "
            f"({u_list}), the signature expected if the Dirichlet parameters were back-computed from "
            "rounded (p, U), not read from raw model outputs. Raw model outputs are not normally this regular.",
        )
    )
    adj = holm([0.003, 0.0005, 0.0005])
    th_ok = abs(adj[0] - 0.003) < 1e-12 and adj[1] < 0.003 and adj[2] < 0.003
    out.append(
        Finding(
            "C24",
            "PASS" if th_ok else "FAIL",
            "Holm family in Supplementary Table S8",
            f"adjusting raw p = (0.003, 0.0005, 0.0005) over a family of 3 gives {[round(x, 4) for x in adj]}; "
            "reported '0.003' and '<0.003' are consistent. (A p-value of exactly '<0.001' cannot be Holm-adjusted precisely.)",
        )
    )
    return out


def external_facts() -> list[Finding]:
    return [
        Finding(
            "C25",
            "FAIL",
            "BioViL-T is not a ViT-B/16 (checked against the official model card, 2026-11-12)",
            "Manuscript Section 2.4.1: 'a ViT-B/16-based radiology image-text model ... 12 layers, 768, ~86 M'. Microsoft's model card "
            "(huggingface.co/microsoft/BiomedVLP-BioViL-T) describes 'a hybrid image encoder composed of a Vision Transformer and ResNet-50, "
            "where the latter is used as backbone network ... The transformer ... aggregate[s] and compare[s] image features extracted across "
            "the temporal dimension'. The official checkpoint cannot supply the 12-layer ViT-B/16 layers 4/8/12 the method needs.",
        ),
        Finding(
            "C26",
            "FLAG",
            "Category-specific allocations cited to a table that is not that table",
            "Section 2.1 says 'Category-specific allocations are provided in Supplementary Table S1'; the supplementary file's Table 1 is the "
            "architecture parameter table. The per-category allocation is therefore not available.",
        ),
        Finding(
            "C27",
            "FLAG",
            "'Zero TBX11K patient overlap' cannot be established from TBX11K metadata",
            "Section 3.2 / Table 5 caption assert zero patient overlap between TBX11K and the external cohorts; Section 2.1 states patient "
            "identifiers and cross-dataset overlap were not established, and TBX11K documents no patient identifiers.",
        ),
        Finding(
            "C28",
            "FLAG",
            "Image tensors: single channel (Fig. 1, Eq. 1) vs three channels (Table 2, Supp. Table 1)",
            "Figure 1 and Eq. 1 use (B,1,512,512); preprocessing replicates to three channels and Supplementary Table 1 has Conv2d(3,768,...). "
            "This repository uses 3-channel input and derives the single-channel DDFM input as the channel mean.",
        ),
    ]


def run_audit(values: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for fn in (check_table1, check_table4, check_table5, check_table6, check_table7, check_table8, check_supplement):
        findings += fn(values)
    return findings + external_facts()


def to_markdown(findings: list[Finding]) -> str:
    counts = {s: sum(f.status == s for f in findings) for s in ("PASS", "FAIL", "FLAG", "INFO")}
    lines = [
        "# Manuscript consistency audit",
        "",
        "Generated by `python scripts/audit_manuscript.py` from `reporting/manuscript_reported_values.json` "
        "(a hand transcription of the PDF; see `docs/known_discrepancies.md`). Statuses: PASS = recomputation agrees; "
        "FAIL = the manuscript contradicts itself or a recomputation; FLAG = needs source records before it is relied on; "
        "INFO = computed observation a reviewer may raise. **Nothing here asserts intent.**",
        "",
        f"Summary: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['FLAG']} FLAG, {counts['INFO']} INFO.",
        "",
        "| ID | Status | Check |",
        "|---|---|---|",
    ]
    lines += [f"| {f.id} | {f.status} | {f.title} |" for f in findings]
    lines.append("")
    for f in findings:
        lines += [f"### {f.id} · {f.status} · {f.title}", "", f.detail, ""]
    return "\n".join(lines)
