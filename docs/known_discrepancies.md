# Known discrepancies

Issues found while reading the manuscript (`TB Paper_finial-1_up5.pdf` and its supplementary PDF) and
re-implementing it. Numbers come from `reporting/manuscript_consistency_audit.md` (generated, finding ids
`Cnn`) or from the cited source. **None of this asserts intent**; each item states what evidence would
resolve it. Statuses: **Open** = needs authors' records; **Mitigated** = the repository avoids the problem
but the manuscript still needs correcting.

| ID | Issue | Status |
|---|---|---|
| A | TBX11K analysed-image count | Open (partly resolved in this draft) |
| B | Internal test-set size | Open |
| C | Stale class-imbalance weight | Open |
| D | Auxiliary (pattern/severity) label provenance | Open, mitigated |
| E | Graph node boundaries and edges | Open, mitigated |
| F | Original model artifacts | Open |
| G | Patient counts and seed-level results | Open |
| H | Effective batch size | Mitigated |
| I | BioViL-T is not a ViT-B/16 | Open |
| J | DDFM cross-attention is degenerate as written | Mitigated |
| K | Further text / supplement / table inconsistencies | Open |

## A. TBX11K analysed-image count
*Task brief:* the abstract reports 11,104 images and Methods 11,190. *In the supplied PDF (up5)* the abstract
says **11,190** (after excluding 10 uncertain cases), matching Methods; no "11,104" appears. The arithmetic
is consistent (C01): 924 + 212 + 54 = 1,190 positives; 1,190 + 5,000 + 5,000 = 11,190; + 10 = 11,200.
*Still open:* the official TBX11K test labels are withheld (the manuscript says so), so how all 11,200 images
obtained image-level labels is undocumented. The repository therefore does not assume a count: `prepare_tbx11k.py`
reports what it finds and prints the manuscript's figure beside it.
*Evidence needed:* the label file used by the authors, with its provenance.

## B. Internal test-set size (audit C03, C07)
Table 1 gives 2,238 test images (238 positive / 2,000 negative). Section 3.1 and Table 4 use **2,221 (404 / 1,817)**.
The Table 4 MCC values reproduce to 0.0004 only with 404 / 1,817 (0.078 with 238 / 2,000), and the Clopper–Pearson
bounds fit 404 / 1,817 (max deviation 0.62 pp) far better than 238 / 2,000 (1.94 pp). The results tables were
therefore computed on an *earlier* test set. Section 3.2 likewise says 6,662 training images versus 6,714 in
Table 1 (C04). No results are carried over from either; the repository keeps results empty until produced.
*Evidence needed:* the prediction files behind Tables 4-5.

## C. Class-imbalance weight (audit C05)
Table 3 keeps `4.50 = 9,085 / 2,019` (an earlier cohort). For the revised training split 6,000 / 714 the same
recipe gives **8.40**. `class_sampling_weights()` derives the weight from the training labels; `sampler_weights:
fixed:4.5` reproduces the stale value deliberately. Which was used originally is unrecorded.

## D. Auxiliary labels
TBX11K has no morphological-pattern or severity labels and the manuscript documents no generation procedure
(Section 2.1.1). Nothing is invented: auxiliary targets are `-1` ("no verified label") and contribute zero loss
(`mars_tb/models/losses.py`). **Without verified labels the three-task objective cannot be said to have been
trained**; the repository can train the TB head alone.

## E. Graph topology
Zone boundaries and a full edge list are not given. `resources/anatomical_graph.yaml` stores a proposal; every
edge carries a provenance tag (`figure4_interpreted` or `proposal`). Laterality is ambiguous (Figure 4 draws
"Left lung" on the image left; a PA radiograph shows the patient's left lung on the image right); a
`laterality: patient` switch mirrors the zones. Self-loops and undirected edges are implementation choices.

## F. Original artifacts
No trained checkpoint, prediction file, exact split manifest or original training code was found. They are
recorded as **unavailable** in `reporting/artifact_inventory.csv` and nothing stands in for them.

## G. Patient counts and seed-level results (audit C20-C23)
* Supplementary Tables S3-S4 report 2,850 and 1,950 unique patients for the external cohorts; Section 2.1 says
  unique-patient counts "were not established". The tables add up (C22) but are not demonstrably observed.
* **Table 8**: in all eight rows seed 42 equals the five-seed mean, and the deviations of seeds 0-3 from it are
  the *same* pattern (-0.0051, -0.0025, +0.0025, +0.0051) scaled by 1.00 / 1.24 / 1.50. "BioViL-T + LoRA" equals
  "MARS-TB" minus exactly 0.0100 in every seed, and "without PRM" is identical to "BioViL-T + LoRA" in all five
  seeds. Independent training runs do not produce this regularity; the values should not be cited as
  measurements without the run logs.
* **Supplementary Table S6**: all ten uncertainty values are exact multiples of 0.02, the signature of
  back-computing (alpha_neg, alpha_pos) from rounded (p, U) rather than reading raw outputs.
* Table 4 AUC intervals are 0.63-0.75 x the Hanley-McNeil expectation (C08).
This repository does not rely on any of these values.

## H. Effective batch size
Table 3: 16 per GPU x 2 accumulation = 32, "effective batch size 32", on 2 GPUs. Under distributed data
parallelism the global batch is 16 x 2 x **2** = **64**; 32 is the single-process value.
`effective_batch_size(cfg, world_size)` returns both and `scripts/train.py` prints the one in use.

## I. BioViL-T is not a ViT-B/16 (audit C25)
Section 2.4.1 describes BioViL-T as a ViT-B/16 with 12 layers, width 768, ~86 M parameters. Microsoft's model
card (`huggingface.co/microsoft/BiomedVLP-BioViL-T`, checked 2026-11-12) describes "a hybrid image encoder
composed of a Vision Transformer and ResNet-50, where the latter is used as backbone network ... The
transformer is included ... to aggregate and compare image features extracted across the temporal dimension."
The official checkpoint cannot supply ViT-B/16 layers 4/8/12. The repository implements the architecture the
manuscript *describes* (random init, optional timm ViT-B/16 weights) and does **not** claim BioViL-T weights.
The text tower (CXR-BERT) is separate and usable through `scripts/precompute_text_embeddings.py`.

## J. DDFM cross-attention (audit via test)
Eq. 8 attends from one query vector to one key/value vector; a softmax over a single key is identically 1, so
the module reduces to a residual linear map of the frequency feature
(`tests/test_frequency_fusion.py::test_literal_single_key_attention_is_degenerate_but_band_tokens_is_not`).
Default here: the three band vectors are three key/value tokens (`kv_mode: band_tokens`), giving a real
low/mid/high weighting. The literal reading remains available as `concat_single`.

## K. Further inconsistencies
* **Table 6 vs Tables 4/7 (C13, C14):** ResNet-50 is 0.876 in Table 4 but 0.862 in Table 6 (A1); "A2 with LoRA"
  (A3) is 0.913 against 0.961 for BioViL-T + LoRA; "without PRM" (0.961) equals the baseline exactly.
* **Supplementary Table 1 vs main text (C18):** it lists CrossScaleAttention (786,432) and per-band
  Linear(256->768) projections (591,360) that Section 2.4 never describes, shows DDFM attention input
  (B,1025,768), and includes a GELU in each PRM stage; Eq. 11 reuses W1 and W3 for two stages each.
  `docs/parameter_accounting.md` compares counts row by row (six rows match exactly).
* **Channels (C28):** Fig. 1 / Eq. 1 use one channel; preprocessing replicates to three and Table S1 has
  `Conv2d(3,...)`. The implementation uses three channels and takes the channel mean for DDFM.
* **Category allocation (C26):** Section 2.1 cites Supplementary Table S1 for per-category split counts; the
  supplied S1 is the parameter table. A split stratified on the *binary label* reproduces Table 1's counts
  exactly (6,714/714, 2,238/238, 2,238/238); stratifying on the five categories cannot (6,713 / 2,238 / 2,239),
  so Table 1's counts do not identify the original strata (`tests/test_data_splits.py`).
* **"Zero patient overlap" (C27)** is asserted in Section 3.2 and Table 5 although TBX11K has no patient ids.
* **FLOPs (C16):** 116.3 G is labelled FLOPs. Measured here (CPU, default config, `FlopCounterMode` with
  attention counted): **231.6 GFLOPs at 2 FLOPs per MAC = 115.8 GMACs**, so the reported figure matches
  *MACs*, not FLOPs, and the manuscript's component breakdown (backbone 91.5 G) is below the analytic backbone
  cost (107.0 GMACs). See `reporting/measured/benchmark_cpu_default_config.json`.
* **Eq. 4** types Z_2 and Z_3 as 1024 x 768 although they have 256 and 64 tokens; Eq. 6 contains OCR-garbled
  indices; Eq. 7 types f_FFT as 256-dimensional although it is 768-dimensional.
* **Sensitivity / PPV at external prevalence:** Table 5 omits PPV/NPV although prevalence is 29.7 % and 66.0 %.
* **Declarations:** no competing-interests or protocol/registration statement (TRIPOD+AI 18b-d).
