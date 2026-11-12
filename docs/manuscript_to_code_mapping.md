# Manuscript-to-code mapping

Category of every item below: **Manuscript-derived implementation**. Nothing here is the authors' original
code. "Deviation" = where the code does something other than the literal manuscript, and why.

| Component | Manuscript | Code | Input -> output | Configurable | Tests | Deviation / assumption |
|---|---|---|---|---|---|---|
| Preprocessing | Sec. 2.2, Table 2 | `data/preprocessing.py`, `data/augmentation.py` | file -> (3,512,512) float | `PreprocessConfig` | `test_preprocessing.py`, `test_augmentation.py` | DICOM scaled by `2^BitsStored-1` and MONOCHROME1 inverted (a literal /65535 would leave 12-bit data ~6 % bright); bicubic rotation via `grid_sample` because torchvision tensor-rotate lacks bicubic |
| ViT backbone | Sec. 2.4.1, Eq. 1-2 | `models/biovil_backbone.py` | (B,3,512,512) -> {4,8,12: (B,1024,768)} | `BackboneConfig` | `test_backbone.py` (numerical parity with timm) | **Not BioViL-T weights** (discrepancy I); Z^(l) = block output before final LN; 3-channel input |
| LoRA | Eq. 3 | `models/lora_adapter.py` | adds (alpha/r) h A B to q, v | rank 16, alpha 32, dropout 0.1 | `test_lora.py` (589,824 params at ViT-B/16) | init A~N(0,0.01^2), B=0 as stated |
| MS-VLE | Sec. 2.4.2, Eq. 4-6, Fig. 2 | `models/multi_scale_encoder.py` | 3 token sets -> z_MS (B,768) | `MSVLEConfig`, `TextConfig` | `test_multiscale_encoder.py` | fixed prompts only (never patient reports); text embeddings learned or loaded from file; no CrossScaleAttention (not in main text) |
| DDFM | Sec. 2.4.3, Eq. 7-8, Fig. 3 | `models/frequency_fusion.py` | image + z_MS -> z_DDFM (B,768) | `DDFMConfig` (edges, geometry, kv_mode) | `test_frequency_fusion.py` | band geometry is an implementation choice; attention uses 3 band tokens (discrepancy J) |
| AGRN | Sec. 2.4.4, Eq. 9-10, Fig. 4 | `models/anatomical_graph.py`, `resources/anatomical_graph.yaml` | layer-12 tokens -> z_AGRN (B,768), p_node (B,12) | graph YAML | `test_graph_reasoning.py` | boxes and edges are proposals (discrepancy E); relation-aware attention bias is an addition |
| PRM | Sec. 2.4.5, Eq. 11 | `models/progressive_reasoning.py` | c, p_node -> z_d (B,768) | `PRMConfig` | `test_progressive_reasoning.py` | one weight matrix per stage; no GELU/residual by default |
| MT-EDL | Eq. 12 | `models/evidential_head.py`, `models/losses.py` | z_d -> alpha, p, U for 3 tasks | `HeadConfig`, `LossConfig` | `test_evidential_learning.py` | NLL variant "log" by default (manuscript says only "NLL"); missing aux labels -> no loss |
| Assembly / ablations | Fig. 1, Sec. 3.3 | `models/mars_tb.py`, `models/ablations.py` | -> output dict | `use_ddfm/agrn/prm`, ablation names | `test_model_smoke.py`, `test_config.py` | ablation wiring defined here (manuscript silent); A1/A2 refused as undefined |
| Training | Sec. 2.5, Table 3 | `training/*` | split -> checkpoint | `TrainConfig` | `test_training.py`, `test_end_to_end.py` | global batch resolved (discrepancy H); sampler weight from labels (C) |
| Evaluation | Sec. 3 | `evaluation/*` | predictions -> metrics | `configs/evaluation/` | `test_evaluation.py`, `test_statistics.py`, `test_calibration.py`, `test_uncertainty.py` | thresholds validation-only; ECE/Brier/NLL added |
| Explainability | Sec. 3.6 | `explainability/*` | model + image -> maps | CLI args | `test_explainability.py` | Grad-CAM++ on layer-12 tokens; occlusion and localisation metrics added |
| Efficiency | Sec. 3.4, Table 7 | `scripts/benchmark_efficiency.py` | model -> params/FLOPs/latency | CLI args | `test_scripts.py` | FLOP convention stated (2 x MAC) |

## Tensor shapes (default config, batch B)

```
image                         (B, 3, 512, 512)
patch tokens Z_p^(4,8,12)     (B, 1024, 768)        # 32 x 32 grid, CLS excluded
scale 1 / 2 / 3 tokens        (B,1024,768) / (B,256,768) / (B,64,768)
z_MS                          (B, 768)
band features                 (B, 3, 256)
z_DDFM                        (B, 768)
AGRN nodes                    (B, 12, 768)  -> z_AGRN (B,768), p_node (B,12)
c = z_DDFM + z_AGRN           (B, 768)
PRM stages                    (B,768) -> (B,768) -> (B,768) -> z_d (B,768)
heads                         TB (B,2), pattern (B,4), severity (B,4)
```
