# Model card: MARS-TB reference implementation

**Status: no trained model is released.** This card documents the *architecture and code*; there are no weights
to which performance statements could attach.

| | |
|---|---|
| Name | MARS-TB (Multi-scale Anatomical Reasoning System for TB), manuscript-derived reference implementation |
| Task | Binary TB-positive vs TB-negative classification of frontal chest radiographs, plus two auxiliary heads (pattern, severity) that need verified labels |
| Input | One chest radiograph, resized to 512 x 512, 3-channel (replicated grey) |
| Output | P(TB), evidential uncertainty U, Dirichlet parameters; auxiliary pattern/severity outputs when trained |
| Architecture | ViT-B/16 + LoRA, multi-scale text-conditioned encoder, FFT dual-domain fusion, 12-zone relational GAT, 4-stage progressive module, evidential multi-task head (see `architecture.md`) |
| Parameters (default config, measured) | 103,002,283 total; 16,567,723 trainable (LoRA 589,824) |
| Training data | none shipped; intended: TBX11K custom split (`training.md`) |
| Pretrained weights | none by default; optional timm ViT-B/16 state dict. **Not BioViL-T** (`known_discrepancies.md` I) |
| Performance | **None claimed.** Manuscript-reported values are unverified (`reporting/manuscript_consistency_audit.md`) |

## Intended use
Research on the method and on reproducibility. The manuscript proposes screening/triage support; that use is
**unvalidated** and out of scope here.

## Out of scope
Clinical decision-making, automated diagnosis, deployment in a medical device, use on paediatric images,
interpreting regional activations as lesion probabilities, or interpreting P(TB)/U as calibrated risk.

## Known limitations
See `limitations.md` and `known_discrepancies.md`. In particular: label provenance, image-level partitions,
external negatives that are not microbiologically confirmed, no demographic or acquisition metadata, no
calibration analysis, no subgroup or fairness evaluation, no prospective study.

## Ethical considerations
Public/controlled-access datasets are used by reference only; no patient data is distributed. TB Portals use
requires a NIAID data-use agreement. Misuse risk: a model trained on non-microbiological labels could be
mistaken for a diagnostic.
