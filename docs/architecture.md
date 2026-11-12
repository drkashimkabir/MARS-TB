# Architecture

```
            +-------------------------------------------------------------+
 image ---> | ViT backbone, 12 blocks, LoRA on q,v (frozen otherwise)      |
 (3,512^2)  | taps: block 4, 8, 12 patch tokens (CLS excluded)             |
            +--------+--------------------------+-------------------------+
                     |                          |
          tokens 4,8,12                    tokens 12
                     v                          v
      +--------------------------+   +--------------------------------+
      | MS-VLE                   |   | AGRN                           |
      | pool 1x / 2x2 / 4x4      |   | 12 zone nodes (mean of tokens) |
      | text cross-attn (fixed   |   | 3 x relational GAT, 8 heads    |
      | prompts), GAP, concat    |   | -> z_AGRN, p_node (12)         |
      | -> z_MS (768)            |   +---------------+----------------+
      +-----------+--------------+                   |
                  v                                  |
      +--------------------------+                   |
      | DDFM (raw image + z_MS)  |                   |
      | FFT low/mid/high -> CNN  |                   |
      | band cross-attention     |                   |
      | -> z_DDFM (768)          |                   |
      +-----------+--------------+                   |
                  +---------------- c = z_DDFM + z_AGRN
                                    v
                    PRM: screen -> localize(+p_node) -> characterize -> diagnose -> z_d
                                    v
                    MT-EDL: TB (2), pattern (4), severity (4) -> alpha, p, U
```

## Design notes

* **Backbone.** A plain ViT-B/16 (separate q/k/v/o linears so LoRA can wrap q and v). Position embeddings are
  bicubically resizable (`interpolate_pos_embed`), and weights from a timm ViT-B/16 state dict load through
  `load_timm_vit_state_dict` (verified numerically against timm). **This is not BioViL-T**
  (`known_discrepancies.md` I).
* **LoRA.** `h' = h W0 + (alpha/r) h A B`, A ~ N(0, 0.01^2), B = 0, so the adapted network starts identical to
  the frozen one. 24 projections x 24,576 = 589,824 parameters (tested at full size).
* **Language.** The three scales are conditioned on *fixed prompts* from Figure 2, never on patient reports, so
  inference needs no report. Embeddings are learned (default) or precomputed by a frozen text encoder.
* **Frequency branch.** Centred FFT, three masks that exactly partition the spectrum, band-limited images
  from the inverse FFT, one 3-layer CNN per band (256-d each).
* **Graph.** Zone pooling is a fixed row-normalised matrix. Attention is `LeakyReLU(a_dst.Wh_i + a_src.Wh_j)`
  plus a learned bias per (edge relation, head), masked to graph neighbours. The 12 regional activations are an
  unsupervised readout and **must not be reported as lesion probabilities**.
* **Evidential head.** `alpha = softplus(logit) + 1`, `p = alpha / S`, `U = K / S`; loss = NLL + annealed KL to
  the uniform Dirichlet on the misleading evidence only; weights 1.0 / 0.3 / 0.3.

## Ablation wiring (defined here; the manuscript does not define it)

| Ablation | Wiring |
|---|---|
| no DDFM | `z_DDFM := z_MS` |
| no AGRN | `c := z_DDFM`; PRM stage 2 has no node input |
| no PRM | `z_d := c` |
| CE head | softmax heads + masked cross-entropy (no U) |
| full fine-tuning | LoRA off, backbone trainable |

Parameter accounting versus the manuscript's Supplementary Table 1 is in `parameter_accounting.md`.
