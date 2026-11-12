#!/usr/bin/env python
"""Encode the fixed scale prompts with a frozen CXR-BERT text encoder and save them for MS-VLE.

Output: a .pt dict {scale1_fine, scale2_lobe, scale3_whole_lung} -> (K, 768) float tensors, loadable
with ``torch.load(weights_only=True)`` and used via ``text: {source: file, path: ...}``.

SECURITY: Microsoft's CXR-BERT checkpoints ship custom model code, so loading them requires
``trust_remote_code=True``, which executes code from the model repository. This script refuses to
run unless you pass --trust-remote-code deliberately, and --revision should pin a commit hash.
The embedding is the [CLS] hidden state (768-d) of the last layer, an IMPLEMENTATION CHOICE: the
manuscript does not say which text representation was used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from mars_tb.models.multi_scale_encoder import SCALE_KEYS, load_prompts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/BiomedVLP-CXR-BERT-specialized")
    ap.add_argument("--revision", default=None, help="pin a commit hash")
    ap.add_argument("--prompts", default="builtin")
    ap.add_argument("--template", default="{}", help="e.g. 'Findings: {}.'")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if not args.trust_remote_code:
        print("refusing to run: pass --trust-remote-code to acknowledge the model repo executes code", file=sys.stderr)
        return 2
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)  # noqa: S615
    model = AutoModel.from_pretrained(args.model, revision=args.revision, trust_remote_code=True).eval()  # noqa: S615
    prompts = load_prompts(args.prompts)
    blob: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key in SCALE_KEYS:
            texts = [args.template.format(p) for p in prompts[key]]
            enc = tok(texts, padding="longest", return_tensors="pt")
            hidden = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state[:, 0]
            blob[key] = hidden.float().contiguous()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"wrote {args.out}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in blob.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
