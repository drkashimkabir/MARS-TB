"""Named ablation / configuration variants (config overrides applied on top of the base config).

The manuscript names these ablations (Section 3.3, Table 6) but does not define their wiring;
definitions live in ``mars_tb.models.mars_tb``. A6a-A6e are removal/replacement experiments on the
full model; A1-A6 are the staged build-up whose intermediate "BioViL-T pretraining configuration"
(A2) is not defined in the manuscript and is therefore NOT provided here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ABLATIONS: dict[str, Mapping[str, Any]] = {
    "full": {},
    "A6a_no_ddfm": {"use_ddfm": False},
    "A6b_no_agrn": {"use_agrn": False},
    "A6c_no_prm": {"use_prm": False},
    "A6d_ce_head": {"head": {"type": "softmax_ce"}},
    "A6e_full_finetune": {"lora": {"enabled": False}, "backbone": {"freeze_backbone": False}},
    # staged build-up on the LoRA backbone with progressively more modules
    "stage_backbone_lora_only": {"use_ddfm": False, "use_agrn": False, "use_prm": False},
    "stage_plus_ddfm": {"use_agrn": False, "use_prm": False},
    "stage_plus_ddfm_agrn": {"use_prm": False},
}

UNDEFINED_IN_MANUSCRIPT = ("A1_resnet50_baseline", "A2_biovil_t_pretraining_configuration")


def get_ablation(name: str) -> Mapping[str, Any]:
    if name in UNDEFINED_IN_MANUSCRIPT:
        raise KeyError(
            f"{name} is not defined in the manuscript and is not implemented; see docs/known_discrepancies.md"
        )
    if name not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; available: {sorted(ABLATIONS)}")
    return ABLATIONS[name]
