# Checkpoints

**Status: no trained MARS-TB checkpoint is available in this repository. None is shipped, and none is
promised.**

| Artifact | Status |
|---|---|
| Original (manuscript) MARS-TB weights | **Not available.** The authors have not supplied them; they were not found in any inspected source. See `reporting/artifact_inventory.csv`. |
| Newly trained checkpoint from this code | **Not available.** Training needs the TBX11K images (not redistributed here) and a GPU. |
| Randomly initialised weights | Deliberately **not** published: they are not a trained model. |

## How a checkpoint is produced and verified

```bash
python scripts/train.py --split-csv data/splits/<split>.csv --root <TBX11K root> --out-dir runs/seed42 --seed 42
```

`runs/seed42/best.ckpt` is written together with `best.ckpt.sha256`.
`mars_tb.training.checkpointing.load_checkpoint` refuses to load a file whose bytes do not match the sidecar,
and checkpoints are readable with `torch.load(weights_only=True)` (no pickle code execution).

If authorised weights are later released, publish them as a GitHub Release asset (or on a model hub) together
with the SHA-256, the exact config, the seed, the validation-selected threshold and the split-manifest hash,
then add the row to `reporting/artifact_inventory.csv`. Until then this directory stays empty.
