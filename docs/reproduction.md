# Reproduction guide

Three different things are called "reproduction"; only the first is possible from this repository alone.

| Goal | Possible now? | What it needs |
|---|---|---|
| **Verify the code** (tests, plumbing, shapes, statistics) | Yes | `pip install -e ".[dev]"`; `pytest` |
| **Run a new experiment** with this implementation | Yes, with data + GPU | TBX11K images, a GPU, `docs/training.md` |
| **Reproduce the manuscript's published numbers** | **No** | the authors' prediction files / checkpoints / exact splits (all unavailable, see `reporting/reproducibility_status.md`) |

## Full workflow for a new experiment
```bash
python scripts/prepare_tbx11k.py   --root <TBX11K> --out data/manifests/tbx11k.csv
python scripts/generate_splits.py  --manifest data/manifests/tbx11k.csv --root <TBX11K> --name s0 --seed 0 --strata label
python scripts/audit_data_leakage.py --split-csv data/splits/s0.csv --root <TBX11K>
for seed in 42 0 1 2 3; do
  python scripts/train.py --split-csv data/splits/s0.csv --root <TBX11K> --out-dir runs/full/seed$seed --seed $seed
  python scripts/export_predictions.py --checkpoint runs/full/seed$seed/best.ckpt --manifest data/splits/s0.csv \
         --root <TBX11K> --dataset tbx11k --split test --only-split test --out runs/full/seed$seed/test.csv
done
python scripts/reproduce_tables.py --predictions MARS-TB=runs/full/seed42/test.csv MARS-TB=runs/full/seed0/test.csv ...
python scripts/run_ablations.py --split-csv data/splits/s0.csv --root <TBX11K> --out-dir runs/ablations
python scripts/benchmark_efficiency.py
```
Any numbers this produces are results of *your* run on *your* split; compare them with the manuscript only
after reading `reporting/manuscript_consistency_audit.md`.

## Regenerating the generated documents
```bash
python scripts/audit_manuscript.py          # reporting/manuscript_consistency_audit.{md,json}
python scripts/parameter_accounting.py      # docs/parameter_accounting.md
python reporting/build_checklists.py        # CLAIM / TRIPOD+AI CSVs from official item text
python scripts/validate_artifacts.py        # inventory, checklists, links, audit currency
```
