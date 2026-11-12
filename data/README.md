# data/

No images or patient information live here. Contents:

* `schemas/` — JSON Schemas for manifest rows, prediction rows and split sidecars (copies of the packaged
  schemas in `src/mars_tb/resources/`; a test keeps them identical).
* `templates/` — header-only / one-row examples of the three file formats (`*.example.csv`; the rows are
  illustrative, not real data).
* `splits/` — generated split CSV + `.meta.json` sidecars go here. **No historical split is shipped** (see
  `splits/README.md`).

Raw data go under a directory of your choice and are referenced through manifests (`path` is relative to the
`--root` you pass; absolute paths and `..` are rejected).
