# Security policy

## Reporting
Report vulnerabilities privately through GitHub's "Report a vulnerability" (Security tab) rather than a public
issue. This is research software without a support commitment.

## Design choices
* No credentials in code, configs, docs, history or CI. `tests/test_security.py` scans the tree for token
  patterns; `.github/workflows/security.yml` runs gitleaks, pip-audit and bandit.
* Checkpoints contain tensors and plain Python types only and are loaded with `torch.load(weights_only=True)`;
  loading requires a matching SHA-256 sidecar.
* File paths from manifests/CLIs are confined to a root (`safe_join`); absolute paths and `..` are rejected by the
  manifest schema and again at read time.
* `scripts/precompute_text_embeddings.py` can execute code from a model repository (`trust_remote_code`); it refuses
  to run without an explicit flag and supports pinning a revision.
* No raw clinical images or identifiable metadata are stored; DICOM headers are not propagated.

## If a token is ever exposed
Revoke it immediately in GitHub settings, then rotate. Do not rely on deleting the commit.
