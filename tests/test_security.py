"""Repository hygiene checks: no credentials, no large/binary artifacts, sensible ignores."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "mars_tb.egg-info",
    "build",
    "dist",
}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".cfg", ".csv", ".cff", ".ini", ""}

# Patterns are assembled from pieces so this file does not match itself.
SECRET_PATTERNS = {
    "github classic token": re.compile("gh" + r"[pousr]_[A-Za-z0-9]{30,}"),
    "github fine-grained token": re.compile("github" + r"_pat_[A-Za-z0-9_]{30,}"),
    "aws access key": re.compile("AK" + r"IA[0-9A-Z]{16}"),
    "private key block": re.compile("-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE" + " KEY-----"),
    "slack token": re.compile("xox" + r"[baprs]-[0-9A-Za-z-]{10,}"),
    "generic bearer": re.compile(r"[Aa]uthorization:\s*[Bb]earer\s+[A-Za-z0-9._-]{20,}"),
}


def _files():
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS) and p.suffix in TEXT_SUFFIXES:
            yield p


def test_no_credentials_anywhere_in_the_working_tree():
    hits = []
    for p in _files():
        if p.stat().st_size > 2_000_000:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for name, pat in SECRET_PATTERNS.items():
            if pat.search(text):
                hits.append(f"{p.relative_to(ROOT)}: {name}")
    assert not hits, hits


def test_no_binary_model_or_image_artifacts_are_committed():
    banned = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".dcm", ".dicom", ".npz", ".h5", ".pkl"}
    found = [
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in banned
        and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS)
        and "pytest" not in str(p)
    ]
    assert found == [], f"remove or git-ignore: {found}"


def test_gitignore_blocks_secrets_data_and_weights():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for needle in (".env", "*.pem", "*.key", "*.ckpt", "*.pt", "*.dcm", "data/raw/", "runs/", "*.safetensors"):
        assert needle in text, f".gitignore is missing {needle!r}"


def test_checkpoints_directory_documents_unavailability_instead_of_shipping_weights():
    readme = (ROOT / "checkpoints" / "README.md").read_text(encoding="utf-8").lower()
    assert "not available" in readme or "unavailable" in readme


def test_no_hardcoded_absolute_user_paths():
    pat = re.compile(r"[A-Za-z]:\\Users\\|/home/[a-z]+/|/Users/[A-Za-z]+/")
    bad = []
    for p in _files():
        if p.suffix == ".py" and p.name != "test_security.py":
            if pat.search(p.read_text(encoding="utf-8", errors="ignore")):
                bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad
