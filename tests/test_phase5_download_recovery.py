from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "download_phase5_tier1_wilds.py"

spec = importlib.util.spec_from_file_location("phase5_download_recovery_tool", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load tool module from {SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

inspect_dataset_state = mod.inspect_dataset_state
quarantine_incomplete = mod.quarantine_incomplete


def test_release_marker_without_metadata_is_incomplete(tmp_path: Path):
    d = tmp_path / "rxrx1_v1.0"
    d.mkdir()
    (d / "RELEASE_v1.0.txt").write_text("marker")
    s = inspect_dataset_state(tmp_path, "rxrx1")
    assert s["release_marker"] is True
    assert s["metadata_csv"] is False
    assert s["status"] == "INCOMPLETE_LOCAL_DIRECTORY"


def test_quarantine_preserves_partial_files(tmp_path: Path):
    root = tmp_path / "raw"
    qroot = tmp_path / "quarantine"
    d = root / "rxrx1_v1.0"
    d.mkdir(parents=True)
    (d / "RELEASE_v1.0.txt").write_text("marker")
    (d / "partial.bin").write_bytes(b"partial")
    q = quarantine_incomplete(root, "rxrx1", qroot)
    assert not d.exists()
    assert (q / "partial.bin").read_bytes() == b"partial"
