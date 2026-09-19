from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "download_phase5_tier1_resumable.py"

spec = importlib.util.spec_from_file_location("phase5_resumable_download_tool", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load tool module from {SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DATASETS = mod.DATASETS


def test_frozen_wilds_sizes_and_versions():
    assert DATASETS["rxrx1"]["version"] == "1.0"
    assert DATASETS["rxrx1"]["compressed_size"] == 7_413_123_845
    assert DATASETS["iwildcam"]["version"] == "2.0"
    assert DATASETS["iwildcam"]["compressed_size"] == 11_957_420_032


def test_urls_remain_official_codalab_bundle_paths():
    assert "0x6b7a05a3056a434498f0bb1252eb8440" in DATASETS["rxrx1"]["url"]
    assert "0x6313da2b204647e79a14b468131fcd64" in DATASETS["iwildcam"]["url"]
