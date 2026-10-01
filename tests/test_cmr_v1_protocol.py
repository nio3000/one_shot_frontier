"""CMR-V1 protocol-freeze tests (T10-T11).

T10: frozen protocol identity (id / version / status) and frozen file hashes.
T11: no X2-X5 outcome artifact exists while X1 is the only authorized stage.

Frozen authority: ``docs/phases/cmr_v1/CMR_V1_FREEZE_MANIFEST.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.cmr_kernel_layer import (
    PROTOCOL_ID,
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
    X1_EXPECTED_BACC,
    X1_EXPECTED_DETERMINISTIC_REGRET,
    X1_EXPECTED_RANDOMIZED_REGRET,
    X1_NUMERIC_TOLERANCE,
)

PROTOCOL_MD = ROOT / "docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md"
PROTOCOL_YAML = ROOT / "configs/cross_mechanism_replication_protocol_v1.yaml"
GATE_MATRIX = ROOT / "docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv"
FREEZE_MANIFEST = ROOT / "docs/phases/cmr_v1/CMR_V1_FREEZE_MANIFEST.json"
IMPL_PROMPT = ROOT / "docs/phases/cmr_v1/CMR_V1_CODEX_X1_IMPLEMENTATION_PROMPT.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_manifest() -> dict:
    return json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))


# -- T10: protocol identity / frozen status ------------------------------------
def test_t10_protocol_identity_frozen():
    doc = yaml.safe_load(PROTOCOL_YAML.read_text(encoding="utf-8"))
    assert doc["protocol"]["id"] == "CMR-V1" == PROTOCOL_ID
    assert doc["protocol"]["version"] == "1.0.0-FROZEN" == PROTOCOL_VERSION
    assert (
        doc["protocol"]["status"]
        == "FROZEN_BEFORE_CMR_OUTCOME_ACCESS"
        == PROTOCOL_STATUS
    )
    manifest = _freeze_manifest()
    assert manifest["protocol_id"] == PROTOCOL_ID
    assert manifest["protocol_version"] == PROTOCOL_VERSION
    assert manifest["status"] == PROTOCOL_STATUS
    assert manifest["x2_x5_outcome_access_authorized"] is False
    markdown = PROTOCOL_MD.read_text(encoding="utf-8")
    assert "**Protocol ID:** `CMR-V1`" in markdown
    assert "`1.0.0-FROZEN`" in markdown
    assert "`FROZEN_BEFORE_CMR_OUTCOME_ACCESS`" in markdown


def test_t10_frozen_file_hashes_match_manifest():
    files = _freeze_manifest()["files"]
    pairs = (
        ("CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.0.md", PROTOCOL_MD),
        ("cross_mechanism_replication_protocol_v1.yaml", PROTOCOL_YAML),
        ("CMR_V1_GATE_MATRIX.csv", GATE_MATRIX),
        ("CMR_V1_CODEX_X1_IMPLEMENTATION_PROMPT.md", IMPL_PROMPT),
    )
    for name, path in pairs:
        assert _sha256(path) == files[name]["sha256"], name


def test_t10_x1_yaml_matches_module_constants():
    x1 = yaml.safe_load(PROTOCOL_YAML.read_text(encoding="utf-8"))["x1"]
    assert x1["mode"] == "exact_analytic"
    assert x1["feature_map"] == "[cos(theta), sin(theta)]"
    assert int(x1["class_count"]) == 6
    assert math.isclose(float(x1["a"]), 0.2, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(float(x1["q_squared"]), 0.96, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(float(x1["expected_bacc_good"]), 1.0)
    assert math.isclose(
        float(x1["expected_bacc_bad"]), X1_EXPECTED_BACC["A"]["layer2"]
    )
    assert math.isclose(
        float(x1["expected_deterministic_regret"]), X1_EXPECTED_DETERMINISTIC_REGRET
    )
    assert math.isclose(
        float(x1["expected_randomized_regret"]), X1_EXPECTED_RANDOMIZED_REGRET
    )
    assert float(x1["numeric_tolerance"]) == X1_NUMERIC_TOLERANCE == 1e-12
    assert bool(x1["hard_gate"]) is True


# -- T11: no X2-X5 outcome artifacts -------------------------------------------
def test_t11_no_x2_to_x5_outcome_artifacts():
    forbidden = [
        ROOT / "runs/cmr_v1/x2",
        ROOT / "runs/cmr_v1/x3",
        ROOT / "runs/cmr_v1/x4",
        ROOT / "runs/cmr_v1/x5",
        ROOT / "evidence/cmr_v1/x2",
        ROOT / "evidence/cmr_v1/x3",
        ROOT / "evidence/cmr_v1/x4",
        ROOT / "evidence/cmr_v1/x5",
        ROOT / "configs/cmr_v1_feature_map_manifest.json",
        ROOT / "configs/cmr_v1_feature_bank_manifest.json",
        ROOT / "configs/cmr_v1_wilds_pair_manifest.json",
        ROOT / "configs/cmr_v1_wilds_unblind_authorization.json",
        ROOT / "configs/cmr_v1_external_raw_manifest.json",
        ROOT / "configs/cmr_v1_external_pair_manifest.json",
        ROOT / "configs/cmr_v1_external_unblind_authorization.json",
    ]
    existing = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in forbidden
        if path.exists()
    ]
    assert existing == [], f"X2-X5 artifacts present at X1 time: {existing}"
