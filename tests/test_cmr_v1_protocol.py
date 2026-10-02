"""CMR-V1 protocol-freeze tests (T10-T11).

T10: frozen protocol identity (id / version / status) and frozen file hashes.
T11: X3-X5 outcome artifacts never exist, and X2 artifacts exist only once the
     X2 stage has been formally started with the frozen protocol.

Frozen authority: ``docs/phases/cmr_v1/CMR_V1_FREEZE_MANIFEST.json``.

Stage-transition note (recorded, not a protocol change): T11 originally asserted
that *no* X2-X5 artifact existed while X1 was the only authorized stage. The X2
stage is now authorized, so T11 keeps the X3-X5 prohibition unconditional and
makes the X2 prohibition conditional on X2 not having started (identified by the
absence of ``configs/cmr_v1_feature_map_manifest.json``, which the frozen X2
stage writes before any X2 BACC is computed). No scientific threshold, layer,
seed or gate is affected.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
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

FEATURE_MAP_MANIFEST = ROOT / "configs/cmr_v1_feature_map_manifest.json"


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


# -- T11: stage artifact discipline --------------------------------------------
# Stage-transition note (recorded, not a protocol change): before the CMR-D
# closeout, ``evidence/cmr_v1/final_gate_summary.json`` was on this list because
# only a full CMR-A/B/C run could produce it. X2 failed its frozen X2-G2 gate and
# the protocol stopped at CMR-D, whose closeout requires exactly that artifact, so
# it is no longer an X3-X5 stage marker. The X3, X4 and X5 prohibitions below are
# unchanged and still unconditional.
X3_X4_X5_PATHS = (
    "runs/cmr_v1/x3",
    "runs/cmr_v1/x4",
    "runs/cmr_v1/x5",
    "evidence/cmr_v1/x3",
    "evidence/cmr_v1/x4",
    "evidence/cmr_v1/x5",
    "configs/cmr_v1_wilds_pair_manifest.json",
    "configs/cmr_v1_wilds_unblind_authorization.json",
    "configs/cmr_v1_external_raw_manifest.json",
    "configs/cmr_v1_external_pair_manifest.json",
    "configs/cmr_v1_external_unblind_authorization.json",
)

X2_STAGE_PATHS = (
    "runs/cmr_v1/x2",
    "evidence/cmr_v1/x2",
    "configs/cmr_v1_feature_map_manifest.json",
    "configs/cmr_v1_feature_bank_manifest.json",
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def test_t11_no_x3_to_x5_artifact_exists_or_was_ever_committed():
    existing = [
        relative
        for relative in X3_X4_X5_PATHS
        if (ROOT / relative).exists()
    ]
    assert existing == [], f"X3-X5 artifacts present: {existing}"
    tracked = _git("ls-files", *X3_X4_X5_PATHS).split()
    assert tracked == [], f"X3-X5 artifacts tracked by git: {tracked}"
    history = _git("log", "--all", "--name-only", "--pretty=format:").splitlines()
    committed = sorted(
        {
            line.strip()
            for line in history
            if line.strip().startswith(("runs/cmr_v1/x3", "runs/cmr_v1/x4", "runs/cmr_v1/x5",
                                        "evidence/cmr_v1/x3", "evidence/cmr_v1/x4",
                                        "evidence/cmr_v1/x5"))
        }
    )
    assert committed == [], f"X3-X5 artifacts committed in history: {committed}"


def test_t11_x2_artifacts_only_after_x2_stage_started():
    x2_started = FEATURE_MAP_MANIFEST.exists()
    if not x2_started:
        existing = [relative for relative in X2_STAGE_PATHS if (ROOT / relative).exists()]
        assert existing == [], f"X2 artifacts present before the X2 stage started: {existing}"
        return
    manifest = json.loads(FEATURE_MAP_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["manifest_type"] == "CMR_V1_FEATURE_MAP_MANIFEST"
    assert manifest["protocol_id"] == PROTOCOL_ID
    assert manifest["protocol_version"] == PROTOCOL_VERSION
    assert manifest["frozen_before_x2_outcome_access"] is True
    assert len(manifest["maps"]) == 8
