"""CMR-V1 formal runner - X1 only (``--mode x1``).

Frozen authority:

- ``docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md``;
- ``configs/cross_mechanism_replication_protocol_v1.yaml``;
- ``docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv``;
- ``docs/phases/cmr_v1/CMR_V1_FREEZE_MANIFEST.json``.

X2-X5 are intentionally not implemented and not authorized. Running this
runner never touches real datasets, encoders or random Fourier maps.

Formal outputs:

- ``runs/cmr_v1/x1/run_summary.json``      (byte-stable scientific output);
- ``runs/cmr_v1/x1/rerun_audit.json``      (execution-history audit);
- ``evidence/cmr_v1/x1/gate_summary.json``;
- ``evidence/cmr_v1/x1/protocol_snapshot.yaml``;
- ``evidence/cmr_v1/x1/evidence_manifest.json``.

Exit codes: 0 = X1 PASS, 1 = X1_IMPLEMENTATION_BLOCKED (artifacts preserved),
2 = local guard failure (baseline mismatch / freeze-manifest hash mismatch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.cmr_kernel_layer import (
    FLOAT_DTYPE_NAME,
    PROTOCOL_ID,
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
    canonical_dumps,
    evaluate_x1_exact_collision,
)

EXPECTED_BASELINE = "d409ee2e1009a876c773c48687c7d72ac6ed4102"

PROTOCOL_MD = ROOT / "docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md"
PROTOCOL_YAML = ROOT / "configs/cross_mechanism_replication_protocol_v1.yaml"
GATE_MATRIX = ROOT / "docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv"
FREEZE_MANIFEST = ROOT / "docs/phases/cmr_v1/CMR_V1_FREEZE_MANIFEST.json"
IMPL_PROMPT = ROOT / "docs/phases/cmr_v1/CMR_V1_CODEX_X1_IMPLEMENTATION_PROMPT.md"

MODULE_SRC = ROOT / "src/frontier/cmr_kernel_layer.py"
RUNNER_SRC = ROOT / "tools/run_cmr_v1.py"
TEST_X1 = ROOT / "tests/test_cmr_v1_x1.py"
TEST_PROTOCOL = ROOT / "tests/test_cmr_v1_protocol.py"

RUN_DIR = ROOT / "runs/cmr_v1/x1"
RUN_SUMMARY = RUN_DIR / "run_summary.json"
RERUN_AUDIT = RUN_DIR / "rerun_audit.json"

EVIDENCE_DIR = ROOT / "evidence/cmr_v1/x1"
GATE_SUMMARY = EVIDENCE_DIR / "gate_summary.json"
PROTOCOL_SNAPSHOT = EVIDENCE_DIR / "protocol_snapshot.yaml"
EVIDENCE_MANIFEST = EVIDENCE_DIR / "evidence_manifest.json"

FROZEN_FILE_EXPECTATIONS = (
    ("CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.0.md", PROTOCOL_MD),
    ("cross_mechanism_replication_protocol_v1.yaml", PROTOCOL_YAML),
    ("CMR_V1_GATE_MATRIX.csv", GATE_MATRIX),
    ("CMR_V1_CODEX_X1_IMPLEMENTATION_PROMPT.md", IMPL_PROMPT),
)

DECISION_PASS = "X1_PASS_EXACT_KERNEL_PROTOTYPE_LAYER_COLLISION"
DECISION_BLOCKED = "X1_IMPLEMENTATION_BLOCKED"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "NO_GIT_HEAD"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def check_baseline(expected: str) -> tuple[str, bool | None]:
    head = git_head()
    if expected == "":
        print("[CMR-V1 X1] WARNING: baseline check explicitly disabled (--expected-baseline '').")
        return head, None
    if head != expected:
        print("[CMR-V1 X1] BASELINE MISMATCH - formal run not started.")
        print(f"[CMR-V1 X1]   expected: {expected}")
        print(f"[CMR-V1 X1]   actual:   {head}")
        print("[CMR-V1 X1] Report this before rerunning. If a new baseline was")
        print("[CMR-V1 X1] authorized, pass it explicitly via --expected-baseline.")
        return head, False
    return head, True


def load_protocol_authority() -> dict:
    doc = yaml.safe_load(PROTOCOL_YAML.read_text(encoding="utf-8"))
    protocol = doc["protocol"]
    if protocol["id"] != PROTOCOL_ID:
        raise RuntimeError(f"protocol id mismatch: {protocol['id']!r}")
    if protocol["version"] != PROTOCOL_VERSION:
        raise RuntimeError(f"protocol version mismatch: {protocol['version']!r}")
    if protocol["status"] != PROTOCOL_STATUS:
        raise RuntimeError(f"protocol status mismatch: {protocol['status']!r}")

    x1 = doc["x1"]
    checks = (
        (x1["mode"] == "exact_analytic", "x1.mode"),
        (x1["feature_map"] == "[cos(theta), sin(theta)]", "x1.feature_map"),
        (int(x1["class_count"]) == 6, "x1.class_count"),
        (math.isclose(float(x1["a"]), 0.2, rel_tol=0.0, abs_tol=1e-15), "x1.a"),
        (math.isclose(float(x1["q_squared"]), 0.96, rel_tol=0.0, abs_tol=1e-15), "x1.q_squared"),
        (math.isclose(float(x1["expected_bacc_good"]), 1.0), "x1.expected_bacc_good"),
        (math.isclose(float(x1["expected_bacc_bad"]), 1.0 / 3.0), "x1.expected_bacc_bad"),
        (
            math.isclose(float(x1["expected_deterministic_regret"]), 2.0 / 3.0),
            "x1.expected_deterministic_regret",
        ),
        (
            math.isclose(float(x1["expected_randomized_regret"]), 1.0 / 3.0),
            "x1.expected_randomized_regret",
        ),
        (float(x1["numeric_tolerance"]) == 1e-12, "x1.numeric_tolerance"),
        (bool(x1["hard_gate"]) is True, "x1.hard_gate"),
    )
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise RuntimeError(f"frozen x1 protocol constants mismatch: {failed}")
    return {
        "id": protocol["id"],
        "version": protocol["version"],
        "status": protocol["status"],
        "x1": {
            "mode": x1["mode"],
            "feature_map": x1["feature_map"],
            "class_count": int(x1["class_count"]),
            "a": float(x1["a"]),
            "q_squared": float(x1["q_squared"]),
            "expected_bacc_good": float(x1["expected_bacc_good"]),
            "expected_bacc_bad": float(x1["expected_bacc_bad"]),
            "expected_deterministic_regret": float(x1["expected_deterministic_regret"]),
            "expected_randomized_regret": float(x1["expected_randomized_regret"]),
            "numeric_tolerance": float(x1["numeric_tolerance"]),
            "hard_gate": bool(x1["hard_gate"]),
        },
    }


def verify_freeze_manifest() -> dict:
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("status") != PROTOCOL_STATUS
    ):
        raise RuntimeError("freeze manifest identity mismatch")
    files = manifest["files"]
    verified = []
    for name, path in FROZEN_FILE_EXPECTATIONS:
        expected = files[name]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen file hash mismatch for {name}: expected {expected} actual {actual}"
            )
        verified.append(
            {
                "name": name,
                "path": rel(path),
                "sha256": actual,
                "expected_sha256": expected,
                "match": True,
            }
        )
    optional = []
    for name in ("README.md", "CMR_V1_REPOSITORY_LAYOUT.md"):
        expected = files[name]["sha256"]
        resolved = None
        for candidate in (
            ROOT / name,
            ROOT / "docs" / "phases" / "cmr_v1" / name,
            ROOT / "docs" / "governance" / "nature" / name,
            ROOT / "configs" / name,
        ):
            if candidate.exists() and sha256_file(candidate) == expected:
                resolved = candidate
                break
        optional.append(
            {
                "name": name,
                "expected_sha256": expected,
                "resolved": resolved is not None,
                "path": rel(resolved) if resolved is not None else None,
            }
        )
    return {
        "manifest_path": rel(FREEZE_MANIFEST),
        "manifest_sha256": sha256_file(FREEZE_MANIFEST),
        "verified_files": verified,
        "optional_entries": optional,
    }


def run_unit_tests() -> dict:
    targets = [TEST_X1]
    if TEST_PROTOCOL.exists():
        targets.append(TEST_PROTOCOL)
    rel_targets = [rel(target) for target in targets]
    command = [sys.executable, "-m", "pytest", *rel_targets, "-q", "-p", "no:cacheprovider"]
    proc = subprocess.run(
        command, cwd=str(ROOT), capture_output=True, text=True, timeout=3600
    )
    stdout = proc.stdout or ""
    passed_tests = None
    failed_tests = None
    match = re.search(r"(\d+) passed", stdout)
    if match:
        passed_tests = int(match.group(1))
    match = re.search(r"(\d+) failed", stdout)
    if match:
        failed_tests = int(match.group(1))
    return {
        "targets": rel_targets,
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "tests_passed": passed_tests,
        "tests_failed": failed_tests,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CMR-V1 formal runner (X1 only; X2-X5 are not authorized)."
    )
    parser.add_argument("--mode", required=True, help="only 'x1' is authorized")
    parser.add_argument(
        "--expected-baseline",
        default=EXPECTED_BASELINE,
        help=(
            "expected git HEAD before the formal run; pass an authorized new "
            "baseline explicitly after a reviewed freeze commit, or '' to record "
            "the HEAD without blocking"
        ),
    )
    args = parser.parse_args(argv)

    if args.mode != "x1":
        print(
            f"[CMR-V1 X1] mode={args.mode!r} is not authorized: X2-X5 must not run "
            "under the X1 implementation prompt.",
            file=sys.stderr,
        )
        return 2

    head, baseline_match = check_baseline(args.expected_baseline)
    if baseline_match is False:
        return 2

    try:
        protocol_meta = load_protocol_authority()
    except Exception as exc:  # noqa: BLE001 - guarded startup failure
        print(f"[CMR-V1 X1] protocol authority check failed: {exc}", file=sys.stderr)
        return 2

    try:
        freeze = verify_freeze_manifest()
    except Exception as exc:  # noqa: BLE001 - guarded startup failure
        print(f"[CMR-V1 X1] freeze manifest verification failed: {exc}", file=sys.stderr)
        return 2

    result = evaluate_x1_exact_collision()
    result_rerun = evaluate_x1_exact_collision()
    in_process_rerun_identical = canonical_dumps(result) == canonical_dumps(result_rerun)

    unit_tests = run_unit_tests()

    gates = dict(result["gates"])
    gates["X1-G6"] = {
        "name": "deterministic rerun",
        "passed": bool(in_process_rerun_identical),
        "in_process_rerun_identical": bool(in_process_rerun_identical),
        "protocol_ref": "execution requirement: repeated formal evaluation is identical",
    }
    gates["X1-G7"] = {
        "name": "unit-test status",
        "passed": bool(unit_tests["passed"]),
        "targets": unit_tests["targets"],
        "returncode": unit_tests["returncode"],
        "tests_passed": unit_tests["tests_passed"],
        "tests_failed": unit_tests["tests_failed"],
        "protocol_ref": "protocol section 7.4 condition 8",
    }
    all_gates_pass = all(bool(gate["passed"]) for gate in gates.values())
    decision = DECISION_PASS if all_gates_pass else DECISION_BLOCKED

    payload = {
        "protocol": {
            "id": protocol_meta["id"],
            "version": protocol_meta["version"],
            "status": protocol_meta["status"],
        },
        "x1_result": result,
        "gates": gates,
        "decision": decision,
    }
    payload_sha = hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()

    previous_payload_sha = None
    if RUN_SUMMARY.exists():
        try:
            previous_payload_sha = json.loads(
                RUN_SUMMARY.read_text(encoding="utf-8")
            ).get("deterministic_payload_sha256")
        except Exception:  # noqa: BLE001 - preserve audit signal
            previous_payload_sha = "UNREADABLE"
    cross_run_payload_match = (
        None if previous_payload_sha is None else previous_payload_sha == payload_sha
    )

    source_files = [MODULE_SRC, RUNNER_SRC, TEST_X1]
    if TEST_PROTOCOL.exists():
        source_files.append(TEST_PROTOCOL)

    run_summary = {
        "run_type": "CMR_V1_X1_FORMAL",
        "stage": "X1",
        "protocol": payload["protocol"],
        "git_head_before_x1": head,
        "expected_baseline": args.expected_baseline,
        "baseline_match": baseline_match,
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "float_dtype": FLOAT_DTYPE_NAME,
        },
        "provenance": {
            "protocol_md_sha256": sha256_file(PROTOCOL_MD),
            "protocol_yaml_sha256": sha256_file(PROTOCOL_YAML),
            "gate_matrix_sha256": sha256_file(GATE_MATRIX),
            "implementation_prompt_sha256": sha256_file(IMPL_PROMPT),
            "freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST),
            "source_files": {rel(path): sha256_file(path) for path in source_files},
            "freeze_manifest_verification": freeze,
        },
        "x1_result": result,
        "gates": gates,
        "decision": decision,
        "deterministic_payload_sha256": payload_sha,
    }
    write_json(RUN_SUMMARY, run_summary)

    rerun_audit = {
        "run_type": "CMR_V1_X1_RERUN_AUDIT",
        "stage": "X1",
        "git_head_before_x1": head,
        "in_process_rerun_identical": bool(in_process_rerun_identical),
        "current_payload_sha256": payload_sha,
        "previous_run_summary_present": previous_payload_sha is not None,
        "previous_payload_sha256": previous_payload_sha,
        "cross_run_payload_match": cross_run_payload_match,
    }
    write_json(RERUN_AUDIT, rerun_audit)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROTOCOL_YAML, PROTOCOL_SNAPSHOT)
    if sha256_file(PROTOCOL_SNAPSHOT) != sha256_file(PROTOCOL_YAML):
        raise RuntimeError("protocol snapshot copy is not byte-identical")

    gate_summary = {
        "protocol_id": protocol_meta["id"],
        "protocol_version": protocol_meta["version"],
        "protocol_status": protocol_meta["status"],
        "stage": "X1",
        "git_head_before_x1": head,
        "provenance": {
            "protocol_md_sha256": sha256_file(PROTOCOL_MD),
            "protocol_yaml_sha256": sha256_file(PROTOCOL_YAML),
            "gate_matrix_sha256": sha256_file(GATE_MATRIX),
            "freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST),
            "module_sha256": sha256_file(MODULE_SRC),
            "runner_sha256": sha256_file(RUNNER_SRC),
        },
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "decision": decision,
    }
    write_json(GATE_SUMMARY, gate_summary)

    artifacts = [
        {
            "path": rel(GATE_SUMMARY),
            "role": "X1 gate summary (X1-G1..X1-G7)",
            "sha256": sha256_file(GATE_SUMMARY),
            "bytes": GATE_SUMMARY.stat().st_size,
        },
        {
            "path": rel(PROTOCOL_SNAPSHOT),
            "role": "frozen protocol snapshot (byte-identical copy of the frozen yaml)",
            "sha256": sha256_file(PROTOCOL_SNAPSHOT),
            "bytes": PROTOCOL_SNAPSHOT.stat().st_size,
        },
        {
            "path": rel(RUN_SUMMARY),
            "role": "formal X1 run summary",
            "sha256": sha256_file(RUN_SUMMARY),
            "bytes": RUN_SUMMARY.stat().st_size,
        },
    ]
    evidence_manifest = {
        "manifest_type": "CMR_V1_X1_EVIDENCE_MANIFEST",
        "protocol_id": protocol_meta["id"],
        "protocol_version": protocol_meta["version"],
        "protocol_status": protocol_meta["status"],
        "stage": "X1",
        "git_head_before_x1": head,
        "decision": decision,
        "artifacts": artifacts,
        "source_files": [
            {"path": rel(path), "sha256": sha256_file(path)} for path in source_files
        ],
        "protocol_files": [
            {"path": rel(path), "sha256": sha256_file(path)}
            for path in (PROTOCOL_MD, PROTOCOL_YAML, GATE_MATRIX, FREEZE_MANIFEST, IMPL_PROMPT)
        ],
        "notes": [
            "X1-only implementation; no X2-X5 artifacts were generated or authorized.",
            "run_summary.json is byte-stable across reruns; rerun_audit.json records execution history.",
        ],
    }
    write_json(EVIDENCE_MANIFEST, evidence_manifest)

    bacc = result["balanced_accuracy"]
    print("=" * 72)
    print(f"[CMR-V1 X1] git HEAD: {head}")
    print(f"[CMR-V1 X1] message equality max_abs: {result['message_equality']['max_abs_vector_sum_diff']:.3e}")
    print(f"[CMR-V1 X1] prototype equality max_abs: {result['prototype_equality']['max_abs_diff']:.3e}")
    print(
        "[CMR-V1 X1] BACC "
        f"A.layer1={bacc['A']['layer1']:.12f} A.layer2={bacc['A']['layer2']:.12f} "
        f"B.layer1={bacc['B']['layer1']:.12f} B.layer2={bacc['B']['layer2']:.12f}"
    )
    print(
        f"[CMR-V1 X1] optimal sets: A={result['optimal_sets']['A']} B={result['optimal_sets']['B']}"
    )
    print(
        f"[CMR-V1 X1] deterministic regret: {result['deterministic_regret']['value']:.12f} "
        f"(expected {result['expected']['deterministic_regret']:.12f})"
    )
    print(
        f"[CMR-V1 X1] randomized regret:    {result['randomized_regret']['value']:.12f} "
        f"(expected {result['expected']['randomized_regret']:.12f})"
    )
    print(f"[CMR-V1 X1] in-process rerun identical: {in_process_rerun_identical}")
    print(
        f"[CMR-V1 X1] unit tests: returncode={unit_tests['returncode']} "
        f"passed={unit_tests['tests_passed']} failed={unit_tests['tests_failed']}"
    )
    print(f"[CMR-V1 X1] payload sha256: {payload_sha}")
    print(f"[CMR-V1 X1] decision: {decision}")
    print(f"[CMR-V1 X1] run summary: {rel(RUN_SUMMARY)}")
    print(f"[CMR-V1 X1] gate summary: {rel(GATE_SUMMARY)}")
    print("=" * 72)
    return 0 if all_gates_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
