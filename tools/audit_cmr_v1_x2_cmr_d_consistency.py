"""Read-only CMR-V1 X2/CMR-D consistency audit (no model is re-run).

Verifies the frozen existing artifacts against the authoritative result set:

- 8 structural conditions;
- 8/8 ACTION_RELEVANT;
- all four ResNet-18 UNIQUE_BEST == layer4;
- ViT UNIQUE_BEST set == {B6, B9, B12};
- aggregation recovery max_abs <= 1e-8;
- all values finite;
- X2-G1 PASS, X2-G2 FAIL, X2-G3 PASS, X2-G4 PASS, X2-G5 PASS;
- final decision CMR-D with protocol_stop true.

Exits non-zero and prints the mismatch if any artifact disagrees.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
X2_RUN = ROOT / "runs/cmr_v1/x2"
EVIDENCE = ROOT / "evidence/cmr_v1/x2"
MIGRATION = ROOT / "runs/cmr_v1/x2_cuda_migration"

EXPECTED = {
    "conditions": 8,
    "action_relevant": 8,
    "resnet_unique_best": {"layer4"},
    "vit_unique_best": {"B6", "B9", "B12"},
    "recovery_max_abs_max": 1e-8,
    "gates": {"X2-G1": True, "X2-G2": False, "X2-G3": True, "X2-G4": True, "X2-G5": True},
    "final_cmr_decision": "CMR-D",
    "protocol_stop": True,
    "x2_decision": "ACTION_FAMILY_NOT_DECISION_RELEVANT",
}

failures: list[str] = []


def check(name: str, actual, expected) -> None:
    ok = actual == expected
    print(f"[audit] {'OK  ' if ok else 'FAIL'} {name}: {actual!r}")
    if not ok:
        failures.append(f"{name}: got {actual!r}, expected {expected!r}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with (X2_RUN / "condition_results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    run_summary = json.loads((X2_RUN / "run_summary.json").read_text(encoding="utf-8"))
    gate_summary = json.loads((EVIDENCE / "gate_summary.json").read_text(encoding="utf-8"))
    evidence_manifest = json.loads((EVIDENCE / "evidence_manifest.json").read_text(encoding="utf-8"))
    feature_map = json.loads(
        (ROOT / "configs/cmr_v1_feature_map_manifest.json").read_text(encoding="utf-8")
    )
    feature_bank = json.loads(
        (ROOT / "configs/cmr_v1_feature_bank_manifest.json").read_text(encoding="utf-8")
    )
    migration_gate = json.loads((MIGRATION / "migration_gate_summary.json").read_text(encoding="utf-8"))
    parity = json.loads((MIGRATION / "parity_report.json").read_text(encoding="utf-8"))

    check("condition rows", len(rows), EXPECTED["conditions"])
    check(
        "condition set",
        sorted(f"{r['dataset']}__{r['encoder']}" for r in rows),
        sorted(
            f"{d}__{e}"
            for d in ("cifar100", "eurosat", "pathmnist", "dermamnist")
            for e in ("resnet18_imagenet1k_v1", "vit_b_16_imagenet1k_v1")
        ),
    )
    check("ACTION_RELEVANT count", sum(int(r["action_relevant"]) for r in rows), EXPECTED["action_relevant"])
    check("finite_pass count", sum(int(r["finite_pass"]) for r in rows), EXPECTED["conditions"])

    resnet_unique = {r["unique_best"] for r in rows if r["encoder"] == "resnet18_imagenet1k_v1"}
    vit_unique = {r["unique_best"] for r in rows if r["encoder"] == "vit_b_16_imagenet1k_v1"}
    check("ResNet UNIQUE_BEST set", resnet_unique, EXPECTED["resnet_unique_best"])
    check("ViT UNIQUE_BEST set", vit_unique, EXPECTED["vit_unique_best"])
    check("distinct ResNet UNIQUE_BEST", len(resnet_unique), 1)
    check("distinct ViT UNIQUE_BEST", len(vit_unique), 3)

    recovery_max = max(float(r["aggregation_recovery_max_abs"]) for r in rows)
    check(
        "recovery max_abs <= 1e-8",
        recovery_max <= EXPECTED["recovery_max_abs_max"],
        True,
    )
    print(f"[audit]      recovery max_abs = {recovery_max:.3e}")

    spreads = [float(r["layer_spread"]) for r in rows]
    check("all layer spreads >= 0.01", all(s >= 0.01 for s in spreads), True)
    print(f"[audit]      layer spread range = {min(spreads):.6f} .. {max(spreads):.6f}")

    for gate_id, expected_pass in EXPECTED["gates"].items():
        check(f"{gate_id} passed", bool(gate_summary["gates"][gate_id]["passed"]), expected_pass)
    check("X2 all_gates_pass", bool(gate_summary["all_gates_pass"]), False)
    check("X2 decision", gate_summary["cmr_x2_decision"], EXPECTED["x2_decision"])
    check("final CMR decision", gate_summary["final_cmr_decision"], EXPECTED["final_cmr_decision"])
    check("protocol stop", bool(gate_summary["protocol_stops"]), EXPECTED["protocol_stop"])

    check(
        "run_summary agrees with gate_summary",
        (run_summary["cmr_x2_decision"], run_summary["final_cmr_decision"], bool(run_summary["protocol_stops"])),
        (EXPECTED["x2_decision"], EXPECTED["final_cmr_decision"], EXPECTED["protocol_stop"]),
    )
    check("run_summary backend", run_summary["inference_backend"]["inference_backend"], "CUDA")

    check("feature map count", len(feature_map["maps"]), 8)
    check(
        "feature map input dims",
        sorted(e["input_dimension"] for e in feature_map["maps"]),
        sorted([64, 128, 256, 512, 768, 768, 768, 768]),
    )
    check(
        "feature map RFF settings",
        (feature_map["rff"]["dimension"], feature_map["rff"]["sigma"], feature_map["rff"]["master_seed"]),
        (256, 1.0, 20261001),
    )
    check("feature bank backend", feature_bank["inference_backend"], "CUDA")
    check("feature bank conditions", len(feature_bank["banks"]), 8)
    check("CPU cache excluded from formal bank", bool(feature_bank["cpu_premigration_cache_excluded"]), True)

    check("CUDA migration status", migration_gate["cuda_migration_status"], "PASS")
    check(
        "formal X2 CUDA extraction authorized",
        migration_gate["formal_x2_cuda_reextraction"],
        "AUTHORIZED",
    )
    check("parity pass", bool(parity["parity_pass"]), True)
    check("parity conditions passed", parity["conditions_passed"], 8)
    min_agreement = min(
        stage["agreement"]
        for condition in parity["conditions"]
        for layer in condition["layers"]
        for stage in layer["stages"]
        if stage["stage"] == "prediction"
    )
    max_bacc_diff = max(
        layer["bacc_abs_diff"] for condition in parity["conditions"] for layer in condition["layers"]
    )
    check("CPU<->GPU prediction agreement", min_agreement, 1.0)
    check("CPU<->GPU BACC abs diff", max_bacc_diff, 0.0)
    check("RFF byte-identical", bool(parity["rff_identity"]["byte_identical_on_regeneration"]), True)

    for artifact in evidence_manifest["artifacts"]:
        path = ROOT / artifact["path"]
        if not path.is_file():
            failures.append(f"missing artifact {artifact['path']}")
            continue
        if sha256_file(path) != artifact["sha256"]:
            failures.append(f"evidence hash mismatch for {artifact['path']}")

    for forbidden in (
        "runs/cmr_v1/x3",
        "runs/cmr_v1/x4",
        "runs/cmr_v1/x5",
        "evidence/cmr_v1/x3",
        "evidence/cmr_v1/x4",
        "evidence/cmr_v1/x5",
        "configs/cmr_v1_wilds_pair_manifest.json",
        "configs/cmr_v1_external_pair_manifest.json",
    ):
        if (ROOT / forbidden).exists():
            failures.append(f"forbidden X3-X5 artifact present: {forbidden}")

    report = {
        "audit_type": "CMR_V1_X2_CMR_D_READ_ONLY_CONSISTENCY_AUDIT",
        "expected": {k: (sorted(v) if isinstance(v, set) else v) for k, v in EXPECTED.items()},
        "observed": {
            "conditions": len(rows),
            "action_relevant": sum(int(r["action_relevant"]) for r in rows),
            "resnet_unique_best": sorted(resnet_unique),
            "vit_unique_best": sorted(vit_unique),
            "recovery_max_abs": recovery_max,
            "layer_spread_min": min(spreads),
            "layer_spread_max": max(spreads),
            "gates": {k: bool(v["passed"]) for k, v in gate_summary["gates"].items()},
            "x2_decision": gate_summary["cmr_x2_decision"],
            "final_cmr_decision": gate_summary["final_cmr_decision"],
            "protocol_stop": bool(gate_summary["protocol_stops"]),
            "cuda_migration_status": migration_gate["cuda_migration_status"],
            "cpu_gpu_prediction_agreement": min_agreement,
            "cpu_gpu_bacc_abs_diff": max_bacc_diff,
        },
        "failures": failures,
        "audit_pass": not failures,
    }
    out = MIGRATION / "x2_cmr_d_consistency_audit.json"
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[audit] wrote {out.relative_to(ROOT)}")
    print(f"[audit] AUDIT {'PASS' if not failures else 'FAIL'}")
    if failures:
        for failure in failures:
            print(f"[audit]   {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
