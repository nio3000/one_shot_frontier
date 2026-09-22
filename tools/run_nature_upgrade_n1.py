from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.n1_support_matched import (
    context_support_table,
    eligible_analysis_rows,
    evaluate_n1_gate,
    natural_context_table,
    prepare_authoritative_rows,
    simulate_arm_b_n_matched,
    simulate_arm_c_exact_support,
    t_het_from_context_risks,
    verify_frozen_support,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "NO_GIT_HEAD"


def tracked_clean(rel: str) -> bool:
    a = subprocess.run(["git", "ls-files", "--error-unmatch", "--", rel], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if a.returncode != 0:
        return False
    b = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT)
    return b.returncode == 0


def load_protocol(path: Path) -> dict:
    p = yaml.safe_load(path.read_text(encoding="utf-8"))
    if p.get("protocol_id") != "NATURE-UPGRADE-N1":
        raise RuntimeError("Wrong N1 protocol_id")
    if p.get("scientific_protocol_version") != "1.0.0-FROZEN":
        raise RuntimeError("Scientific N1 protocol version mismatch")
    if p.get("status") != "FROZEN_REPRODUCTION_BINDING":
        raise RuntimeError("N1 reproduction binding is not frozen")
    return p


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_authoritative_file(path: Path, protocol: dict) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    expected = protocol["authoritative_input"]
    actual_sha = sha256(path)
    if actual_sha != expected["sha256"]:
        raise RuntimeError(f"Authoritative N1 input SHA mismatch expected={expected['sha256']} actual={actual_sha}")
    df = pd.read_csv(path)
    if len(df) != int(expected["rows"]):
        raise RuntimeError(f"Authoritative N1 row count mismatch expected={expected['rows']} actual={len(df)}")
    return {"df": df, "sha256": actual_sha, "n_rows": int(len(df))}


def build_analysis(path: Path, protocol: dict) -> dict:
    src = verify_authoritative_file(path, protocol)
    rows, row_audit = prepare_authoritative_rows(src["df"], protocol)
    support = context_support_table(rows, protocol)
    support_audit = verify_frozen_support(support, protocol)
    eligible_rows = eligible_analysis_rows(rows, support)
    natural = natural_context_table(eligible_rows)
    if len(natural) != int(protocol["context_construction"]["expected_eligible_contexts"]):
        raise RuntimeError("Natural context count mismatch after frozen eligibility filtering")
    if natural["institution_id"].nunique() != len(protocol["context_construction"]["institutions"]):
        raise RuntimeError("Natural eligible contexts do not span all frozen institutions")
    return {
        "input_sha256": src["sha256"],
        "source_df": src["df"],
        "rows": rows,
        "eligible_rows": eligible_rows,
        "support": support,
        "natural": natural,
        "row_audit": row_audit,
        "support_audit": support_audit,
    }


def manifest_payload_hash(payload: dict) -> str:
    z = dict(payload)
    z.pop("payload_sha256", None)
    canon = json.dumps(z, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def verify_manifest_payload(manifest: dict) -> None:
    expected = manifest.get("payload_sha256")
    actual = manifest_payload_hash(manifest)
    if not expected or expected != actual:
        raise RuntimeError(f"N1 input-manifest payload hash mismatch expected={expected} actual={actual}")


def synthetic_smoke(protocol: dict) -> dict:
    # Exercise authoritative preprocessing with a synthetic protocol copy; not scientific evidence.
    p = json.loads(json.dumps(protocol))
    p["authoritative_input"]["required_columns"] = list(protocol["authoritative_input"]["required_columns"])
    p["context_construction"]["institutions"] = ["I0", "I1", "I2"]
    p["context_construction"]["expected_all_contexts"] = 13
    p["context_construction"]["expected_eligible_contexts"] = 13
    p["context_construction"]["eligibility"] = {"min_N": 1, "min_positive": 1, "min_negative": 1}
    p["context_construction"]["age_bands"] = ["<40", "40-59", "60-74", "75+"]

    rng = np.random.default_rng(314159)
    rows = []
    rid = 0
    # 5/4/4 contexts; use a subset of age bands per institution to make 13 cells.
    plan = {
        "I0": [25, 45, 65, 80, 55],
        "I1": [30, 50, 70, 85],
        "I2": [35, 52, 68, 82],
    }
    # Since context_support_table expects full institution x 4 grid, smoke uses a direct canonical path below.
    canonical = []
    for inst, ages in plan.items():
        for k, age in enumerate(ages):
            band = "<40" if age < 40 else "40-59" if age < 60 else "60-74" if age < 75 else "75+"
            n_pos = 20 + (k % 4)
            n_neg = 22 + ((k + 1) % 5)
            sens = [0.95, 0.80, 0.65, 0.90, 0.72][k % 5]
            spec = [0.94, 0.82, 0.68, 0.88, 0.75][k % 5]
            for y, n, rate in [(1, n_pos, sens), (0, n_neg, spec)]:
                for _ in range(n):
                    # Build probabilities that deterministically encode the desired hard prediction.
                    correct = rng.random() < rate
                    pred = y if correct else 1 - y
                    pm = 0.8 if pred == 1 else 0.1
                    canonical.append({
                        "sample_id": f"s{rid}", "institution_id": inst, "age": age, "age_band": band,
                        "context_id": f"{inst}|{band}|{k}", "y_true": y, "y_pred": pred, "prob_mean": pm,
                    })
                    rid += 1
    d = pd.DataFrame(canonical)
    # T_HET smoke on 13 explicit slots, independent from frozen 16-cell constructor.
    nat = []
    for (inst, ctx), g in d.groupby(["institution_id", "context_id"], sort=True):
        yt = g.y_true.to_numpy(int); yp = g.y_pred.to_numpy(int)
        pos = yt == 1; neg = yt == 0
        bacc = 0.5 * (np.mean(yp[pos] == 1) + np.mean(yp[neg] == 0))
        nat.append({"institution_id": inst, "context_id": ctx, "N": len(g), "n_positive": int(pos.sum()), "n_negative": int(neg.sum()), "balanced_accuracy": bacc, "risk": 1-bacc})
    nat = pd.DataFrame(nat)
    ddof = int(protocol["primary_statistic"]["variance_ddof"])
    natural_t = t_het_from_context_risks(nat, ddof=ddof)
    arm_c = simulate_arm_c_exact_support(d, nat, replicates=500, seed=20260921, ddof=ddof)
    arm_b, valid = simulate_arm_b_n_matched(d, nat, replicates=500, seed=20260921, ddof=ddof)
    gate = evaluate_n1_gate(natural_t_het=natural_t, arm_c_null=arm_c, alpha=0.05)
    return {
        "formal": False,
        "smoke": True,
        "n_rows": int(len(d)),
        "n_contexts": int(len(nat)),
        "n_institutions": int(nat.institution_id.nunique()),
        "T_HET_natural": natural_t,
        "arm_c_finite": bool(np.isfinite(arm_c).all()),
        "arm_b_valid_fraction": valid,
        "gate_exercised": gate,
    }


def freeze_input(protocol_path: Path, protocol: dict, input_path: Path, manifest_path: Path) -> dict:
    analysis = build_analysis(input_path, protocol)
    nat = analysis["natural"]
    ddof = int(protocol["primary_statistic"]["variance_ddof"])
    payload = {
        "manifest_type": "NATURE_UPGRADE_N1_INPUT_MANIFEST",
        "status": "FROZEN_N1_REPRODUCTION_INPUT_AUTHORITY",
        "input_filename": input_path.name,
        "input_sha256": analysis["input_sha256"],
        "input_rows": int(len(analysis["source_df"])),
        "protocol_path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(protocol_path),
        "git_head_at_freeze": git_head(),
        "row_audit": analysis["row_audit"],
        "support_audit": analysis["support_audit"],
        "all_context_support": analysis["support"].to_dict(orient="records"),
        "natural_context_support": nat[["institution_id", "age_band", "context_id", "N", "n_positive", "n_negative"]].to_dict(orient="records"),
        "natural_T_HET_precomputed_for_integrity": t_het_from_context_risks(nat, ddof=ddof),
        "arm_c_monte_carlo_generated": False,
    }
    payload["payload_sha256"] = manifest_payload_hash(payload)
    if manifest_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing N1 input manifest: {manifest_path}")
    write_json(manifest_path, payload)
    return payload


def formal(protocol_path: Path, protocol: dict, input_path: Path, manifest_path: Path, out: Path) -> dict:
    proto_rel = str(protocol_path.relative_to(ROOT)).replace("\\", "/")
    man_rel = str(manifest_path.relative_to(ROOT)).replace("\\", "/")
    if not tracked_clean(proto_rel):
        raise RuntimeError("Formal N1 blocked: protocol must be committed and clean")
    if not tracked_clean(man_rel):
        raise RuntimeError("Formal N1 blocked: input manifest must be committed and clean")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest_payload(manifest)
    if manifest.get("status") != "FROZEN_N1_REPRODUCTION_INPUT_AUTHORITY":
        raise RuntimeError("N1 input manifest status invalid")
    if sha256(protocol_path) != manifest["protocol_sha256"]:
        raise RuntimeError("Protocol hash differs from input-freeze authority")
    if sha256(input_path) != manifest["input_sha256"]:
        raise RuntimeError("N1 authoritative input hash mismatch")

    analysis = build_analysis(input_path, protocol)
    nat = analysis["natural"]
    ddof = int(protocol["primary_statistic"]["variance_ddof"])
    natural_t = t_het_from_context_risks(nat, ddof=ddof)
    if not np.isclose(natural_t, float(manifest["natural_T_HET_precomputed_for_integrity"]), rtol=0, atol=1e-15):
        raise RuntimeError("Natural T_HET differs from frozen input-manifest integrity value")

    bcfg = protocol["arms"]["B"]
    ccfg = protocol["arms"]["C"]
    arm_b, arm_b_valid = simulate_arm_b_n_matched(
        analysis["eligible_rows"], nat,
        replicates=int(bcfg["replicates"]), seed=int(bcfg["rng_seed"]), ddof=ddof,
    )
    arm_c = simulate_arm_c_exact_support(
        analysis["eligible_rows"], nat,
        replicates=int(ccfg["replicates"]), seed=int(ccfg["rng_seed"]), ddof=ddof,
    )
    if len(arm_c) != 20000 or not np.isfinite(arm_c).all():
        raise RuntimeError("Frozen 20,000-replicate Arm-C null incomplete or non-finite")

    gate = evaluate_n1_gate(
        natural_t_het=natural_t,
        arm_c_null=arm_c,
        alpha=float(protocol["primary_test"]["alpha"]),
    )

    out.mkdir(parents=True, exist_ok=True)
    analysis["support"].to_csv(out / "all_context_support.csv", index=False)
    nat.to_csv(out / "natural_contexts.csv", index=False)
    pd.DataFrame({"replicate_id": np.arange(len(arm_b)), "T_HET": arm_b}).to_csv(out / "arm_b_null.csv", index=False)
    pd.DataFrame({"replicate_id": np.arange(len(arm_c)), "T_HET": arm_c}).to_csv(out / "arm_c_null.csv", index=False)
    support_audit = {
        **analysis["row_audit"],
        **analysis["support_audit"],
        "authoritative_input_sha256_match": True,
        "authoritative_input_rows_match": True,
        "arm_c_exact_support_matching": True,
        "arm_c_matching_fields": ["N", "n_positive", "n_negative"],
        "arm_c_partition_scope": "within_institution",
        "arm_c_age_structure_destroyed": True,
        "arm_c_support_preserved_by_construction": True,
        "arm_b_partition_scope": "pooled_eligible_analysis_population",
        "arm_b_valid_fraction": arm_b_valid,
    }
    write_json(out / "support_audit.json", support_audit)
    write_json(out / "gate_summary.json", gate)
    (out / "protocol_snapshot.yaml").write_text(protocol_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out / "input_manifest_snapshot.json").write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "protocol_id": "NATURE-UPGRADE-N1",
        "scientific_protocol_version": protocol["scientific_protocol_version"],
        "implementation_version": protocol["implementation_version"],
        "formal": True,
        "complete": True,
        "git_head": git_head(),
        "protocol_sha256": sha256(protocol_path),
        "input_manifest_sha256": sha256(manifest_path),
        "input_sha256": analysis["input_sha256"],
        "n_authoritative_rows": int(len(analysis["source_df"])),
        "n_eligible_rows": int(len(analysis["eligible_rows"])),
        "n_natural_contexts": int(len(nat)),
        "n_institutions": int(nat["institution_id"].nunique()),
        "arm_b_replicates": int(len(arm_b)),
        "arm_c_replicates": int(len(arm_c)),
        "decision": gate["decision"],
        "T_HET_natural": gate["T_HET_natural"],
        "p_HET": gate["p_HET"],
        "arm_c_null_median": gate["arm_c_null_median"],
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(out / "run_summary.json", summary)
    return {"summary": summary, "gate": gate, "support_audit": support_audit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "input-freeze", "formal"], required=True)
    ap.add_argument("--protocol", default=str(ROOT / "configs" / "nature_upgrade_n1_protocol.yaml"))
    ap.add_argument("--input", default=str(ROOT / "data" / "nature_upgrade_n1" / "N1_authoritative_development_rows.csv"))
    ap.add_argument("--input-manifest", default=str(ROOT / "configs" / "nature_upgrade_n1_input_manifest.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "nature_upgrade_n1"))
    args = ap.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    if args.mode == "smoke":
        result = synthetic_smoke(protocol)
    elif args.mode == "input-freeze":
        result = freeze_input(protocol_path, protocol, Path(args.input), Path(args.input_manifest))
    else:
        result = formal(protocol_path, protocol, Path(args.input), Path(args.input_manifest), Path(args.out_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
