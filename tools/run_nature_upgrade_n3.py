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

from frontier.n3_patient_dependence import (
    bind_ptb_patients,
    binding_digest,
    cluster_bootstrap_status,
    context_support,
    evaluate_primary_gate,
    exact_support_matched_null,
    identity_fingerprint,
    monte_carlo_upper_p,
    natural_contexts,
    prepare_authoritative_rows,
    prepare_ptbxl_metadata,
    select_one_ecg_per_patient,
    sha256_file,
    status_polarity_reversal,
    t_het,
    verify_identity_fingerprint,
    wilson_status,
)


def load_protocol(path: Path) -> dict:
    p = yaml.safe_load(path.read_text(encoding="utf-8"))
    if p.get("protocol_id") != "NATURE-UPGRADE-N3":
        raise RuntimeError("Wrong N3 protocol_id")
    if p.get("scientific_protocol_version") != "V1.0-FROZEN":
        raise RuntimeError("N3 scientific protocol version mismatch")
    if p.get("status") != "RECONSTRUCTED_EXECUTABLE_FROM_FROZEN_HISTORICAL_PROTOCOL":
        raise RuntimeError("N3 executable authority status mismatch")
    return p


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


def rel_to_root(path: Path) -> str:
    p = Path(path).resolve()
    return str(p.relative_to(ROOT.resolve())).replace("\\", "/")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def manifest_payload_hash(payload: dict) -> str:
    z = dict(payload)
    z.pop("payload_sha256", None)
    canon = json.dumps(z, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def verify_manifest_payload(manifest: dict) -> None:
    expected = manifest.get("payload_sha256")
    actual = manifest_payload_hash(manifest)
    if not expected or expected != actual:
        raise RuntimeError(f"N3 binding-manifest payload hash mismatch expected={expected} actual={actual}")


def load_and_verify_inputs(n1_path: Path, ptb_meta_path: Path, protocol: dict) -> dict:
    n1_path = Path(n1_path).resolve()
    ptb_meta_path = Path(ptb_meta_path).resolve()
    n1cfg = protocol["inputs"]["n1_authoritative_rows"]
    ptbcfg = protocol["inputs"]["ptbxl_metadata"]
    if not n1_path.exists():
        raise FileNotFoundError(n1_path)
    if not ptb_meta_path.exists():
        raise FileNotFoundError(ptb_meta_path)
    n1_sha = sha256_file(n1_path)
    ptb_sha = sha256_file(ptb_meta_path)
    if n1_sha != n1cfg["sha256"]:
        raise RuntimeError(f"N1 authoritative SHA mismatch expected={n1cfg['sha256']} actual={n1_sha}")
    if ptb_sha != ptbcfg["sha256"]:
        raise RuntimeError(f"PTB-XL metadata SHA mismatch expected={ptbcfg['sha256']} actual={ptb_sha}")

    n1_src = pd.read_csv(n1_path)
    ptb_src = pd.read_csv(ptb_meta_path)
    if len(n1_src) != int(n1cfg["rows"]):
        raise RuntimeError(f"N1 row count mismatch expected={n1cfg['rows']} actual={len(n1_src)}")
    if len(ptb_src) != int(ptbcfg["rows"]):
        raise RuntimeError(f"PTB metadata row count mismatch expected={ptbcfg['rows']} actual={len(ptb_src)}")

    rows, row_audit = prepare_authoritative_rows(n1_src, protocol)
    meta, meta_audit = prepare_ptbxl_metadata(ptb_src, protocol)
    if meta_audit["unique_ecg_id"] != int(ptbcfg["unique_ecg_id"]):
        raise RuntimeError("PTB metadata unique ecg_id fingerprint mismatch")
    if meta_audit["unique_patient_id"] != int(ptbcfg["unique_patient_id"]):
        raise RuntimeError("PTB metadata unique patient_id fingerprint mismatch")
    ptb, linkage_audit = bind_ptb_patients(rows, meta)
    fp = identity_fingerprint(ptb)
    fp_match = verify_identity_fingerprint(fp, protocol["identity_fingerprint"])
    return {
        "n1_path": n1_path,
        "ptb_meta_path": ptb_meta_path,
        "n1_sha256": n1_sha,
        "ptb_meta_sha256": ptb_sha,
        "n1_source": n1_src,
        "ptb_source": ptb_src,
        "rows": rows,
        "meta": meta,
        "ptb": ptb,
        "row_audit": row_audit,
        "meta_audit": meta_audit,
        "linkage_audit": linkage_audit,
        "identity_fingerprint": fp,
        "identity_fingerprint_match": fp_match,
        "binding_sha256": binding_digest(ptb),
    }


def implementation_hashes(protocol_path: Path) -> dict:
    paths = {
        "protocol": protocol_path,
        "frozen_protocol_source": ROOT / "protocols" / "NATURE_UPGRADE_N3_PROTOCOL_V1_0_FROZEN.md",
        "module": ROOT / "src" / "frontier" / "n3_patient_dependence.py",
        "runner": ROOT / "tools" / "run_nature_upgrade_n3.py",
        "tests": ROOT / "tests" / "test_n3_patient_dependence.py",
    }
    return {k: {"path": rel_to_root(v), "sha256": sha256_file(v)} for k, v in paths.items()}


def require_implementation_committed_clean(protocol_path: Path) -> None:
    for item in implementation_hashes(protocol_path).values():
        rel = item["path"]
        if not tracked_clean(rel):
            raise RuntimeError(f"N3 authority freeze blocked: implementation file must be committed and clean: {rel}")


def binding_freeze(protocol_path: Path, protocol: dict, n1_path: Path, ptb_meta_path: Path, manifest_path: Path) -> dict:
    # Input authority is frozen only after the reconstructed executable itself is committed and clean.
    require_implementation_committed_clean(protocol_path)
    a = load_and_verify_inputs(n1_path, ptb_meta_path, protocol)
    payload = {
        "manifest_type": "NATURE_UPGRADE_N3_BINDING_MANIFEST",
        "status": "FROZEN_N3_REPRODUCTION_INPUT_AUTHORITY",
        "historical_outcome_known": True,
        "git_head_at_binding_freeze": git_head(),
        "n1_input_filename": Path(n1_path).name,
        "n1_input_sha256": a["n1_sha256"],
        "n1_input_rows": int(len(a["n1_source"])),
        "ptbxl_metadata_filename": Path(ptb_meta_path).name,
        "ptbxl_metadata_sha256": a["ptb_meta_sha256"],
        "ptbxl_metadata_rows": int(len(a["ptb_source"])),
        "row_audit": a["row_audit"],
        "metadata_audit": a["meta_audit"],
        "linkage_audit": a["linkage_audit"],
        "identity_fingerprint": a["identity_fingerprint"],
        "identity_fingerprint_match": True,
        "binding_sha256": a["binding_sha256"],
        "implementation_authority": implementation_hashes(protocol_path),
        "formal_outcomes_generated": False,
    }
    payload["payload_sha256"] = manifest_payload_hash(payload)
    if manifest_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing N3 binding manifest: {manifest_path}")
    write_json(manifest_path, payload)
    return payload


def verify_binding_manifest(manifest_path: Path, protocol_path: Path, n1_path: Path, ptb_meta_path: Path, protocol: dict) -> tuple[dict, dict]:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest_payload(m)
    if m.get("status") != "FROZEN_N3_REPRODUCTION_INPUT_AUTHORITY":
        raise RuntimeError("N3 binding manifest status invalid")
    # Formal execution requires the complete frozen implementation authority plus manifest to be committed+clean.
    require_implementation_committed_clean(protocol_path)
    man_rel = rel_to_root(manifest_path)
    if not tracked_clean(man_rel):
        raise RuntimeError(f"Formal N3 blocked: authority file must be committed and clean: {man_rel}")
    current_impl = implementation_hashes(protocol_path)
    for key, item in m["implementation_authority"].items():
        if key not in current_impl or current_impl[key]["sha256"] != item["sha256"]:
            raise RuntimeError(f"N3 implementation SHA changed after binding freeze: {key}")
    if sha256_file(Path(n1_path)) != m["n1_input_sha256"]:
        raise RuntimeError("N3 N1 input SHA differs from binding authority")
    if sha256_file(Path(ptb_meta_path)) != m["ptbxl_metadata_sha256"]:
        raise RuntimeError("N3 PTB metadata SHA differs from binding authority")
    analysis = load_and_verify_inputs(n1_path, ptb_meta_path, protocol)
    if analysis["binding_sha256"] != m["binding_sha256"]:
        raise RuntimeError("N3 patient binding digest differs from frozen binding authority")
    if analysis["identity_fingerprint"] != m["identity_fingerprint"]:
        raise RuntimeError("N3 identity fingerprint differs from frozen binding authority")
    return m, analysis


def ptb_record_status_table(ptb: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    src = float(protocol["prediction"]["source_bacc"])
    margin = float(protocol["prediction"]["primary_margin"])
    z = float(protocol["record_level_wilson"]["z"])
    rows = []
    for band in protocol["context"]["ptb_primary_age_bands"]:
        g = ptb[ptb.age_band == band]
        s = wilson_status(g, src, margin, z)
        rows.append({"institution_id": "PTB-XL", "age_band": band, "context_id": f"PTB-XL|{band}", **s})
    return pd.DataFrame(rows)


def ptb_cluster_table(ptb: pd.DataFrame, record_status: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    src = float(protocol["prediction"]["source_bacc"])
    margin = float(protocol["prediction"]["primary_margin"])
    cfg = protocol["cluster_bootstrap"]
    rng = np.random.default_rng(int(cfg["master_seed"]))
    rows = []
    for band in protocol["context"]["ptb_primary_age_bands"]:
        g = ptb[ptb.age_band == band]
        b = cluster_bootstrap_status(
            g, source_bacc=src, margin=margin,
            replicates=int(cfg["replicates_per_context"]), rng=rng,
            qlo=float(cfg["ci_lower_quantile"]), qhi=float(cfg["ci_upper_quantile"]),
        )
        ref = str(record_status.loc[record_status.age_band == band, "status"].iloc[0])
        rows.append({
            "institution_id": "PTB-XL", "age_band": band, "context_id": f"PTB-XL|{band}",
            "record_level_status": ref,
            **b,
            "status_concordant": bool(b["status"] == ref),
            "polarity_reversal": bool(status_polarity_reversal(ref, b["status"])),
        })
    return pd.DataFrame(rows)


def build_dedup_rows(rows: pd.DataFrame, ptb: pd.DataFrame, protocol: dict, rule: str = "hash") -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(protocol["one_ecg_per_patient"]["hash_seed"])
    selected = select_one_ecg_per_patient(ptb, seed=seed, rule=rule)
    non_ptb = rows[rows.institution_id != "PTB-XL"].copy()
    # selected includes binding-only identity columns; keep canonical N1 columns plus identity for PTB audit.
    dedup = pd.concat([non_ptb, selected[rows.columns.tolist()]], ignore_index=True)
    return dedup, selected


def dedup_ptb_status(selected: pd.DataFrame, record_status: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    src = float(protocol["prediction"]["source_bacc"])
    margin = float(protocol["prediction"]["primary_margin"])
    z = float(protocol["record_level_wilson"]["z"])
    rows = []
    for band in protocol["context"]["ptb_primary_age_bands"]:
        g = selected[selected.age_band == band]
        s = wilson_status(g, src, margin, z)
        ref = str(record_status.loc[record_status.age_band == band, "status"].iloc[0])
        rows.append({
            "institution_id": "PTB-XL", "age_band": band, "context_id": f"PTB-XL|{band}",
            "record_level_status": ref,
            **s,
            "status_concordant": bool(s["status"] == ref),
            "polarity_reversal": bool(status_polarity_reversal(ref, s["status"])),
        })
    return pd.DataFrame(rows)


def eligible_set(support: pd.DataFrame) -> set[str]:
    return set(support.loc[support.eligible, "context_id"].astype(str))


def formal(protocol_path: Path, protocol: dict, n1_path: Path, ptb_meta_path: Path, manifest_path: Path, outdir: Path) -> dict:
    m, a = verify_binding_manifest(manifest_path, protocol_path, n1_path, ptb_meta_path, protocol)
    rows, ptb = a["rows"], a["ptb"]

    original_support = context_support(rows, protocol)
    original_eligible = eligible_set(original_support)
    if len(original_eligible) != int(protocol["context"]["expected_eligible_contexts"]):
        raise RuntimeError(f"Original N1 eligible-context count is not 13: {len(original_eligible)}")

    record_status = ptb_record_status_table(ptb, protocol)
    cluster = ptb_cluster_table(ptb, record_status, protocol)

    dedup_rows, selected = build_dedup_rows(rows, ptb, protocol, rule="hash")
    dedup_support = context_support(dedup_rows, protocol)
    dedup_eligible = eligible_set(dedup_support)
    same_eligible = dedup_eligible == original_eligible and len(dedup_eligible) == int(protocol["context"]["expected_eligible_contexts"])
    dedup_status = dedup_ptb_status(selected, record_status, protocol)

    natural = natural_contexts(dedup_rows, dedup_support)
    ddof = int(protocol["class_support_replay"]["variance_ddof"])
    natural_t = t_het(natural, ddof=ddof)
    rcfg = protocol["class_support_replay"]
    null = exact_support_matched_null(
        dedup_rows, natural,
        replicates=int(rcfg["replicates"]), seed=int(rcfg["rng_seed"]), ddof=ddof,
    )
    p_het = monte_carlo_upper_p(null, natural_t)
    null_median = float(np.median(null))

    cluster_conc = int(cluster.status_concordant.sum())
    dedup_conc = int(dedup_status.status_concordant.sum())
    cluster_rev = int(cluster.polarity_reversal.sum())
    dedup_rev = int(dedup_status.polarity_reversal.sum())
    gate = evaluate_primary_gate(
        linkage_fraction=float(a["linkage_audit"]["linkage_fraction"]),
        fingerprint_match=bool(a["identity_fingerprint_match"]),
        same_eligible_contexts=bool(same_eligible),
        cluster_concordance=cluster_conc,
        dedup_concordance=dedup_conc,
        cluster_polarity_reversals=cluster_rev,
        dedup_polarity_reversals=dedup_rev,
        replay_p=p_het,
        natural_t=natural_t,
        null_median=null_median,
        protocol=protocol,
    )

    # Frozen secondary deterministic selection-rule checks. They do not affect the primary gate.
    secondary = {}
    for rule in ["min_ecg_id", "max_ecg_id"]:
        sec_rows, sec_selected = build_dedup_rows(rows, ptb, protocol, rule=rule)
        sec_support = context_support(sec_rows, protocol)
        sec_status = dedup_ptb_status(sec_selected, record_status, protocol)
        secondary[rule] = {
            "dedup_rows": int(len(sec_rows)),
            "ptb_selected_rows": int(len(sec_selected)),
            "eligible_contexts": sorted(eligible_set(sec_support)),
            "ptb_status": sec_status.to_dict(orient="records"),
        }

    # Cross-age patient reporting under the three deterministic selection rules.
    cross = ptb.groupby("patient_id")["age_band"].nunique()
    cross_ids = sorted(cross[cross > 1].index.astype(str).tolist())
    cross_rows = []
    for pid in cross_ids:
        g = ptb[ptb.patient_id.astype(str) == pid]
        item = {"patient_id": pid, "age_bands_present": "|".join(sorted(set(g.age_band), key=lambda x: ["<40","40-59","60-74","75+"].index(x)))}
        for rule in ["hash", "min_ecg_id", "max_ecg_id"]:
            s = select_one_ecg_per_patient(g, int(protocol["one_ecg_per_patient"]["hash_seed"]), rule=rule).iloc[0]
            item[f"{rule}_record_id"] = str(s.record_id)
            item[f"{rule}_ecg_id"] = str(s.ecg_id)
            item[f"{rule}_age_band"] = str(s.age_band)
        cross_rows.append(item)
    cross_df = pd.DataFrame(cross_rows)

    outdir = Path(outdir).resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty N3 formal output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    record_status.to_csv(outdir / "ptb_record_level_wilson_status.csv", index=False)
    cluster.to_csv(outdir / "ptb_patient_cluster_bootstrap_status.csv", index=False)
    dedup_status.to_csv(outdir / "ptb_one_ecg_per_patient_wilson_status.csv", index=False)
    dedup_support.to_csv(outdir / "one_ecg_per_patient_support.csv", index=False)
    natural.to_csv(outdir / "one_ecg_per_patient_natural_contexts.csv", index=False)
    pd.DataFrame({"replicate_id": np.arange(len(null)), "T_HET": null}).to_csv(outdir / "one_ecg_per_patient_arm_c_null.csv", index=False)
    cross_df.to_csv(outdir / "cross_age_patient_selection_audit.csv", index=False)
    write_json(outdir / "secondary_selection_rule_sensitivity.json", secondary)
    write_json(outdir / "primary_gate.json", gate)
    (outdir / "protocol_snapshot.yaml").write_text(protocol_path.read_text(encoding="utf-8"), encoding="utf-8")
    (outdir / "binding_manifest_snapshot.json").write_text(Path(manifest_path).read_text(encoding="utf-8"), encoding="utf-8")
    write_json(outdir / "binding_audit.json", {
        "manifest_status": m["status"],
        "binding_sha256": a["binding_sha256"],
        "linkage_audit": a["linkage_audit"],
        "identity_fingerprint": a["identity_fingerprint"],
        "identity_fingerprint_match": True,
        "cpsc_patient_identity": "unavailable__no_inferred_clustering",
        "chapman_shaoxing_synthetic_patient_ids": False,
        "ningbo_synthetic_patient_ids": False,
    })

    summary = {
        "protocol_id": protocol["protocol_id"],
        "scientific_protocol_version": protocol["scientific_protocol_version"],
        "implementation_version": protocol["implementation_version"],
        "reproduction_of_historical_frozen_analysis": True,
        "formal": True,
        "complete": True,
        "git_head": git_head(),
        "n1_input_sha256": a["n1_sha256"],
        "ptbxl_metadata_sha256": a["ptb_meta_sha256"],
        "binding_sha256": a["binding_sha256"],
        "bound_ptbxl_rows": int(len(ptb)),
        "bound_ptbxl_unique_patients": int(ptb.patient_id.nunique()),
        "cluster_bootstrap_replicates_per_context": int(protocol["cluster_bootstrap"]["replicates_per_context"]),
        "cluster_status_concordance": f"{cluster_conc}/3",
        "cluster_polarity_reversals": cluster_rev,
        "one_ecg_per_patient_total_rows": int(len(dedup_rows)),
        "one_ecg_per_patient_ptb_rows": int(len(selected)),
        "one_ecg_status_concordance": f"{dedup_conc}/3",
        "one_ecg_polarity_reversals": dedup_rev,
        "eligible_contexts_after_dedup": int(len(dedup_eligible)),
        "same_eligible_context_set": bool(same_eligible),
        "T_HET_natural": float(natural_t),
        "arm_c_replicates": int(len(null)),
        "arm_c_null_median": null_median,
        "arm_c_null_mean": float(np.mean(null)),
        "arm_c_null_q025": float(np.quantile(null, 0.025)),
        "arm_c_null_q975": float(np.quantile(null, 0.975)),
        "arm_c_exceedances": int(np.sum(null >= natural_t)),
        "p_HET": float(p_het),
        "decision": gate["decision"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(outdir / "N3_SUMMARY.json", summary)
    return {"summary": summary, "gate": gate, "cluster_status": cluster.to_dict(orient="records"), "dedup_status": dedup_status.to_dict(orient="records")}


def synthetic_smoke(protocol: dict) -> dict:
    # Engineering smoke only. It exercises Wilson status, cluster resampling,
    # deterministic one-patient selection and the primary gate without using scientific data.
    rng = np.random.default_rng(7)
    rows = []
    rid = 0
    for band, age, sens, spec in [("40-59", 50, .90, .94), ("60-74", 67, .82, .89), ("75+", 80, .70, .82)]:
        for p in range(40):
            pid = f"P{band}-{p}"
            mult = 2 if p % 5 == 0 else 1
            for _ in range(mult):
                y = 1 if rid % 4 == 0 else 0
                correct = rng.random() < (sens if y else spec)
                pred = y if correct else 1-y
                rows.append({"record_id": f"R{rid}", "institution_id": "PTB-XL", "age": age, "age_band": band,
                             "context_id": f"PTB-XL|{band}", "y_true": y, "y_pred": pred, "prob_mean": .8 if pred else .1,
                             "patient_id": pid, "ecg_id": str(rid+1)})
                rid += 1
    ptb = pd.DataFrame(rows)
    src = float(protocol["prediction"]["source_bacc"])
    margin = float(protocol["prediction"]["primary_margin"])
    z = float(protocol["record_level_wilson"]["z"])
    rec = []
    clu = []
    rrng = np.random.default_rng(20260922)
    for band in protocol["context"]["ptb_primary_age_bands"]:
        g = ptb[ptb.age_band == band]
        s = wilson_status(g, src, margin, z)
        rec.append({"age_band": band, **s})
        b = cluster_bootstrap_status(g, source_bacc=src, margin=margin, replicates=100, rng=rrng)
        clu.append({"age_band": band, **b})
    sel1 = select_one_ecg_per_patient(ptb, 20260922, rule="hash")
    sel2 = select_one_ecg_per_patient(ptb.sample(frac=1.0, random_state=9), 20260922, rule="hash")
    deterministic = set(sel1.record_id) == set(sel2.record_id)
    return {
        "formal": False,
        "smoke": True,
        "n_rows": int(len(ptb)),
        "n_patients": int(ptb.patient_id.nunique()),
        "selection_deterministic": deterministic,
        "record_status": rec,
        "cluster_status": clu,
        "historical_scientific_outcome_used": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "binding-freeze", "formal"], required=True)
    ap.add_argument("--protocol", default=str(ROOT / "configs" / "nature_upgrade_n3_protocol.yaml"))
    ap.add_argument("--n1-input", default=str(ROOT / "data" / "nature_upgrade_n1" / "N1_authoritative_development_rows.csv"))
    ap.add_argument("--ptbxl-meta", default=str(ROOT / "data" / "nature_upgrade_n3" / "ptbxl_database.csv"))
    ap.add_argument("--binding-manifest", default=str(ROOT / "configs" / "nature_upgrade_n3_binding_manifest.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "nature_upgrade_n3"))
    args = ap.parse_args()

    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    if args.mode == "smoke":
        result = synthetic_smoke(protocol)
    elif args.mode == "binding-freeze":
        result = binding_freeze(protocol_path, protocol, Path(args.n1_input), Path(args.ptbxl_meta), Path(args.binding_manifest).resolve())
    else:
        result = formal(protocol_path, protocol, Path(args.n1_input), Path(args.ptbxl_meta), Path(args.binding_manifest), Path(args.out_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
