from __future__ import annotations

import argparse
import hashlib
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase1_featurebank import file_sha256, load_feature_bank, validate_manifest
from frontier.phase1_models import covariance_family
from frontier.phase1_objects import central_sufficient_stats, federated_sufficient_stats
from frontier.phase1_partition import dirichlet_label_partition
from frontier.phase1_projection import project_features
from frontier.phase4_interventions import class_means, reflection_mask, reflect_by_class, second_order_equivalence
from frontier.phase4_escape_summaries import (
    rademacher_directions,
    diagonal_third_summary,
    rademacher_third_summary,
    rademacher_fourth_summary,
    federated_raw_power_sum,
    central_raw_power_sum,
    normalized_summary_distance,
    normalized_raw_aggregation_error,
)
from frontier.phase4_eval import paired_risk_curves, pair_decision_metrics
from frontier.phase4_gates import evaluate_phase4_gates
from frontier.phase4_reporting import write_phase4_report


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "NO_GIT_HEAD"


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _normalized_max_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b)) / denom)


def _head_parameter_diff(stats_a, stats_b, taus: list[float], alpha0: float) -> float:
    diff = max(
        _normalized_max_diff(stats_a.means, stats_b.means),
        _normalized_max_diff(stats_a.priors, stats_b.priors),
    )
    for tau in taus:
        ca = covariance_family(stats_a, float(tau), float(alpha0))
        cb = covariance_family(stats_b, float(tau), float(alpha0))
        diff = max(diff, _normalized_max_diff(ca, cb))
    return float(diff)


def _candidate_metrics(
    X: np.ndarray,
    Xcf: np.ndarray,
    y: np.ndarray,
    client_ids: np.ndarray,
    n_clients: int,
    C: int,
    k: int,
    seed: int,
    base_t2_bytes: int,
) -> dict[str, float]:
    d = X.shape[1]
    I = np.eye(d, dtype=np.float64)
    R = rademacher_directions(d, int(k), int(seed))

    diag_a = diagonal_third_summary(X, y, C)
    diag_b = diagonal_third_summary(Xcf, y, C)
    sk3_a = rademacher_third_summary(X, y, C, int(k), int(seed))
    sk3_b = rademacher_third_summary(Xcf, y, C, int(k), int(seed))
    sk4_a = rademacher_fourth_summary(X, y, C, int(k), int(seed))
    sk4_b = rademacher_fourth_summary(Xcf, y, C, int(k), int(seed))

    diag_c = central_raw_power_sum(X, y, C, I, 3)
    diag_f, present_pairs = federated_raw_power_sum(X, y, client_ids, n_clients, C, I, 3)
    sk_c = central_raw_power_sum(X, y, C, R, 3)
    sk_f, present_pairs_2 = federated_raw_power_sum(X, y, client_ids, n_clients, C, R, 3)
    if present_pairs != present_pairs_2:
        raise RuntimeError("present-pair accounting mismatch")

    diag_extra = int(present_pairs) * int(d) * 4
    sketch_extra = int(present_pairs) * int(k) * 4
    return {
        "diag_m3_distance": normalized_summary_distance(diag_a, diag_b),
        "sketch_m3_distance": normalized_summary_distance(sk3_a, sk3_b),
        "sketch_m4_distance": normalized_summary_distance(sk4_a, sk4_b),
        "diag_m3_aggregation_norm_error": normalized_raw_aggregation_error(diag_c, diag_f),
        "sketch_m3_aggregation_norm_error": normalized_raw_aggregation_error(sk_c, sk_f),
        "diag_m3_upload_ratio_vs_t2": float((base_t2_bytes + diag_extra) / max(base_t2_bytes, 1)),
        "sketch_m3_upload_ratio_vs_t2": float((base_t2_bytes + sketch_extra) / max(base_t2_bytes, 1)),
        "present_client_class_pairs": int(present_pairs),
    }


def _condition(bank_entry: dict, projection_dim: int, protocol: dict) -> dict:
    bank = load_feature_bank(bank_entry["resolved_path"], bank_entry)
    f = protocol["fixed"]
    Xtr = project_features(bank.X_train, int(projection_dim), bank.encoder_id, int(f["projection_seed"]))
    Xte = project_features(bank.X_test, int(projection_dim), bank.encoder_id, int(f["projection_seed"]))
    ytr = bank.y_train
    yte = bank.y_test
    C = int(max(ytr.max(), yte.max()) + 1)
    taus = [float(x) for x in f["tau_grid"]]
    alpha0 = float(f["alpha0"])

    means = class_means(Xtr, ytr, C)
    stats_orig = central_sufficient_stats(Xtr, ytr, C)
    client_ids = dirichlet_label_partition(
        ytr,
        int(f["n_clients"]),
        float(f["client_dirichlet_alpha"]),
        int(f["client_partition_seed"]),
        bank.bank_id,
    )
    fed_stats, comm = federated_sufficient_stats(
        Xtr, ytr, client_ids, int(f["n_clients"]), C
    )
    # The existing Phase-1 object route remains the baseline communication authority.
    recovery = max(
        _normalized_max_diff(stats_orig.sums, fed_stats.sums),
        _normalized_max_diff(stats_orig.second, fed_stats.second),
        _normalized_max_diff(stats_orig.means, fed_stats.means),
        _normalized_max_diff(stats_orig.class_covs, fed_stats.class_covs),
    )

    condition_id = f"{bank.bank_id}__d{projection_dim}"
    pair_rows = []
    curve_rows = []

    for pattern in protocol["interventions"]["patterns"]:
        mask = reflection_mask(C, str(pattern), int(f["reflection_seed"]), condition_id)
        Xtr_cf = reflect_by_class(Xtr, ytr, means, mask)
        Xte_cf = reflect_by_class(Xte, yte, means, mask)
        stats_cf = central_sufficient_stats(Xtr_cf, ytr, C)
        eq = second_order_equivalence(Xtr, Xtr_cf, ytr, C)
        head_diff = _head_parameter_diff(stats_orig, stats_cf, taus, alpha0)

        curves = paired_risk_curves(
            stats_orig,
            {
                "original": (Xte, yte),
                "counterfactual": (Xte_cf, yte),
            },
            taus,
            alpha0,
        )
        dm = pair_decision_metrics(curves["original"], curves["counterfactual"], taus)
        cm = _candidate_metrics(
            Xtr,
            Xtr_cf,
            ytr,
            client_ids,
            int(f["n_clients"]),
            C,
            int(protocol["escape_candidates"]["rademacher_m3_k16"]["k"]),
            int(protocol["escape_candidates"]["rademacher_m3_k16"]["seed"]),
            int(comm["class_specific_second_moment_bytes_float32"]),
        )

        pair_id = f"{condition_id}__{pattern}"
        pair_rows.append({
            "pair_id": pair_id,
            "condition_id": condition_id,
            "bank_id": bank.bank_id,
            "dataset_id": bank.dataset_id,
            "modality": bank.modality,
            "encoder_id": bank.encoder_id,
            "projection_dim": int(projection_dim),
            "intervention": str(pattern),
            "n_reflected_classes": int(mask.sum()),
            "n_classes": int(C),
            "counts_equal": bool(eq["counts_equal"]),
            "t2_comm_norm_diff": float(eq["normalized_communication_diff"]),
            "t2_model_norm_diff": float(eq["normalized_model_parameter_diff"]),
            "max_head_parameter_norm_diff": float(head_diff),
            "phase1_object_recovery_norm_error": float(recovery),
            **dm,
            **cm,
        })
        for task_name, curve in curves.items():
            for tau, bacc in zip(taus, curve):
                curve_rows.append({
                    "pair_id": pair_id,
                    "condition_id": condition_id,
                    "dataset_id": bank.dataset_id,
                    "encoder_id": bank.encoder_id,
                    "projection_dim": int(projection_dim),
                    "intervention": str(pattern),
                    "task": task_name,
                    "tau": float(tau),
                    "balanced_accuracy": float(bacc),
                })

    return {"pairs": pair_rows, "curves": curve_rows}


def _bank_task(bank_entry: dict, dims: list[int], protocol: dict) -> dict:
    out = {"pairs": [], "curves": []}
    for d in dims:
        r = _condition(bank_entry, int(d), protocol)
        out["pairs"].extend(r["pairs"])
        out["curves"].extend(r["curves"])
    return out


def _smoke(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(20260910)
    C, d = 3, 12
    Xtr=[]; ytr=[]; Xte=[]; yte=[]
    for c in range(C):
        # Deliberately skewed but numerically simple real-valued engineering fixture.
        z = rng.lognormal(mean=0.05*c, sigma=0.45+0.05*c, size=(180, d))
        z[:, :2] += 1.5*c
        Xtr.append(z); ytr.append(np.full(len(z), c))
        z2 = rng.lognormal(mean=0.05*c, sigma=0.45+0.05*c, size=(90, d))
        z2[:, :2] += 1.5*c
        Xte.append(z2); yte.append(np.full(len(z2), c))
    Xtr=np.vstack(Xtr).astype(np.float32); ytr=np.concatenate(ytr)
    Xte=np.vstack(Xte).astype(np.float32); yte=np.concatenate(yte)
    tmp = ROOT / "runs" / "phase4" / "smoke_bank.npz"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    meta={"bank_id":"phase4_smoke","dataset_id":"smoke","modality":"engineering","encoder_id":"synthetic"}
    np.savez(tmp,X_train=Xtr,y_train=ytr,X_test=Xte,y_test=yte,metadata_json=json.dumps(meta))
    p=json.loads(json.dumps(protocol))
    p["fixed"]["projection_seed"] = 20260910
    entry={**meta,"resolved_path":str(tmp)}
    # Bypass the Phase-1 projector because smoke feature dimension already equals target dimension.
    bank = load_feature_bank(tmp, entry)
    # Create a temporary entry with a monkey-sized direct projection via output dimension d.
    r=_condition(entry,d,p)
    pairs=pd.DataFrame(r["pairs"])
    curves=pd.DataFrame(r["curves"])
    smoke_integrity={
        "n_pair_rows":int(len(pairs)),
        "n_curve_rows":int(len(curves)),
        "max_t2_comm_norm_diff":float(pairs["t2_comm_norm_diff"].max()),
        "max_t2_model_norm_diff":float(pairs["t2_model_norm_diff"].max()),
        "max_head_parameter_norm_diff":float(pairs["max_head_parameter_norm_diff"].max()),
        "max_sketch_m4_distance":float(pairs["sketch_m4_distance"].max()),
        "all_finite":bool(np.isfinite(pairs.select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    return pairs, curves, smoke_integrity


def _authority_guard(protocol: dict) -> dict[str, str]:
    hashes = {}
    for key in ["nature_direction_freeze", "external_validation_v2", "phase3_theory_authority"]:
        path = ROOT / protocol["authorities"][key]
        if not path.exists():
            raise RuntimeError(f"Phase 4 blocked: authority missing: {path}")
        hashes[key] = _sha256(path)
    return hashes


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["smoke","formal-development"],required=True)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--protocol",default=str(ROOT/"configs"/"phase4_protocol.yaml"))
    ap.add_argument("--manifest",default=str(ROOT/"configs"/"phase1_feature_manifest.json"))
    ap.add_argument("--out-dir",default=str(ROOT/"runs"/"phase4"))
    args=ap.parse_args()

    protocol_path=Path(args.protocol)
    protocol=yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    authority_hashes=_authority_guard(protocol)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    gate=None; manifest_hash=None; smoke_integrity=None

    if args.mode=="smoke":
        pairs,curves,smoke_integrity=_smoke(protocol)
    else:
        if protocol.get("version")!="1.0.0-FROZEN" or protocol.get("status")!="FROZEN":
            raise RuntimeError("Phase 4 formal-development blocked: protocol not 1.0.0-FROZEN / FROZEN")
        if _git_head()=="NO_GIT_HEAD":
            raise RuntimeError("Phase 4 formal-development blocked: real Git HEAD required")
        manifest_path=Path(args.manifest)
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status")!="FROZEN":
            raise RuntimeError("Phase 4 blocked: Phase 1 feature manifest not FROZEN")
        required_ids=list(protocol["development_authority"]["required_bank_ids"])
        by_id={e["bank_id"]:e for e in manifest.get("banks",[])}
        if sorted(by_id)!=sorted(required_ids):
            raise RuntimeError("Phase 4 blocked: development bank IDs changed")
        p1_protocol={"feature_banks":{"required":[{k:by_id[bid][k] for k in ["bank_id","dataset_id","modality","encoder_id"]} for bid in required_ids]}}
        entries=validate_manifest(manifest,p1_protocol,ROOT)
        manifest_hash=_sha256(manifest_path)
        dims=[int(x) for x in protocol["fixed"]["projection_dims"]]

        rows={"pairs":[],"curves":[]}
        shard_dir=out/"shards"; shard_dir.mkdir(parents=True,exist_ok=True)
        pending=[]
        for e in entries:
            shard=shard_dir/f"{e['bank_id']}.json"
            if args.resume and shard.exists():
                r=json.loads(shard.read_text(encoding="utf-8"))
                rows["pairs"].extend(r["pairs"]); rows["curves"].extend(r["curves"])
            else:
                pending.append((e,shard))
        if pending:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
                futs={ex.submit(_bank_task,e,dims,protocol):(e,shard) for e,shard in pending}
                for fut in as_completed(futs):
                    e,shard=futs[fut]
                    r=fut.result()
                    shard.write_text(json.dumps(r,ensure_ascii=False),encoding="utf-8")
                    rows["pairs"].extend(r["pairs"]); rows["curves"].extend(r["curves"])
        pairs=pd.DataFrame(rows["pairs"]).sort_values(["dataset_id","encoder_id","projection_dim","intervention"]).reset_index(drop=True)
        curves=pd.DataFrame(rows["curves"]).sort_values(["pair_id","task","tau"]).reset_index(drop=True)
        gate=evaluate_phase4_gates(pairs,protocol)

    pairs.to_csv(out/"pair_diagnostics.csv",index=False)
    curves.to_csv(out/"risk_curves.csv",index=False)
    if gate is not None:
        (out/"gate_summary.json").write_text(json.dumps(gate,indent=2),encoding="utf-8")
    if smoke_integrity is not None:
        (out/"smoke_integrity.json").write_text(json.dumps(smoke_integrity,indent=2),encoding="utf-8")

    expected_pairs=2 if args.mode=="smoke" else int(protocol["formal_counts"]["pair_rows"])
    expected_curves=expected_pairs*2*len(protocol["fixed"]["tau_grid"])
    summary={
        "mode":args.mode,
        "formal_development":args.mode=="formal-development",
        "n_pair_rows":int(len(pairs)),
        "expected_pair_rows":int(expected_pairs),
        "n_curve_rows":int(len(curves)),
        "expected_curve_rows":int(expected_curves),
        "complete":bool(len(pairs)==expected_pairs and len(curves)==expected_curves),
        "git_head":_git_head(),
        "protocol_sha256":_sha256(protocol_path),
        "manifest_sha256":manifest_hash,
        "authority_sha256":authority_hashes,
        "protocol_status":protocol.get("status"),
        "protocol_version":protocol.get("version"),
        "external_outcome_blackout":"ACTIVE",
    }
    (out/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (out/"environment_manifest.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__},indent=2),encoding="utf-8")
    (out/"protocol_snapshot.yaml").write_text(protocol_path.read_text(encoding="utf-8"),encoding="utf-8")

    report_dir=ROOT/"reports"/"phase4"
    report=report_dir/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase4_{args.mode}_experiment_report.md"
    write_phase4_report(report,summary,gate," ".join(sys.argv))

    print(json.dumps(summary,indent=2))
    if smoke_integrity is not None:
        print(json.dumps(smoke_integrity,indent=2))
    if gate is not None:
        print(json.dumps(gate,indent=2))
    print(f"REPORT={report}")


if __name__=="__main__":
    main()
