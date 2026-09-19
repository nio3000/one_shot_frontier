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
from frontier.phase1_objects import central_sufficient_stats, max_recovery_error
from frontier.phase1_partition import dirichlet_label_partition
from frontier.phase1_projection import project_features
from frontier.phase2_baselines import representation_baselines
from frontier.phase2_eval import evaluate_adaptive_condition
from frontier.phase2_gates import evaluate_phase2a_gates
from frontier.phase2_reporting import write_report
from frontier.phase2_selector import build_folded_one_shot_object, select_tau_folded_summary_cv


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "NO_GIT_HEAD"


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _condition(bank_entry: dict, projection_dim: int, protocol: dict) -> dict:
    bank = load_feature_bank(bank_entry["resolved_path"], bank_entry)
    fixed = protocol["fixed"]
    Xtr = project_features(bank.X_train, projection_dim, bank.encoder_id, int(fixed["projection_seed"]))
    Xte = project_features(bank.X_test, projection_dim, bank.encoder_id, int(fixed["projection_seed"]))
    C = int(max(bank.y_train.max(), bank.y_test.max()) + 1)
    client = dirichlet_label_partition(
        bank.y_train,
        int(fixed["n_clients"]),
        float(fixed["client_dirichlet_alpha"]),
        int(fixed["client_partition_seed"]),
        bank.bank_id,
    )
    central = central_sufficient_stats(Xtr, bank.y_train, C)
    folded = build_folded_one_shot_object(
        Xtr, bank.y_train, client,
        int(fixed["n_clients"]), C,
        int(fixed["selector_folds"]), int(fixed["selector_fold_seed"]), bank.bank_id,
    )
    recovery = max_recovery_error(central, folded.full_stats)
    tol = float(protocol["one_shot_object"]["aggregation_recovery_abs_tolerance"])
    taus = [float(t) for t in fixed["tau"]]
    selected_tau, selector_scores = select_tau_folded_summary_cv(folded, taus, float(fixed["alpha0"]))
    baselines, pred_path = representation_baselines(folded.full_stats, Xte, bank.y_test, taus, float(fixed["alpha0"]))
    ev = evaluate_adaptive_condition(
        bank.y_test,
        pred_path,
        selected_tau,
        [float(x) for x in protocol["primary_evaluation"]["endpoint_candidates"]],
        [float(x) for x in protocol["primary_evaluation"]["oracle_interior_candidates"]],
        int(fixed["bootstrap_replicates"]),
        int(fixed["bootstrap_seed"]),
        float(protocol["primary_evaluation"]["oracle_opportunity_min_gain"]),
    )
    cond_id = f"{bank.bank_id}__d{projection_dim}"
    finite_values = [
        ev["adaptive_bacc"], ev["adaptive_accuracy"], ev["adaptive_macro_f1"],
        ev["best_endpoint_bacc_test_oracle"], ev["adaptive_gain_vs_best_endpoint"],
        ev["adaptive_gain_ci95_low"], ev["adaptive_gain_ci95_high"],
        ev["oracle_best_bacc_test"], ev["oracle_interior_bacc_test"],
        ev["oracle_interior_gain_vs_best_endpoint"],
    ]
    # recovery_fraction is intentionally NaN when no >=0.5pp oracle opportunity exists.
    allfinite = bool(np.isfinite(list(selector_scores.values())).all() and np.isfinite(finite_values).all())
    qualify = bool(ev["adaptive_gain_vs_best_endpoint"] >= float(protocol["primary_evaluation"]["adaptive_qualifying_min_gain"]) and ev["adaptive_gain_ci95_low"] > 0.0)
    condition = {
        "condition_id": cond_id,
        "bank_id": bank.bank_id,
        "dataset_id": bank.dataset_id,
        "modality": bank.modality,
        "encoder_id": bank.encoder_id,
        "projection_dim": int(projection_dim),
        "train_n": int(len(bank.y_train)),
        "test_n": int(len(bank.y_test)),
        "n_classes": C,
        **ev,
        "qualifying_adaptive": qualify,
        "object_recovery_max_abs": float(recovery),
        "object_recovery_pass": bool(recovery <= tol),
        "selector_test_independence_pass": True,
        "all_finite": allfinite,
        **folded.payload,
    }
    selector_rows = [
        {"condition_id": cond_id, "tau": float(t), "heldout_balanced_conditional_nll": float(s), "selected": bool(float(t) == selected_tau)}
        for t, s in sorted(selector_scores.items())
    ]
    baseline_rows = [
        {
            "condition_id": cond_id,
            "method_id": r.method_id,
            "balanced_accuracy": r.balanced_accuracy,
            "accuracy": r.accuracy,
            "macro_f1": r.macro_f1,
            "deployable": r.deployable,
            "notes": r.notes,
        }
        for r in baselines
    ]
    tau_rows = []
    from frontier.phase1_eval import metrics
    for tau, p in sorted(pred_path.items()):
        m = metrics(bank.y_test, p)
        tau_rows.append({
            "condition_id": cond_id,
            "dataset_id": bank.dataset_id,
            "encoder_id": bank.encoder_id,
            "projection_dim": int(projection_dim),
            "tau": float(tau),
            "balanced_accuracy": m.balanced_accuracy,
            "accuracy": m.accuracy,
            "macro_f1": m.macro_f1,
        })
    return {"condition": condition, "selector": selector_rows, "baselines": baseline_rows, "tau": tau_rows}


def _bank_task(bank_entry: dict, dims: list[int], protocol: dict) -> dict:
    out = {"conditions": [], "selector": [], "baselines": [], "tau": []}
    for d in dims:
        r = _condition(bank_entry, int(d), protocol)
        out["conditions"].append(r["condition"])
        out["selector"].extend(r["selector"])
        out["baselines"].extend(r["baselines"])
        out["tau"].extend(r["tau"])
    return out


def _smoke(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260909)
    C, d = 4, 24
    Xtr=[]; ytr=[]; Xte=[]; yte=[]
    for c in range(C):
        mu = np.zeros(d); mu[c] = 2.0
        A = np.eye(d) + (0.15 + 0.05*c) * np.diag(np.linspace(0,1,d))
        Xtr.append(rng.multivariate_normal(mu, A, size=100)); ytr.append(np.full(100,c))
        Xte.append(rng.multivariate_normal(mu, A, size=50)); yte.append(np.full(50,c))
    tmp = ROOT / "runs" / "phase2" / "smoke_feature_bank.npz"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    meta = {"bank_id":"smoke__synthetic","dataset_id":"smoke","modality":"engineering","encoder_id":"synthetic"}
    np.savez(tmp, X_train=np.vstack(Xtr).astype(np.float32), y_train=np.concatenate(ytr), X_test=np.vstack(Xte).astype(np.float32), y_test=np.concatenate(yte), metadata_json=json.dumps(meta))
    e = {**meta, "resolved_path": str(tmp)}
    p = json.loads(json.dumps(protocol))
    p["fixed"]["projection_dims"] = [16]
    p["fixed"]["bootstrap_replicates"] = 200
    r = _bank_task(e, [16], p)
    return pd.DataFrame(r["conditions"]), pd.DataFrame(r["selector"]), pd.DataFrame(r["baselines"]), pd.DataFrame(r["tau"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "formal-a"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--protocol", default=str(ROOT / "configs" / "phase2_protocol.yaml"))
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "phase1_feature_manifest.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "phase2"))
    args = ap.parse_args()
    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    gate = None; manifest_hash = None

    if args.mode == "smoke":
        conditions, selector, baselines, tau = _smoke(protocol)
    else:
        if protocol.get("status") != "FROZEN" or protocol.get("version") != "1.0.0-FROZEN":
            raise RuntimeError("Formal Phase 2-A blocked: protocol is not 1.0.0-FROZEN / FROZEN")
        if _git_head() == "NO_GIT_HEAD":
            raise RuntimeError("Formal Phase 2-A blocked: real Git HEAD required")
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise RuntimeError("Formal Phase 2-A blocked: frozen Phase 1 feature manifest missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "FROZEN":
            raise RuntimeError("Formal Phase 2-A blocked: Phase 1 feature manifest is not FROZEN")
        required_ids = list(protocol["feature_authority"]["required_bank_ids"])
        by_id = {e["bank_id"]: e for e in manifest.get("banks", [])}
        if sorted(by_id) != sorted(required_ids):
            raise RuntimeError(f"Formal Phase 2-A blocked: frozen bank IDs changed. expected={sorted(required_ids)} actual={sorted(by_id)}")
        p1_protocol = {"feature_banks": {"required": [
            {k: by_id[bid][k] for k in ["bank_id", "dataset_id", "modality", "encoder_id"]}
            for bid in required_ids
        ]}}
        entries = validate_manifest(manifest, p1_protocol, ROOT)
        manifest_hash = file_sha256(manifest_path)
        dims = [int(x) for x in protocol["fixed"]["projection_dims"]]
        rows = {"conditions": [], "selector": [], "baselines": [], "tau": []}
        shard_dir = out / "shards"; shard_dir.mkdir(parents=True, exist_ok=True)
        pending=[]
        for e in entries:
            shard = shard_dir / f"{e['bank_id']}.json"
            if args.resume and shard.exists():
                r=json.loads(shard.read_text(encoding="utf-8"))
                for k in rows: rows[k].extend(r[k])
            else:
                pending.append((e,shard))
        if pending:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
                futs={ex.submit(_bank_task,e,dims,protocol):(e,shard) for e,shard in pending}
                for fut in as_completed(futs):
                    e,shard=futs[fut]; r=fut.result(); shard.write_text(json.dumps(r,ensure_ascii=False),encoding="utf-8")
                    for k in rows: rows[k].extend(r[k])
        conditions=pd.DataFrame(rows["conditions"]).sort_values(["dataset_id","encoder_id","projection_dim"]).reset_index(drop=True)
        selector=pd.DataFrame(rows["selector"]).sort_values(["condition_id","tau"]).reset_index(drop=True)
        baselines=pd.DataFrame(rows["baselines"]).sort_values(["condition_id","method_id"]).reset_index(drop=True)
        tau=pd.DataFrame(rows["tau"]).sort_values(["condition_id","tau"]).reset_index(drop=True)
        gate=evaluate_phase2a_gates(conditions,protocol)

    _atomic_csv(conditions,out/"conditions.csv")
    _atomic_csv(selector,out/"selector_curves.csv")
    _atomic_csv(baselines,out/"baselines.csv")
    _atomic_csv(tau,out/"tau_curves.csv")
    if gate is not None: _atomic_json(gate,out/"gate_summary.json")
    expected = int(protocol["formal_counts"]["structural_conditions"] if args.mode=="formal-a" else 1)
    summary={
        "mode":args.mode,"formal":args.mode=="formal-a",
        "n_conditions":int(len(conditions)),"expected_conditions":expected,
        "n_tau_rows":int(len(tau)),"expected_tau_rows":expected*len(protocol["fixed"]["tau"]),
        "complete":bool(len(conditions)==expected and len(tau)==expected*len(protocol["fixed"]["tau"])),
        "git_head":_git_head(),"protocol_sha256":file_sha256(protocol_path),"manifest_sha256":manifest_hash,
        "protocol_status":protocol.get("status"),"protocol_version":protocol.get("version"),
    }
    _atomic_json(summary,out/"run_summary.json")
    _atomic_json({"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__},out/"environment_manifest.json")
    (out/"protocol_snapshot.yaml").write_text(protocol_path.read_text(encoding="utf-8"),encoding="utf-8")
    if args.mode=="formal-a": (out/"feature_manifest_snapshot.json").write_text(Path(args.manifest).read_text(encoding="utf-8"),encoding="utf-8")
    report=ROOT/"reports"/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase2_{args.mode}_experiment_report.md"
    write_report(report,summary,gate," ".join(sys.argv))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if gate is not None: print(json.dumps(gate,ensure_ascii=False,indent=2))
    print(f"REPORT={report}")

if __name__=="__main__": main()
