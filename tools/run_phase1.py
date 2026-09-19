from __future__ import annotations

import argparse
import hashlib
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import subprocess
import sys
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase1_eval import crossfit_interior_vs_endpoints, metrics, stratified_paired_bootstrap_bacc_delta, tau_0_5
from frontier.phase1_featurebank import file_sha256, load_feature_bank, validate_manifest
from frontier.phase1_gates import evaluate_phase1_gates
from frontier.phase1_models import predict_diagonal_gaussian, predict_ncm, predict_tau_path
from frontier.phase1_objects import central_sufficient_stats, federated_sufficient_stats, max_recovery_error
from frontier.phase1_partition import dirichlet_label_partition
from frontier.phase1_projection import project_features
from frontier.phase1_reporting import write_report


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
    client = dirichlet_label_partition(bank.y_train, int(fixed["n_clients"]), float(fixed["client_dirichlet_alpha"]), int(fixed["client_partition_seed"]), bank.bank_id)
    central = central_sufficient_stats(Xtr, bank.y_train, C)
    fed, payload = federated_sufficient_stats(Xtr, bank.y_train, client, int(fixed["n_clients"]), C)
    recovery = max_recovery_error(central, fed)
    tol = float(protocol["federated_object"]["aggregation_recovery_abs_tolerance"])
    taus = [float(t) for t in fixed["tau"]]
    pred = predict_tau_path(fed, Xte, taus, float(fixed["alpha0"]))
    t05, best_bacc, score_map = tau_0_5(bank.y_test, pred, float(protocol["primary_evaluation"]["tau_0_5_tolerance"]))
    cf = crossfit_interior_vs_endpoints(
        bank.y_test, pred,
        [float(t) for t in protocol["primary_evaluation"]["interior_candidates"]],
        [float(t) for t in protocol["primary_evaluation"]["endpoint_candidates"]],
        int(fixed["test_crossfit_seed"]),
    )
    ci_lo, ci_hi = stratified_paired_bootstrap_bacc_delta(
        bank.y_test, cf["pred_interior"], cf["pred_endpoint"], int(fixed["bootstrap_replicates"]), int(fixed["bootstrap_seed"])
    )
    ncm = metrics(bank.y_test, predict_ncm(fed, Xte))
    diag = metrics(bank.y_test, predict_diagonal_gaussian(fed, Xte, float(fixed["alpha0"])))
    threshold = float(protocol["phase1_gates"]["P1"]["min_crossfit_gain"])
    qualify = bool(cf["delta_balanced_accuracy"] >= threshold and ci_lo > 0.0)
    cond_id = f"{bank.bank_id}__d{projection_dim}"
    condition = {
        "condition_id": cond_id,
        "bank_id": bank.bank_id,
        "dataset_id": bank.dataset_id,
        "modality": bank.modality,
        "encoder_id": bank.encoder_id,
        "source_feature_dim": int(bank.X_train.shape[1]),
        "projection_dim": int(projection_dim),
        "train_n": int(len(bank.y_train)),
        "test_n": int(len(bank.y_test)),
        "n_classes": C,
        "tau_0_5": float(t05),
        "best_full_test_bacc": float(best_bacc),
        "crossfit_interior_bacc": float(cf["interior_metrics"].balanced_accuracy),
        "crossfit_endpoint_bacc": float(cf["endpoint_metrics"].balanced_accuracy),
        "crossfit_gain": float(cf["delta_balanced_accuracy"]),
        "crossfit_gain_ci95_low": float(ci_lo),
        "crossfit_gain_ci95_high": float(ci_hi),
        "qualifying_interior": qualify,
        "selected_interior_tau_fold1": float(cf["selected"][0]["interior_tau"]),
        "selected_endpoint_tau_fold1": float(cf["selected"][0]["endpoint_tau"]),
        "selected_interior_tau_fold2": float(cf["selected"][1]["interior_tau"]),
        "selected_endpoint_tau_fold2": float(cf["selected"][1]["endpoint_tau"]),
        "object_recovery_max_abs": float(recovery),
        "object_recovery_pass": bool(recovery <= tol),
        "all_finite": bool(np.isfinite(list(score_map.values())).all()),
    }
    curves = []
    for tau in taus:
        m = metrics(bank.y_test, pred[tau])
        curves.append({
            "condition_id": cond_id, "bank_id": bank.bank_id, "dataset_id": bank.dataset_id,
            "modality": bank.modality, "encoder_id": bank.encoder_id, "projection_dim": int(projection_dim),
            "tau": float(tau), "balanced_accuracy": m.balanced_accuracy, "accuracy": m.accuracy, "macro_f1": m.macro_f1,
        })
    baselines = [
        {"condition_id": cond_id, "baseline": "ncm", "balanced_accuracy": ncm.balanced_accuracy, "accuracy": ncm.accuracy, "macro_f1": ncm.macro_f1},
        {"condition_id": cond_id, "baseline": "diagonal_class_gaussian", "balanced_accuracy": diag.balanced_accuracy, "accuracy": diag.accuracy, "macro_f1": diag.macro_f1},
        {"condition_id": cond_id, "baseline": "shared_covariance_tau0", "balanced_accuracy": score_map[0.0]},
        {"condition_id": cond_id, "baseline": "class_specific_tau1", "balanced_accuracy": score_map[1.0]},
    ]
    objects = [{"condition_id": cond_id, "bank_id": bank.bank_id, "projection_dim": int(projection_dim), **payload, "object_recovery_max_abs": float(recovery)}]
    return {"condition": condition, "curves": curves, "baselines": baselines, "objects": objects}


def _bank_task(bank_entry: dict, dims: list[int], protocol: dict) -> dict:
    out = {"conditions": [], "curves": [], "baselines": [], "objects": []}
    for d in dims:
        r = _condition(bank_entry, int(d), protocol)
        out["conditions"].append(r["condition"])
        out["curves"].extend(r["curves"])
        out["baselines"].extend(r["baselines"])
        out["objects"].extend(r["objects"])
    return out


def _smoke(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260908)
    C, d0 = 4, 32
    ntr, nte = 80, 40
    means = rng.normal(size=(C, d0)) * 1.2
    Xtr, ytr, Xte, yte = [], [], [], []
    for c in range(C):
        cov = np.eye(d0) + 0.2 * np.diag(np.linspace(0, c + 1, d0))
        Xtr.append(rng.multivariate_normal(means[c], cov, size=ntr))
        ytr.append(np.full(ntr, c))
        Xte.append(rng.multivariate_normal(means[c], cov, size=nte))
        yte.append(np.full(nte, c))
    tmp = ROOT / "runs" / "phase1" / "smoke_feature_bank.npz"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    meta = {"bank_id": "smoke__synthetic", "dataset_id": "smoke", "modality": "engineering", "encoder_id": "synthetic"}
    np.savez(tmp, X_train=np.vstack(Xtr).astype(np.float32), y_train=np.concatenate(ytr), X_test=np.vstack(Xte).astype(np.float32), y_test=np.concatenate(yte), metadata_json=json.dumps(meta))
    e = {**meta, "resolved_path": str(tmp)}
    # Override projection dimensions to fit synthetic feature dimension.
    p = json.loads(json.dumps(protocol))
    p["fixed"]["projection_dims"] = [16]
    p["fixed"]["bootstrap_replicates"] = 200
    r = _bank_task(e, [16], p)
    return pd.DataFrame(r["conditions"]), pd.DataFrame(r["curves"]), pd.DataFrame(r["baselines"]), pd.DataFrame(r["objects"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "formal"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--protocol", default=str(ROOT / "configs" / "phase1_protocol.yaml"))
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "phase1_feature_manifest.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "phase1"))
    args = ap.parse_args()
    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate = None
    manifest_hash = None

    if args.mode == "smoke":
        conditions, curves, baselines, objects = _smoke(protocol)
    else:
        if protocol.get("status") != "FROZEN" or protocol.get("version") != "1.0.0-FROZEN":
            raise RuntimeError("Formal Phase 1 blocked: protocol is not 1.0.0-FROZEN / FROZEN")
        if _git_head() == "NO_GIT_HEAD":
            raise RuntimeError("Formal Phase 1 blocked: real Git HEAD required")
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise RuntimeError("Formal Phase 1 blocked: frozen feature-bank manifest missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = validate_manifest(manifest, protocol, ROOT)
        manifest_hash = file_sha256(manifest_path)
        dims = [int(x) for x in protocol["fixed"]["projection_dims"]]
        rows = {"conditions": [], "curves": [], "baselines": [], "objects": []}
        shard_dir = out / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for e in entries:
            shard = shard_dir / f"{e['bank_id']}.json"
            if args.resume and shard.exists():
                r = json.loads(shard.read_text(encoding="utf-8"))
                for k in rows: rows[k].extend(r[k])
            else:
                pending.append((e, shard))
        if pending:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
                futs = {ex.submit(_bank_task, e, dims, protocol): (e, shard) for e, shard in pending}
                for fut in as_completed(futs):
                    e, shard = futs[fut]
                    r = fut.result()
                    shard.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
                    for k in rows: rows[k].extend(r[k])
        conditions = pd.DataFrame(rows["conditions"])
        curves = pd.DataFrame(rows["curves"])
        baselines = pd.DataFrame(rows["baselines"])
        objects = pd.DataFrame(rows["objects"])
        conditions = conditions.sort_values(["dataset_id", "encoder_id", "projection_dim"]).reset_index(drop=True)
        curves = curves.sort_values(["dataset_id", "encoder_id", "projection_dim", "tau"]).reset_index(drop=True)
        gate = evaluate_phase1_gates(conditions, protocol)

    _atomic_csv(conditions, out / "conditions.csv")
    _atomic_csv(curves, out / "tau_curves.csv")
    _atomic_csv(conditions[[c for c in conditions.columns if c.startswith("crossfit") or c.startswith("selected") or c in ["condition_id", "bank_id", "dataset_id", "encoder_id", "projection_dim", "qualifying_interior"]]], out / "crossfit_results.csv")
    _atomic_csv(baselines, out / "baselines.csv")
    _atomic_csv(objects, out / "object_metrics.csv")
    if gate is not None:
        _atomic_json(gate, out / "gate_summary.json")
    expected_conditions = int(protocol["formal_counts"]["structural_conditions"] if args.mode == "formal" else 1)
    expected_curves = expected_conditions * len(protocol["fixed"]["tau"])
    summary = {
        "mode": args.mode,
        "formal": args.mode == "formal",
        "n_conditions": int(len(conditions)),
        "expected_conditions": expected_conditions,
        "n_curve_rows": int(len(curves)),
        "expected_curve_rows": int(expected_curves),
        "complete": bool(len(conditions) == expected_conditions and len(curves) == expected_curves),
        "git_head": _git_head(),
        "protocol_sha256": file_sha256(protocol_path),
        "manifest_sha256": manifest_hash,
        "protocol_status": protocol.get("status"),
        "protocol_version": protocol.get("version"),
    }
    _atomic_json(summary, out / "run_summary.json")
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn
        env["sklearn"] = sklearn.__version__
    except Exception:
        env["sklearn"] = "unknown"
    _atomic_json(env, out / "environment_manifest.json")
    (out / "protocol_snapshot.yaml").write_text(protocol_path.read_text(encoding="utf-8"), encoding="utf-8")
    if args.mode == "formal":
        (out / "feature_manifest_snapshot.json").write_text(Path(args.manifest).read_text(encoding="utf-8"), encoding="utf-8")
    report = ROOT / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase1_{args.mode}_experiment_report.md"
    write_report(report, summary, gate, " ".join(sys.argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if gate is not None:
        print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"REPORT={report}")


if __name__ == "__main__":
    main()
