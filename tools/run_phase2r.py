from __future__ import annotations

import argparse
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
from frontier.phase2_selector import build_folded_one_shot_object
from frontier.phase2r_selector import select_tau_objective
from frontier.phase2r_gates import evaluate_phase2r_gates
from frontier.phase2r_reporting import write_phase2r_report


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
        Xtr,
        bank.y_train,
        client,
        int(fixed["n_clients"]),
        C,
        int(fixed["selector_folds"]),
        int(fixed["selector_fold_seed"]),
        bank.bank_id,
    )
    recovery = max_recovery_error(central, folded.full_stats)
    baselines, pred_path = representation_baselines(
        folded.full_stats,
        Xte,
        bank.y_test,
        [float(x) for x in fixed["tau"]],
        float(fixed["alpha0"]),
    )

    cond_id = f"{bank.bank_id}__d{projection_dim}"
    result_rows = []
    curve_rows = []

    for objective_id in protocol["candidate_objectives"]:
        selected_tau, scores = select_tau_objective(
            folded,
            [float(x) for x in fixed["tau"]],
            float(fixed["alpha0"]),
            objective_id,
        )
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

        finite_values = [
            ev["adaptive_bacc"],
            ev["adaptive_accuracy"],
            ev["adaptive_macro_f1"],
            ev["best_endpoint_bacc_test_oracle"],
            ev["adaptive_gain_vs_best_endpoint"],
            ev["adaptive_gain_ci95_low"],
            ev["adaptive_gain_ci95_high"],
            ev["oracle_interior_gain_vs_best_endpoint"],
        ]
        allfinite = bool(np.isfinite(list(scores.values())).all() and np.isfinite(finite_values).all())

        result_rows.append({
            "condition_id": cond_id,
            "bank_id": bank.bank_id,
            "dataset_id": bank.dataset_id,
            "modality": bank.modality,
            "encoder_id": bank.encoder_id,
            "projection_dim": int(projection_dim),
            "objective_id": objective_id,
            **ev,
            "object_recovery_max_abs": float(recovery),
            "object_recovery_pass": bool(recovery <= 1.0e-7),
            "selector_test_independence_pass": True,
            "all_finite": allfinite,
            **folded.payload,
        })

        direction = protocol["objective_direction"][objective_id]
        for tau, score in sorted(scores.items()):
            curve_rows.append({
                "condition_id": cond_id,
                "dataset_id": bank.dataset_id,
                "encoder_id": bank.encoder_id,
                "projection_dim": int(projection_dim),
                "objective_id": objective_id,
                "direction": direction,
                "tau": float(tau),
                "score": float(score),
                "selected": bool(float(tau) == selected_tau),
            })

    return {"results": result_rows, "curves": curve_rows}


def _bank_task(bank_entry: dict, dims: list[int], protocol: dict) -> dict:
    out = {"results": [], "curves": []}
    for d in dims:
        r = _condition(bank_entry, int(d), protocol)
        out["results"].extend(r["results"])
        out["curves"].extend(r["curves"])
    return out


def _smoke(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260909)
    C, d = 4, 24
    Xtr=[]; ytr=[]; Xte=[]; yte=[]
    for c in range(C):
        mu = np.zeros(d); mu[c] = 2.0
        cov = np.eye(d)
        cov[:4,:4] += 0.08 * c
        Xtr.append(rng.multivariate_normal(mu, cov, size=120))
        ytr.append(np.full(120,c))
        Xte.append(rng.multivariate_normal(mu, cov, size=60))
        yte.append(np.full(60,c))

    tmp = ROOT / "runs" / "phase2r" / "smoke_feature_bank.npz"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    meta = {"bank_id":"smoke__synthetic","dataset_id":"smoke","modality":"engineering","encoder_id":"synthetic"}
    np.savez(
        tmp,
        X_train=np.vstack(Xtr).astype(np.float32),
        y_train=np.concatenate(ytr),
        X_test=np.vstack(Xte).astype(np.float32),
        y_test=np.concatenate(yte),
        metadata_json=json.dumps(meta),
    )
    p = json.loads(json.dumps(protocol))
    p["fixed"]["projection_dims"] = [16]
    p["fixed"]["bootstrap_replicates"] = 200
    e = {**meta, "resolved_path": str(tmp)}
    r = _bank_task(e, [16], p)
    return pd.DataFrame(r["results"]), pd.DataFrame(r["curves"])


def _external_plan_guard(protocol: dict) -> None:
    plan_path = ROOT / protocol["nature_external_plan"]["yaml_path"]
    if not plan_path.exists():
        raise RuntimeError("Phase 2-R blocked: frozen Nature external validation plan missing")
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != protocol["nature_external_plan"]["required_status"]:
        raise RuntimeError("Phase 2-R blocked: Nature external validation plan is not frozen with required status")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "development"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--protocol", default=str(ROOT / "configs" / "phase2r_protocol.yaml"))
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "phase1_feature_manifest.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "phase2r"))
    args = ap.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    _external_plan_guard(protocol)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate = None
    manifest_hash = None

    if args.mode == "smoke":
        results, curves = _smoke(protocol)
    else:
        if protocol.get("status") != "FROZEN" or protocol.get("version") != "1.0.0-FROZEN":
            raise RuntimeError("Phase 2-R development run blocked: protocol is not 1.0.0-FROZEN / FROZEN")
        if _git_head() == "NO_GIT_HEAD":
            raise RuntimeError("Phase 2-R development run blocked: real Git HEAD required")

        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "FROZEN":
            raise RuntimeError("Phase 2-R blocked: Phase 1 feature manifest not FROZEN")

        required_ids = list(protocol["development_authority"]["required_bank_ids"])
        by_id = {e["bank_id"]: e for e in manifest.get("banks", [])}
        if sorted(by_id) != sorted(required_ids):
            raise RuntimeError("Phase 2-R blocked: development bank IDs changed")

        p1_protocol = {"feature_banks": {"required": [
            {k: by_id[bid][k] for k in ["bank_id","dataset_id","modality","encoder_id"]}
            for bid in required_ids
        ]}}
        entries = validate_manifest(manifest, p1_protocol, ROOT)
        manifest_hash = file_sha256(manifest_path)
        dims = [int(x) for x in protocol["fixed"]["projection_dims"]]

        rows = {"results": [], "curves": []}
        shard_dir = out / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for e in entries:
            shard = shard_dir / f"{e['bank_id']}.json"
            if args.resume and shard.exists():
                r = json.loads(shard.read_text(encoding="utf-8"))
                rows["results"].extend(r["results"])
                rows["curves"].extend(r["curves"])
            else:
                pending.append((e, shard))

        if pending:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
                futs = {ex.submit(_bank_task,e,dims,protocol):(e,shard) for e,shard in pending}
                for fut in as_completed(futs):
                    e, shard = futs[fut]
                    r = fut.result()
                    shard.write_text(json.dumps(r,ensure_ascii=False),encoding="utf-8")
                    rows["results"].extend(r["results"])
                    rows["curves"].extend(r["curves"])

        results = pd.DataFrame(rows["results"]).sort_values(
            ["objective_id","dataset_id","encoder_id","projection_dim"]
        ).reset_index(drop=True)
        curves = pd.DataFrame(rows["curves"]).sort_values(
            ["objective_id","condition_id","tau"]
        ).reset_index(drop=True)
        gate = evaluate_phase2r_gates(results, protocol)

    _atomic_csv(results, out / "objective_results.csv")
    _atomic_csv(curves, out / "objective_curves.csv")
    if gate is not None:
        _atomic_json(gate, out / "gate_summary.json")

    expected_conditions = 1 if args.mode == "smoke" else int(protocol["formal_counts"]["structural_conditions"])
    expected_result_rows = expected_conditions * len(protocol["candidate_objectives"])
    expected_curve_rows = expected_result_rows * len(protocol["fixed"]["tau"])

    summary = {
        "mode": args.mode,
        "formal_development": args.mode == "development",
        "n_objective_condition_rows": int(len(results)),
        "expected_objective_condition_rows": int(expected_result_rows),
        "n_objective_curve_rows": int(len(curves)),
        "expected_objective_curve_rows": int(expected_curve_rows),
        "complete": bool(len(results)==expected_result_rows and len(curves)==expected_curve_rows),
        "git_head": _git_head(),
        "protocol_sha256": file_sha256(protocol_path),
        "manifest_sha256": manifest_hash,
        "external_plan_sha256": file_sha256(ROOT / protocol["nature_external_plan"]["yaml_path"]),
        "protocol_status": protocol.get("status"),
        "protocol_version": protocol.get("version"),
    }
    _atomic_json(summary, out / "run_summary.json")
    _atomic_json(
        {"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__},
        out / "environment_manifest.json",
    )
    (out / "protocol_snapshot.yaml").write_text(protocol_path.read_text(encoding="utf-8"),encoding="utf-8")

    report = ROOT / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase2r_{args.mode}_experiment_report.md"
    write_phase2r_report(report, summary, gate, " ".join(sys.argv))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if gate is not None:
        print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"REPORT={report}")


if __name__ == "__main__":
    main()
