from __future__ import annotations

import argparse
import json
import os

# Prevent BLAS oversubscription when ProcessPool workers are used.
# This changes only computational scheduling, not any scientific calculation.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase0r import run_phase0r_replicate
from frontier.phase0r_aggregate import aggregate_curve_points, summarize_fixed_ridge_cells, summarize_lw_cells
from frontier.phase0r_gates import (
    evaluate_holdout, evaluate_r1, evaluate_r2, evaluate_r3,
    fit_predictor_with_grouped_cv,
)
from frontier.phase0r_reporting import write_phase0r_report
from frontier.utils import environment_snapshot, file_sha256, get_git_head, load_yaml


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _section(protocol: dict, mode: str) -> dict:
    if mode in {"smoke", "discovery"}:
        return protocol["discovery"]
    if mode == "holdout_a":
        return protocol["holdout_A"]
    if mode == "holdout_b":
        return protocol["holdout_B"]
    raise ValueError(mode)


def _formal_grid(protocol: dict, mode: str):
    sec = _section(protocol, mode)
    grid = [(float(h), float(s)) for h in sec["H_sigma"] for s in sec["effective_support"]]
    seeds = [int(x) for x in sec["seeds"]]
    return grid, seeds


def _smoke_grid(protocol: dict, smoke_replicates: int):
    seeds = [int(x) for x in protocol["discovery"]["seeds"][:smoke_replicates]]
    return [(0.0, 1.0), (0.40, 4.0), (0.80, 16.0)], seeds


def _shard_name(h: float, s: float, seed: int) -> str:
    return f"H{h:.4f}__S{s:.4f}__seed{seed}.csv"


def _run_task(protocol, mode, h, s, seed, test_n, tau_override, alpha_override):
    return run_phase0r_replicate(
        protocol, mode, h, s, seed,
        test_n_override=test_n,
        tau_override=tau_override,
        alpha_override=alpha_override,
    )


def _materialize_curves(shards_dir: Path, out: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(shards_dir.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df = df.sort_values(["target_H_sigma", "requested_effective_support", "seed", "estimator_family", "alpha", "tau"], na_position="last")
        _atomic_csv(df, out)
    return df


def _materialize_runs(run_rows_dir: Path, out: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(run_rows_dir.glob("*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["target_H_sigma", "requested_effective_support", "seed"])
        _atomic_csv(df, out)
    return df


def _check_holdout_unblocked(root: Path) -> None:
    gate = root / "runs" / "phase0r" / "discovery" / "gate_summary.json"
    if not gate.exists():
        raise RuntimeError("Holdout blocked: discovery gate_summary.json does not exist")
    g = json.loads(gate.read_text(encoding="utf-8"))
    if not bool(g.get("holdouts_unblocked")):
        raise RuntimeError(f"Holdout blocked by frozen R1/R2 gates: {g.get('decision')}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "phase0r_protocol.yaml"))
    p.add_argument("--mode", choices=["smoke", "discovery", "holdout_a", "holdout_b"], required=True)
    p.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "phase0r"))
    p.add_argument("--phase0a-cells", default=str(ROOT / "runs" / "phase0" / "discovery" / "cells.csv"))
    p.add_argument("--test-n-override", type=int, default=None)
    p.add_argument("--smoke-replicates", type=int, default=2)
    args = p.parse_args()

    protocol_path = Path(args.config).resolve()
    protocol = load_yaml(protocol_path)
    if args.mode in {"holdout_a", "holdout_b"}:
        _check_holdout_unblocked(ROOT)

    is_smoke = args.mode == "smoke"
    if is_smoke:
        grid, seeds = _smoke_grid(protocol, args.smoke_replicates)
        test_n = int(args.test_n_override or 2000)
        tau_override = [0.0, 0.5, 1.0]
        alpha_override = [0.03, 0.30]
    else:
        grid, seeds = _formal_grid(protocol, args.mode)
        test_n = args.test_n_override
        tau_override = None
        alpha_override = None
        if args.test_n_override is not None:
            raise ValueError("--test-n-override is allowed only for engineering smoke; formal runs must use the frozen protocol")

    stage_dir = Path(args.out_dir).resolve() / args.mode
    shards_dir = stage_dir / "curve_shards"
    run_rows_dir = stage_dir / "run_rows"
    shards_dir.mkdir(parents=True, exist_ok=True)
    run_rows_dir.mkdir(parents=True, exist_ok=True)
    failures_path = stage_dir / "failures.jsonl"

    current_git = get_git_head(ROOT)
    config_hash = file_sha256(protocol_path)
    pending = []
    for h, s in grid:
        for seed in seeds:
            sp = shards_dir / _shard_name(h, s, seed)
            rp = run_rows_dir / (_shard_name(h, s, seed).replace(".csv", ".json"))
            if args.resume and sp.exists() and rp.exists():
                continue
            pending.append((h, s, seed))

    run_start = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(_run_task, protocol, args.mode, h, s, seed, test_n, tau_override, alpha_override): (h, s, seed)
                for h, s, seed in pending
            }
            for fut in as_completed(futs):
                h, s, seed = futs[fut]
                try:
                    run_row, curves = fut.result()
                    meta = {
                        "protocol_sha256": config_hash,
                        "git_head": current_git,
                    }
                    run_row.update(meta)
                    for r in curves:
                        r.update(meta)
                    shard = shards_dir / _shard_name(h, s, seed)
                    rowfile = run_rows_dir / (_shard_name(h, s, seed).replace(".csv", ".json"))
                    # Curve shard first; base row acts as completion marker only after curve persistence succeeds.
                    tmp = shard.with_suffix(".csv.tmp")
                    pd.DataFrame(curves).to_csv(tmp, index=False)
                    os.replace(tmp, shard)
                    _atomic_text(json.dumps(run_row, ensure_ascii=False, indent=2), rowfile)
                except Exception as e:
                    fail = {"H": h, "S": s, "seed": seed, "error": repr(e), "traceback": traceback.format_exc()}
                    with open(failures_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(fail, ensure_ascii=False) + "\n")

    runs_path = stage_dir / "runs.csv"
    curves_path = stage_dir / "curves.csv"
    runs = _materialize_runs(run_rows_dir, runs_path)
    curves = _materialize_curves(shards_dir, curves_path)

    expected_base = len(grid) * len(seeds)
    expected_curve_per_rep = (len(alpha_override) * len(tau_override) + len(tau_override)) if is_smoke else (
        len(_section(protocol, args.mode)["alpha"]) * len(_section(protocol, args.mode)["tau"]) + len(_section(protocol, args.mode)["tau"])
    )
    expected_curves = expected_base * expected_curve_per_rep
    complete = len(runs) == expected_base and len(curves) == expected_curves

    cells_path = stage_dir / "cells.csv"
    lw_cells_path = stage_dir / "lw_cells.csv"
    curve_means_path = stage_dir / "curve_means.csv"
    predictor_path = stage_dir / "predictor_fit.json"
    gate_path = stage_dir / "gate_summary.json"
    predictions_path = stage_dir / "holdout_predictions.csv"

    gate_result = None
    if len(curves):
        ci = float(protocol["numerical"]["ci_level"])
        tol = float(protocol["primary_targets"]["near_optimal_tolerance_abs_accuracy"])
        curve_means = aggregate_curve_points(curves, ci)
        cells = summarize_fixed_ridge_cells(curves, tol, ci)
        lw_cells = summarize_lw_cells(curves, ci)
        _atomic_csv(curve_means, curve_means_path)
        _atomic_csv(cells, cells_path)
        _atomic_csv(lw_cells, lw_cells_path)
        if complete and not is_smoke:
            if args.mode == "discovery":
                fit = fit_predictor_with_grouped_cv(cells, int(protocol["gates"]["R2"]["grouped_cv_folds"]))
                _atomic_text(json.dumps(fit, ensure_ascii=False, indent=2), predictor_path)
                r1 = evaluate_r1(cells, protocol["gates"]["R1"])
                r2 = evaluate_r2(fit, protocol["gates"]["R2"])
                p0a = Path(args.phase0a_cells).resolve()
                r3 = evaluate_r3(lw_cells, pd.read_csv(p0a), protocol["gates"]["R3"]) if p0a.exists() else {"pass": False, "status": "NOT_EVALUATED_MISSING_PHASE0A_CELLS"}
                if r1["pass"] and r2["pass"]:
                    decision = "DISCOVERY_SUPPORTS_LOCKED_CONFIRMATION"
                elif r3.get("pass") and (not r1["pass"] or not r2["pass"]):
                    decision = "REGULARIZATION_EXPLAINS_REENTRY_BUT_NO_USEFUL_CONTINUOUS_SURFACE"
                else:
                    decision = "REGULARIZATION_AWARE_ESTIMABILITY_DIRECTION_NOT_SUPPORTED_AT_DISCOVERY"
                gate_result = {"R1": r1, "R2": r2, "R3": r3, "decision": decision, "holdouts_unblocked": bool(r1["pass"] and r2["pass"])}
                _atomic_text(json.dumps(gate_result, ensure_ascii=False, indent=2), gate_path)
            else:
                discovery_dir = Path(args.out_dir).resolve() / "discovery"
                fit = json.loads((discovery_dir / "predictor_fit.json").read_text(encoding="utf-8"))
                hold, pred = evaluate_holdout(cells, curve_means, fit, protocol["confirmation_gates"])
                _atomic_csv(pred, predictions_path)
                gate_result = {"confirmation": hold, "decision": "HOLDOUT_PASS" if hold["pass"] else "HOLDOUT_FAIL"}
                _atomic_text(json.dumps(gate_result, ensure_ascii=False, indent=2), gate_path)

    summary = {
        "mode": args.mode,
        "formal": not is_smoke,
        "n_base_runs": int(len(runs)),
        "expected_base_runs": int(expected_base),
        "n_curve_rows": int(len(curves)),
        "expected_curve_rows": int(expected_curves),
        "complete": bool(complete),
        "workers": int(args.workers),
        "total_wall_time_sec_this_invocation": float(time.perf_counter() - run_start),
        "mean_wall_time_per_completed_base_run": float(runs["wall_time_sec"].mean()) if len(runs) and "wall_time_sec" in runs else None,
        "max_heterogeneity_abs_error": float(runs["heterogeneity_abs_error"].max()) if len(runs) else None,
        "max_tau0_class_cov_identity_abs": float(runs["tau0_class_cov_identity_max_abs"].max()) if len(runs) else None,
        "min_tau1_eigenvalue_alpha_min": float(runs["tau1_min_eigenvalue_alpha_min"].min()) if len(runs) else None,
        "all_curve_values_finite": bool(runs["all_curve_values_finite"].astype(bool).all()) if len(runs) else False,
        "git_head": current_git,
        "protocol_sha256": config_hash,
        "failures_file_exists": failures_path.exists(),
        "artifacts": {
            "runs": str(runs_path), "curves": str(curves_path), "cells": str(cells_path),
            "curve_means": str(curve_means_path), "lw_cells": str(lw_cells_path),
            "predictor_fit": str(predictor_path), "gate_summary": str(gate_path),
        },
    }
    _atomic_text(json.dumps(summary, ensure_ascii=False, indent=2), stage_dir / "run_summary.json")
    _atomic_text(json.dumps(environment_snapshot(), ensure_ascii=False, indent=2), stage_dir / "environment_manifest.json")
    shutil.copy2(protocol_path, stage_dir / "protocol_snapshot.yaml")

    report = ROOT / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase0r_{args.mode}_experiment_report.md"
    write_phase0r_report(
        report, args.mode, protocol_path, ROOT, summary, " ".join(sys.argv),
        cells_path if cells_path.exists() else None,
        gate_path if gate_path.exists() else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if gate_result is not None:
        print(json.dumps(gate_result, ensure_ascii=False, indent=2))
    print(f"REPORT={report}")


if __name__ == "__main__":
    main()
