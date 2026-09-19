from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase0h import run_phase0h_population_replicate
from frontier.phase0h_aggregate import (
    aggregate_curve_points,
    aggregate_mechanism_metrics,
    aggregate_oracle_curve_points,
    summarize_cells,
)
from frontier.phase0h_gates import (
    discovery_decision,
    evaluate_h1,
    evaluate_h2,
    evaluate_h3,
    evaluate_h4,
    evaluate_holdout,
    evaluate_sanity,
)
from frontier.phase0h_predictors import fit_leave_one_dimension_out
from frontier.phase0h_reporting import write_phase0h_report
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


def _formal_population_grid(protocol: dict, mode: str):
    sec = _section(protocol, mode)
    if mode == "discovery":
        dims = [int(x) for x in sec["dimensions"]]
    else:
        dims = [int(sec["dimension"])]
    qs = [float(x) for x in sec["q_effective_rank_over_d"]]
    hs = [float(x) for x in sec["H_sigma"]]
    seeds = [int(x) for x in sec["seeds"]]
    return [(d, q, h, seed) for d in dims for q in qs for h in hs for seed in seeds]


def _smoke_population_grid(protocol: dict, replicates: int):
    seeds = [int(x) for x in protocol["discovery"]["seeds"][:replicates]]
    # Engineering-only reduction; not scientific evidence.
    return [
        (32, 0.25, 0.0, s) for s in seeds
    ] + [
        (32, 0.25, 0.4, s) for s in seeds
    ]


def _shard_stem(d: int, q: float, h: float, seed: int) -> str:
    return f"d{d}__q{q:.6f}__H{h:.4f}__seed{seed}"


def _run_task(protocol, mode, d, q, h, seed, rho_override, tau_override, test_n_override):
    return run_phase0h_population_replicate(
        protocol, mode, d, q, h, seed,
        rho_override=rho_override,
        tau_override=tau_override,
        test_n_override=test_n_override,
    )


def _materialize_csv_dir(path: Path, out: Path, sort_cols: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(path.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df = df.sort_values([c for c in sort_cols if c in df.columns])
        _atomic_csv(df, out)
    return df


def _check_holdout_unblocked(root: Path) -> None:
    gate = root / "runs" / "phase0h" / "discovery" / "gate_summary.json"
    sanity = root / "runs" / "phase0h" / "discovery" / "sanity_summary.json"
    if not gate.exists() or not sanity.exists():
        raise RuntimeError("Holdout blocked: discovery gate/sanity summaries are missing")
    g = json.loads(gate.read_text(encoding="utf-8"))
    s = json.loads(sanity.read_text(encoding="utf-8"))
    if not bool(s.get("pass")):
        raise RuntimeError("Holdout blocked: discovery numerical sanity failed")
    if not bool(g.get("holdouts_unblocked")):
        raise RuntimeError(f"Holdout blocked by frozen H1-H4 gates: {g.get('decision')}")


def _require_formal_freeze(protocol: dict, mode: str) -> None:
    if mode == "smoke":
        return
    status = str(protocol.get("status", ""))
    version = str(protocol.get("version", ""))
    if status != "FROZEN" or "FROZEN" not in version:
        raise RuntimeError(
            f"Formal Phase 0-H run blocked: protocol is {status}/{version}. "
            "Engineering smoke must be audited and the protocol promoted to v1.0.0-FROZEN first."
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "phase0h_protocol.yaml"))
    p.add_argument("--mode", choices=["smoke", "discovery", "holdout_a", "holdout_b"], required=True)
    p.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "phase0h"))
    p.add_argument("--test-n-override", type=int, default=None)
    p.add_argument("--smoke-replicates", type=int, default=2)
    args = p.parse_args()

    protocol_path = Path(args.config).resolve()
    protocol = load_yaml(protocol_path)
    _require_formal_freeze(protocol, args.mode)
    if args.mode in {"holdout_a", "holdout_b"}:
        _check_holdout_unblocked(ROOT)

    is_smoke = args.mode == "smoke"
    if is_smoke:
        tasks = _smoke_population_grid(protocol, args.smoke_replicates)
        rho_override = [0.5, 1.0, 2.0]
        tau_override = [0.0, 0.5, 1.0]
        test_n_override = int(args.test_n_override or 2000)
    else:
        if args.test_n_override is not None:
            raise ValueError("--test-n-override is allowed only for engineering smoke")
        tasks = _formal_population_grid(protocol, args.mode)
        rho_override = None
        tau_override = None
        test_n_override = None

    sec = _section(protocol, args.mode)
    rhos = rho_override if rho_override is not None else [float(x) for x in sec["rho_nminus1_over_d"]]
    taus = tau_override if tau_override is not None else [float(x) for x in protocol["fixed"]["tau"]]

    stage_dir = Path(args.out_dir).resolve() / args.mode
    shard_root = stage_dir / "shards"
    dirs = {name: shard_root / name for name in ["runs", "curves", "oracle_curves", "mechanism"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    failures_path = stage_dir / "failures.jsonl"

    current_git = get_git_head(ROOT)
    config_hash = file_sha256(protocol_path)
    pending = []
    for d, q, h, seed in tasks:
        stem = _shard_stem(d, q, h, seed)
        complete_files = [dirs[k] / f"{stem}.csv" for k in dirs]
        if args.resume and all(x.exists() for x in complete_files):
            continue
        pending.append((d, q, h, seed))

    start = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(_run_task, protocol, args.mode, d, q, h, seed, rho_override, tau_override, test_n_override): (d, q, h, seed)
                for d, q, h, seed in pending
            }
            for fut in as_completed(futs):
                d, q, h, seed = futs[fut]
                stem = _shard_stem(d, q, h, seed)
                try:
                    run_rows, curves, oracle_rows, mechanism = fut.result()
                    meta = {"protocol_sha256": config_hash, "git_head": current_git}
                    groups = {
                        "runs": run_rows,
                        "curves": curves,
                        "oracle_curves": oracle_rows,
                        "mechanism": mechanism,
                    }
                    # Write all four to temporary files, then atomically promote.
                    temp_paths = []
                    for name, rows in groups.items():
                        final = dirs[name] / f"{stem}.csv"
                        tmp = final.with_suffix(".csv.tmp")
                        frame = pd.DataFrame([{**r, **meta} for r in rows])
                        frame.to_csv(tmp, index=False)
                        temp_paths.append((tmp, final))
                    for tmp, final in temp_paths:
                        os.replace(tmp, final)
                except Exception as e:
                    fail = {"d": d, "q": q, "H": h, "seed": seed, "error": repr(e), "traceback": traceback.format_exc()}
                    with open(failures_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(fail, ensure_ascii=False) + "\n")

    runs_path = stage_dir / "runs.csv"
    curves_path = stage_dir / "curves.csv"
    oracle_path = stage_dir / "oracle_curves.csv"
    mechanism_path = stage_dir / "mechanism_metrics.csv"
    runs = _materialize_csv_dir(dirs["runs"], runs_path, ["d", "q_target", "target_H_sigma", "requested_rho", "seed"])
    curves = _materialize_csv_dir(dirs["curves"], curves_path, ["d", "q_target", "target_H_sigma", "requested_rho", "seed", "tau"])
    oracle = _materialize_csv_dir(dirs["oracle_curves"], oracle_path, ["d", "q_target", "target_H_sigma", "requested_rho", "seed", "tau"])
    mechanism = _materialize_csv_dir(dirs["mechanism"], mechanism_path, ["d", "q_target", "target_H_sigma", "requested_rho", "seed"])

    expected_runs = len(tasks) * len(rhos)
    expected_curves = expected_runs * len(taus)
    complete = len(runs) == expected_runs and len(curves) == expected_curves and len(oracle) == expected_curves and len(mechanism) == expected_runs

    curve_means_path = stage_dir / "curve_means.csv"
    oracle_curve_means_path = stage_dir / "oracle_curve_means.csv"
    cells_path = stage_dir / "cells.csv"
    mechanism_summary_path = stage_dir / "mechanism_summary.csv"
    predictor_path = stage_dir / "predictor_fit.json"
    prediction_path = stage_dir / "predictor_oof_predictions.csv"
    gate_path = stage_dir / "gate_summary.json"
    sanity_path = stage_dir / "sanity_summary.json"
    holdout_pred_path = stage_dir / "holdout_predictions.csv"

    gate_result = None
    sanity_result = None
    if len(curves) and len(oracle):
        ci_level = 0.95
        tol = float(protocol["primary_targets"]["tau_near_optimal_tolerance_accuracy"])
        curve_means = aggregate_curve_points(curves, ci_level)
        oracle_means = aggregate_oracle_curve_points(oracle, ci_level)
        cells = summarize_cells(curves, oracle, tol, ci_level)
        mech_summary = aggregate_mechanism_metrics(mechanism)
        _atomic_csv(curve_means, curve_means_path)
        _atomic_csv(oracle_means, oracle_curve_means_path)
        _atomic_csv(cells, cells_path)
        _atomic_csv(mech_summary, mechanism_summary_path)

        if complete:
            sanity_result = evaluate_sanity(runs, cells, oracle_means, protocol, expected_runs, expected_curves, len(curves))
            _atomic_text(json.dumps(sanity_result, ensure_ascii=False, indent=2), sanity_path)

        if complete and not is_smoke and sanity_result and sanity_result["pass"]:
            if args.mode == "discovery":
                fit, pred_df = fit_leave_one_dimension_out(cells, curve_means)
                _atomic_text(json.dumps(fit, ensure_ascii=False, indent=2), predictor_path)
                _atomic_csv(pred_df, prediction_path)
                h1 = evaluate_h1(cells, protocol["discovery_gates"]["H1"])
                h2 = evaluate_h2(fit, protocol["discovery_gates"]["H2"])
                h3 = evaluate_h3(fit, protocol["discovery_gates"]["H3"])
                h4 = evaluate_h4(fit, protocol["discovery_gates"]["H4"])
                decision, unblocked = discovery_decision(h1, h2, h3, h4)
                gate_result = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "decision": decision, "holdouts_unblocked": bool(unblocked)}
                _atomic_text(json.dumps(gate_result, ensure_ascii=False, indent=2), gate_path)
            else:
                discovery_fit = json.loads((Path(args.out_dir).resolve() / "discovery" / "predictor_fit.json").read_text(encoding="utf-8"))
                frozen_spec = discovery_fit["models"]["M_hinge"]["all_discovery_fit"]
                hold, pred = evaluate_holdout(cells, curve_means, frozen_spec, protocol["confirmation_gates"])
                _atomic_csv(pred, holdout_pred_path)
                gate_result = {"confirmation": hold, "decision": "HOLDOUT_PASS" if hold["pass"] else "HOLDOUT_FAIL"}
                _atomic_text(json.dumps(gate_result, ensure_ascii=False, indent=2), gate_path)
        elif complete and not is_smoke and sanity_result and not sanity_result["pass"]:
            gate_result = {"decision": "INVALID_NUMERICAL_SANITY_FAILURE", "holdouts_unblocked": False}
            _atomic_text(json.dumps(gate_result, ensure_ascii=False, indent=2), gate_path)

    summary = {
        "mode": args.mode,
        "formal": not is_smoke,
        "n_population_tasks": int(len(tasks)),
        "n_run_rows": int(len(runs)),
        "expected_run_rows": int(expected_runs),
        "n_curve_rows": int(len(curves)),
        "expected_curve_rows": int(expected_curves),
        "n_oracle_curve_rows": int(len(oracle)),
        "expected_oracle_curve_rows": int(expected_curves),
        "n_mechanism_rows": int(len(mechanism)),
        "expected_mechanism_rows": int(expected_runs),
        "complete": bool(complete),
        "workers": int(args.workers),
        "total_wall_time_sec_this_invocation": float(time.perf_counter() - start),
        "git_head": current_git,
        "protocol_sha256": config_hash,
        "protocol_status": str(protocol.get("status")),
        "protocol_version": str(protocol.get("version")),
        "failures_file_exists": failures_path.exists(),
        "artifacts": {
            "runs": str(runs_path), "curves": str(curves_path), "curve_means": str(curve_means_path),
            "cells": str(cells_path), "oracle_curves": str(oracle_path), "oracle_curve_means": str(oracle_curve_means_path),
            "mechanism_metrics": str(mechanism_path), "mechanism_summary": str(mechanism_summary_path),
            "predictor_fit": str(predictor_path), "gate_summary": str(gate_path), "sanity_summary": str(sanity_path),
        },
    }
    _atomic_text(json.dumps(summary, ensure_ascii=False, indent=2), stage_dir / "run_summary.json")
    _atomic_text(json.dumps(environment_snapshot(), ensure_ascii=False, indent=2), stage_dir / "environment_manifest.json")
    shutil.copy2(protocol_path, stage_dir / "protocol_snapshot.yaml")

    report = ROOT / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase0h_{args.mode}_experiment_report.md"
    write_phase0h_report(
        report, args.mode, protocol_path, ROOT, summary, " ".join(sys.argv),
        cells_path if cells_path.exists() else None,
        gate_path if gate_path.exists() else None,
        sanity_path if sanity_path.exists() else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if sanity_result is not None:
        print(json.dumps({"sanity": sanity_result}, ensure_ascii=False, indent=2))
    if gate_result is not None:
        print(json.dumps(gate_result, ensure_ascii=False, indent=2))
    print(f"REPORT={report}")


if __name__ == "__main__":
    main()
