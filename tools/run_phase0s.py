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

from frontier.phase0s import run_phase0s_replicate
from frontier.phase0s_aggregate import aggregate_curve_points, aggregate_mechanism_metrics, summarize_cells
from frontier.phase0s_gates import evaluate_panel, global_decision
from frontier.phase0s_reporting import write_phase0s_report
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


def _materialize(path: Path, out: Path, sort_cols: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(path.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df = df.sort_values([c for c in sort_cols if c in df.columns]).reset_index(drop=True)
        _atomic_csv(df, out)
    return df


def _panel_grid(protocol: dict, panel: str):
    Hs = [float(x) for x in protocol["fixed"]["H_sigma"]]
    Ss = [float(x) for x in protocol["fixed"]["effective_support"]]
    if panel == "panel_a":
        sec = protocol["panel_A"]
        ds = [int(x) for x in sec["dimensions"]]
        qs = [float(x) for x in sec["q"]]
    elif panel == "panel_b":
        sec = protocol["panel_B"]
        ds = [int(sec["dimension"])]
        qs = [float(x) for x in sec["q"]]
    elif panel == "panel_c":
        sec = protocol["panel_C"]
        ds = [int(sec["dimension"])]
        qs = [float(x) for x in sec["q"]]
    else:
        raise ValueError(panel)
    seeds = [int(x) for x in sec["seeds"]]
    return [(d, q, h, s, seed) for d in ds for q in qs for h in Hs for s in Ss for seed in seeds]


def _smoke_grid(protocol: dict, reps: int):
    seeds = [8000 + i for i in range(int(reps))]
    return [
        ("smoke", 32, 0.25, 0.0, 1.0, seed, "gaussian") for seed in seeds
    ] + [
        ("smoke", 32, 0.25, 0.4, 2.0, seed, "gaussian") for seed in seeds
    ] + [
        ("smoke", 32, 0.25, 0.4, 2.0, seed, "student_t") for seed in seeds
    ]


def _shard_stem(d: int, q: float, h: float, s: float, seed: int) -> str:
    return f"d{d}__q{q:.6f}__H{h:.4f}__S{s:.6f}__seed{seed}"


def _run_task(protocol, panel, d, q, h, s, seed, tau_override, test_n_override, distribution_override):
    return run_phase0s_replicate(
        protocol, panel, d, q, h, s, seed,
        tau_override=tau_override,
        test_n_override=test_n_override,
        distribution_override=distribution_override,
    )


def _require_formal(protocol: dict, predictor_path: Path) -> dict:
    if str(protocol.get("status")) != "FROZEN" or "FROZEN" not in str(protocol.get("version")):
        raise RuntimeError("Formal Phase 0-S run blocked: protocol is not v1.0.0-FROZEN")
    if not predictor_path.exists():
        raise RuntimeError("Formal Phase 0-S run blocked: configs/phase0s_predictor_freeze.json is missing")
    p = json.loads(predictor_path.read_text(encoding="utf-8"))
    if p.get("freeze_type") != "PHASE0S_PREDICTOR_FREEZE":
        raise RuntimeError("Invalid Phase 0-S predictor freeze artifact")
    if get_git_head(ROOT) == "NO_GIT_HEAD":
        raise RuntimeError("Formal Phase 0-S run blocked: real Git HEAD required")
    return p


def _run_one_panel(args, protocol: dict, panel: str, predictor_freeze: dict | None, smoke_distribution: str | None = None) -> dict:
    is_smoke = panel == "smoke"
    if is_smoke:
        tasks_ext = _smoke_grid(protocol, args.smoke_replicates)
        tasks = [(d, q, h, s, seed, dist) for _, d, q, h, s, seed, dist in tasks_ext]
        tau_override = [0.0, 0.5, 1.0]
        test_n_override = int(args.test_n_override or 2000)
    else:
        tasks = [(d, q, h, s, seed, None) for d, q, h, s, seed in _panel_grid(protocol, panel)]
        tau_override = None
        test_n_override = None
        if args.test_n_override is not None:
            raise ValueError("--test-n-override is allowed only for smoke")

    stage_dir = Path(args.out_dir).resolve() / panel
    shard_root = stage_dir / "shards"
    dirs = {k: shard_root / k for k in ["runs", "curves", "oracle_curves", "mechanism"]}
    for ddir in dirs.values():
        ddir.mkdir(parents=True, exist_ok=True)
    failures = stage_dir / "failures.jsonl"

    git_head = get_git_head(ROOT)
    protocol_hash = file_sha256(Path(args.protocol).resolve())
    pred_hash = file_sha256(Path(args.predictor_freeze).resolve()) if predictor_freeze is not None else None

    pending = []
    for d, q, h, s, seed, dist in tasks:
        stem = _shard_stem(d, q, h, s, seed) + (f"__{dist}" if is_smoke else "")
        finals = [dirs[k] / f"{stem}.csv" for k in dirs]
        if args.resume and all(p.exists() for p in finals):
            continue
        pending.append((d, q, h, s, seed, dist, stem))

    start = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(_run_task, protocol, panel, d, q, h, s, seed, tau_override, test_n_override, dist): (d, q, h, s, seed, dist, stem)
                for d, q, h, s, seed, dist, stem in pending
            }
            for fut in as_completed(futs):
                d, q, h, s, seed, dist, stem = futs[fut]
                try:
                    run, curves, oracle, mech = fut.result()
                    meta = {"protocol_sha256": protocol_hash, "git_head": git_head, "predictor_freeze_sha256": pred_hash}
                    groups = {"runs": [run], "curves": curves, "oracle_curves": oracle, "mechanism": [mech]}
                    tmps = []
                    for name, rows in groups.items():
                        final = dirs[name] / f"{stem}.csv"
                        tmp = final.with_suffix(".csv.tmp")
                        pd.DataFrame([{**r, **meta} for r in rows]).to_csv(tmp, index=False)
                        tmps.append((tmp, final))
                    for tmp, final in tmps:
                        os.replace(tmp, final)
                except Exception as e:
                    with open(failures, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"panel": panel, "d": d, "q": q, "H": h, "S": s, "seed": seed, "distribution": dist, "error": repr(e), "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n")

    runs_path = stage_dir / "runs.csv"
    curves_path = stage_dir / "curves.csv"
    oracle_path = stage_dir / "oracle_curves.csv"
    mech_path = stage_dir / "mechanism_metrics.csv"
    sort = ["d", "q_target", "target_H_sigma", "requested_effective_support", "seed"]
    runs = _materialize(dirs["runs"], runs_path, sort)
    curves = _materialize(dirs["curves"], curves_path, sort + ["tau"])
    oracle = _materialize(dirs["oracle_curves"], oracle_path, sort + ["tau"])
    mech = _materialize(dirs["mechanism"], mech_path, sort)

    expected_runs = len(tasks)
    expected_curves = expected_runs * len(tau_override if tau_override is not None else protocol["fixed"]["tau"])
    complete = len(runs) == expected_runs and len(curves) == expected_curves and len(oracle) == expected_curves and len(mech) == expected_runs

    curve_means_path = stage_dir / "curve_means.csv"
    oracle_means_path = stage_dir / "oracle_curve_means.csv"
    cells_path = stage_dir / "cells.csv"
    mech_summary_path = stage_dir / "mechanism_summary.csv"
    gate_path = stage_dir / "gate_summary.json"
    pred_table_path = stage_dir / "predictions.csv"

    gate = None
    if len(curves) and len(oracle):
        cm = aggregate_curve_points(curves, oracle=False)
        om = aggregate_curve_points(oracle, oracle=True)
        cells = summarize_cells(curves, oracle, tolerance=0.005)
        ms = aggregate_mechanism_metrics(mech)
        _atomic_csv(cm, curve_means_path)
        _atomic_csv(om, oracle_means_path)
        _atomic_csv(cells, cells_path)
        _atomic_csv(ms, mech_summary_path)
        if complete and not is_smoke:
            gate, table = evaluate_panel(cells, cm, predictor_freeze, protocol)
            _atomic_csv(table, pred_table_path)
            _atomic_text(json.dumps(gate, ensure_ascii=False, indent=2), gate_path)

    summary = {
        "mode": panel,
        "formal": not is_smoke,
        "n_run_rows": int(len(runs)),
        "expected_run_rows": int(expected_runs),
        "n_curve_rows": int(len(curves)),
        "expected_curve_rows": int(expected_curves),
        "n_oracle_curve_rows": int(len(oracle)),
        "expected_oracle_curve_rows": int(expected_curves),
        "n_mechanism_rows": int(len(mech)),
        "expected_mechanism_rows": int(expected_runs),
        "complete": bool(complete),
        "workers": int(args.workers),
        "total_wall_time_sec_this_invocation": float(time.perf_counter() - start),
        "git_head": git_head,
        "protocol_sha256": protocol_hash,
        "protocol_status": protocol.get("status"),
        "protocol_version": protocol.get("version"),
        "predictor_freeze_sha256": pred_hash,
        "failures_file_exists": failures.exists(),
        "artifacts": {"runs": str(runs_path), "curves": str(curves_path), "cells": str(cells_path), "curve_means": str(curve_means_path), "oracle_curves": str(oracle_path), "mechanism_metrics": str(mech_path), "gate_summary": str(gate_path)},
    }
    _atomic_text(json.dumps(summary, ensure_ascii=False, indent=2), stage_dir / "run_summary.json")
    _atomic_text(json.dumps(environment_snapshot(), ensure_ascii=False, indent=2), stage_dir / "environment_manifest.json")
    shutil.copy2(Path(args.protocol).resolve(), stage_dir / "protocol_snapshot.yaml")
    if predictor_freeze is not None:
        shutil.copy2(Path(args.predictor_freeze).resolve(), stage_dir / "predictor_freeze_snapshot.json")

    report = ROOT / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase0s_{panel}_experiment_report.md"
    write_phase0s_report(report, panel, Path(args.protocol).resolve(), summary, " ".join(sys.argv), gate_path if gate_path.exists() else None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if gate is not None:
        print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"REPORT={report}")
    return {"summary": summary, "gate": gate, "report": str(report)}


def _write_global(args, results: dict[str, dict]) -> dict:
    gates = {p: results[p]["gate"] for p in ["panel_a", "panel_b", "panel_c"]}
    if any(g is None for g in gates.values()):
        raise RuntimeError("Global Phase 0-S decision requires completed gate summaries for all three panels")
    decision = global_decision(gates["panel_a"], gates["panel_b"], gates["panel_c"])
    out = {"decision": decision, "panels": {p: {"pass": bool(gates[p]["pass"]), "gates": gates[p]} for p in gates}}
    root = Path(args.out_dir).resolve()
    _atomic_text(json.dumps(out, ensure_ascii=False, indent=2), root / "phase0s_global_gate_summary.json")
    rows = []
    for p in ["panel_a", "panel_b", "panel_c"]:
        g = gates[p]
        rows.append({"panel": p, "pass": g["pass"], "S1": g["S1"]["pass"], "S2": g["S2"]["pass"], "S3": g["S3"]["pass"], "S4": g["S4"]["pass"], "mae": g["S1"]["mae"], "r2": g["S1"]["r2"]})
    _atomic_csv(pd.DataFrame(rows), root / "phase0s_all_panels_summary.csv")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "panel_a", "panel_b", "panel_c", "all"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--protocol", default=str(ROOT / "configs" / "phase0s_protocol.yaml"))
    ap.add_argument("--predictor-freeze", default=str(ROOT / "configs" / "phase0s_predictor_freeze.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "phase0s"))
    ap.add_argument("--smoke-replicates", type=int, default=2)
    ap.add_argument("--test-n-override", type=int)
    args = ap.parse_args()

    protocol = load_yaml(args.protocol)
    predictor = None
    if args.mode != "smoke":
        predictor = _require_formal(protocol, Path(args.predictor_freeze))

    if args.mode == "smoke":
        _run_one_panel(args, protocol, "smoke", None)
    elif args.mode in {"panel_a", "panel_b", "panel_c"}:
        _run_one_panel(args, protocol, args.mode, predictor)
    else:
        # Locked all-panels execution: no gate is consulted before starting the next panel.
        results = {}
        for panel in ["panel_a", "panel_b", "panel_c"]:
            results[panel] = _run_one_panel(args, protocol, panel, predictor)
        _write_global(args, results)


if __name__ == "__main__":
    main()
