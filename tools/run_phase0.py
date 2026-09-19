from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.experiment import compute_mean_scale, run_one
from frontier.aggregate import aggregate_runs
from frontier.reporting import write_experiment_report
from frontier.utils import file_sha256, get_git_head, load_yaml


def task(section, h, s, seed, mean_scale, floor, test_n):
    return run_one(section, h, s, seed, mean_scale, floor, test_n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "phase0_protocol.yaml"))
    p.add_argument("--mode", choices=["smoke", "discovery", "holdout"], required=True)
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "phase0"))
    p.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--test-n-override", type=int, default=None)
    p.add_argument("--smoke-replicates", type=int, default=3)
    args = p.parse_args()

    protocol_path = Path(args.config).resolve()
    protocol = load_yaml(protocol_path)
    phase = "discovery" if args.mode in {"smoke", "discovery"} else "holdout"
    section = dict(protocol[phase])
    if args.mode == "smoke":
        grid = [(0.0, 1.0), (0.40, 1.0), (0.80, 16.0)]
        start = int(section["training_seeds"]["start"])
        seeds = list(range(start, start + args.smoke_replicates))
    else:
        grid = [(float(h), float(s)) for h in section["H_sigma_grid"] for s in section["effective_support_grid"]]
        start = int(section["training_seeds"]["start"])
        count = int(section["training_seeds"]["count"])
        seeds = list(range(start, start + count))

    mean_scale = compute_mean_scale(section)
    floor = float(protocol["regularization"]["eigenvalue_floor_relative"])
    out_dir = Path(args.out_dir).resolve() / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "runs.csv"
    failures_path = out_dir / "failures.jsonl"

    existing = set()
    rows = []
    if args.resume and results_path.exists():
        old = pd.read_csv(results_path)
        rows = old.to_dict("records")
        existing = {(r["condition_id"], int(r["seed"])) for r in rows}

    pending = []
    for h, s in grid:
        for seed in seeds:
            cid = f"{section['generator_family']}__d{section['d']}__H{h:.4f}__S{s:.4f}"
            if (cid, seed) not in existing:
                pending.append((h, s, seed))

    meta = {
        "phase": phase,
        "mode": args.mode,
        "mean_scale": mean_scale,
        "config_sha256": file_sha256(protocol_path),
        "git_head": get_git_head(ROOT),
    }

    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(task, section, h, s, seed, mean_scale, floor, args.test_n_override): (h, s, seed)
                for h, s, seed in pending
            }
            for fut in as_completed(futures):
                h, s, seed = futures[fut]
                try:
                    row = fut.result()
                    row.update(meta)
                    rows.append(row)
                    pd.DataFrame(rows).sort_values(["target_H_sigma", "requested_effective_support", "seed"]).to_csv(results_path, index=False)
                except Exception as e:
                    failure = {"H": h, "S": s, "seed": seed, "error": repr(e), "traceback": traceback.format_exc()}
                    with open(failures_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    df = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    cells_path = out_dir / "cells.csv"
    cells = aggregate_runs(df, float(protocol["falsification_gates"]["F1"]["ci_level"])) if len(df) else pd.DataFrame()
    if len(cells):
        cells.to_csv(cells_path, index=False)
    summary = {
        "mode": args.mode,
        "n_rows": int(len(df)),
        "expected_rows": int(len(grid) * len(seeds)),
        "n_cells": int(len(cells)),
        "complete": bool(len(df) == len(grid) * len(seeds)),
        "mean_scale": mean_scale,
        "raw_runs": str(results_path),
        "cell_summary": str(cells_path),
        "delta_cell_mean_min": float(cells["delta_mean"].min()) if len(cells) else None,
        "delta_cell_mean_max": float(cells["delta_mean"].max()) if len(cells) else None,
        "max_object_recovery_abs": float(df["object_recovery_max_abs"].max()) if len(df) else None,
        "max_decomposition_residual_abs": float(df["decomposition_residual"].abs().max()) if len(df) else None,
        "failures_file_exists": failures_path.exists(),
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = ROOT / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.mode}_experiment_report.md"
    command = " ".join(sys.argv)
    write_experiment_report(report_path, args.mode, phase, protocol_path, cells_path if cells_path.exists() else results_path, command, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
