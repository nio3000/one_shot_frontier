from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase0r_aggregate import aggregate_curve_points, summarize_fixed_ridge_cells, summarize_lw_cells
from frontier.utils import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "phase0r_protocol.yaml"))
    p.add_argument("--curves", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    protocol = load_yaml(args.config)
    curves = pd.read_csv(args.curves)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ci = float(protocol["numerical"]["ci_level"])
    tol = float(protocol["primary_targets"]["near_optimal_tolerance_abs_accuracy"])
    curve_means = aggregate_curve_points(curves, ci)
    cells = summarize_fixed_ridge_cells(curves, tol, ci)
    lw_cells = summarize_lw_cells(curves, ci)
    curve_means.to_csv(out / "curve_means.csv", index=False)
    cells.to_csv(out / "cells.csv", index=False)
    lw_cells.to_csv(out / "lw_cells.csv", index=False)
    print(f"curve_means={len(curve_means)} cells={len(cells)} lw_cells={len(lw_cells)}")


if __name__ == "__main__":
    main()
