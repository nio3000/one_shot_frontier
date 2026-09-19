from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frontier.phase0s_aggregate import aggregate_curve_points, summarize_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True)
    args = ap.parse_args()
    d = Path(args.stage_dir)
    curves = pd.read_csv(d / "curves.csv")
    oracle = pd.read_csv(d / "oracle_curves.csv")
    aggregate_curve_points(curves).to_csv(d / "curve_means.csv", index=False)
    aggregate_curve_points(oracle, oracle=True).to_csv(d / "oracle_curve_means.csv", index=False)
    summarize_cells(curves, oracle).to_csv(d / "cells.csv", index=False)
    print(d / "cells.csv")

if __name__ == "__main__":
    main()
