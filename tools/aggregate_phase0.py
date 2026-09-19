from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frontier.aggregate import aggregate_runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ci", type=float, default=0.95)
    args = p.parse_args()
    df = pd.read_csv(args.runs)
    agg = aggregate_runs(df, args.ci)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(f"WROTE {len(agg)} cells -> {args.out}")


if __name__ == "__main__":
    main()
