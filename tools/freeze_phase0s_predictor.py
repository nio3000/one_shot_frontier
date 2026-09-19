from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase0s_predictor import freeze_predictor_from_phase0h
from frontier.utils import get_git_head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0h-dir", default=str(ROOT / "runs" / "phase0h" / "discovery"))
    ap.add_argument("--out", default=str(ROOT / "configs" / "phase0s_predictor_freeze.json"))
    ap.add_argument("--engineering-reset", action="store_true")
    args = ap.parse_args()
    src = Path(args.phase0h_dir)
    out = Path(args.out)
    if out.exists() and not args.engineering_reset:
        raise RuntimeError(f"Predictor freeze already exists: {out}. Refusing overwrite.")
    payload = freeze_predictor_from_phase0h(
        phase0h_predictor_fit=src / "predictor_fit.json",
        phase0h_cells=src / "cells.csv",
        phase0h_run_summary=src / "run_summary.json",
        output=out,
        fitting_git_head=get_git_head(ROOT),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"PREDICTOR_FREEZE={out}")

if __name__ == "__main__":
    main()
