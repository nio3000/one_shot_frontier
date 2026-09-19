from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frontier.phase0s_gates import evaluate_panel
from frontier.utils import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--protocol", default=str(ROOT / "configs" / "phase0s_protocol.yaml"))
    ap.add_argument("--predictor-freeze", default=str(ROOT / "configs" / "phase0s_predictor_freeze.json"))
    args = ap.parse_args()
    d = Path(args.stage_dir)
    cells = pd.read_csv(d / "cells.csv")
    cm = pd.read_csv(d / "curve_means.csv")
    freeze = json.loads(Path(args.predictor_freeze).read_text(encoding="utf-8"))
    result, pred = evaluate_panel(cells, cm, freeze, load_yaml(args.protocol))
    (d / "gate_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pred.to_csv(d / "predictions.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
