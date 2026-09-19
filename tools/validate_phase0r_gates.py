from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frontier.phase0r_gates import evaluate_holdout, evaluate_r1, evaluate_r2, evaluate_r3
from frontier.utils import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "phase0r_protocol.yaml"))
    p.add_argument("--cells", required=True)
    p.add_argument("--curve-means", required=True)
    p.add_argument("--predictor-fit", required=True)
    p.add_argument("--lw-cells", default=None)
    p.add_argument("--phase0a-cells", default=None)
    p.add_argument("--mode", choices=["discovery", "holdout_a", "holdout_b"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--predictions-out", default=None)
    args = p.parse_args()
    protocol = load_yaml(args.config)
    cells = pd.read_csv(args.cells)
    curve_means = pd.read_csv(args.curve_means)
    fit = json.loads(Path(args.predictor_fit).read_text(encoding="utf-8"))
    result = {"mode": args.mode}
    if args.mode == "discovery":
        result["R1"] = evaluate_r1(cells, protocol["gates"]["R1"])
        result["R2"] = evaluate_r2(fit, protocol["gates"]["R2"])
        if args.lw_cells and args.phase0a_cells and Path(args.phase0a_cells).exists():
            result["R3"] = evaluate_r3(pd.read_csv(args.lw_cells), pd.read_csv(args.phase0a_cells), protocol["gates"]["R3"])
        else:
            result["R3"] = {"pass": False, "status": "NOT_EVALUATED_MISSING_PHASE0A_CELLS"}
        r1 = bool(result["R1"]["pass"]); r2 = bool(result["R2"]["pass"]); r3 = bool(result["R3"].get("pass"))
        if r1 and r2:
            result["decision"] = "DISCOVERY_SUPPORTS_LOCKED_CONFIRMATION"
        elif r3 and (not r1 or not r2):
            result["decision"] = "REGULARIZATION_EXPLAINS_REENTRY_BUT_NO_USEFUL_CONTINUOUS_SURFACE"
        else:
            result["decision"] = "REGULARIZATION_AWARE_ESTIMABILITY_DIRECTION_NOT_SUPPORTED_AT_DISCOVERY"
        result["holdouts_unblocked"] = bool(r1 and r2)
    else:
        hold, pred = evaluate_holdout(cells, curve_means, fit, protocol["confirmation_gates"])
        result["confirmation"] = hold
        result["decision"] = "HOLDOUT_PASS" if hold["pass"] else "HOLDOUT_FAIL"
        if args.predictions_out:
            po = Path(args.predictions_out).resolve(); po.parent.mkdir(parents=True, exist_ok=True)
            pred.to_csv(po, index=False)
    out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
