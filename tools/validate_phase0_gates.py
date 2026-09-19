from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frontier.gates import evaluate_f1, evaluate_f2, evaluate_f3, sanity_summary
from frontier.reporting import write_experiment_report
from frontier.utils import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "phase0_protocol.yaml"))
    p.add_argument("--discovery", required=True)
    p.add_argument("--holdout", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    protocol_path = Path(args.config).resolve()
    protocol = load_yaml(protocol_path)
    discovery = pd.read_csv(args.discovery)
    gates = {
        "sanity_discovery": sanity_summary(discovery, protocol),
        "F1": evaluate_f1(discovery, protocol["falsification_gates"]["F1"]),
        "F2": evaluate_f2(discovery, protocol["falsification_gates"]["F2"]),
    }
    if args.holdout:
        holdout = pd.read_csv(args.holdout)
        gates["sanity_holdout"] = sanity_summary(holdout, protocol)
        gates["F3"] = evaluate_f3(discovery, holdout, protocol["falsification_gates"]["F3"])
    else:
        gates["F3"] = {"pass": False, "status": "NOT_EVALUATED_NO_HOLDOUT"}

    sanity_pass = all(v for k, v in gates["sanity_discovery"].items() if k.endswith("_pass"))
    f1 = bool(gates["F1"].get("pass"))
    f2 = bool(gates["F2"].get("pass"))
    f3 = bool(gates["F3"].get("pass"))
    if sanity_pass and f1 and f2 and f3:
        decision = "A_STABLE_ESTIMABILITY_FRONTIER_SUPPORTED"
    elif f1 and not f3:
        decision = "B_CROSSOVER_EXISTS_BUT_FRONTIER_NOT_GENERAL"
    else:
        decision = "C_ESTIMABILITY_FRONTIER_FALSIFIED_OR_NOT_SUPPORTED"
    gates["decision"] = decision

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gates, ensure_ascii=False, indent=2))

    report_path = ROOT / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase0_gate_report.md"
    summary = {"decision": decision, "discovery_cells": len(discovery), "holdout_present": bool(args.holdout)}
    write_experiment_report(report_path, "phase0_gate_evaluation", "phase0", protocol_path, Path(args.discovery), " ".join(sys.argv), summary, out)
    print(f"REPORT={report_path}")


if __name__ == "__main__":
    main()
