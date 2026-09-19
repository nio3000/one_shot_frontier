from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase0r_reporting import write_phase0r_report


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase0r-dir", default=str(ROOT / "runs" / "phase0r"))
    p.add_argument("--config", default=str(ROOT / "configs" / "phase0r_protocol.yaml"))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    base = Path(args.phase0r_dir).resolve()
    discovery = _load(base / "discovery" / "gate_summary.json")
    hold_a = _load(base / "holdout_a" / "gate_summary.json")
    hold_b = _load(base / "holdout_b" / "gate_summary.json")
    if discovery is None:
        raise RuntimeError("Discovery gate summary is missing")
    r1 = bool(discovery.get("R1", {}).get("pass"))
    r2 = bool(discovery.get("R2", {}).get("pass"))
    r3 = bool(discovery.get("R3", {}).get("pass"))
    if r1 and r2:
        if hold_a is None or hold_b is None:
            decision = "LOCKED_CONFIRMATION_INCOMPLETE"
        else:
            a_pass = bool(hold_a.get("confirmation", {}).get("pass"))
            b_pass = bool(hold_b.get("confirmation", {}).get("pass"))
            decision = (
                "REGULARIZATION_AWARE_CONTINUOUS_COMPLEXITY_SUPPORTED"
                if a_pass and b_pass
                else "CONTINUOUS_COMPLEXITY_EXISTS_BUT_NOT_TRANSFERABLE"
            )
    else:
        decision = (
            "REGULARIZATION_EXPLAINS_REENTRY_BUT_NO_USEFUL_CONTINUOUS_SURFACE"
            if r3
            else "REGULARIZATION_AWARE_ESTIMABILITY_DIRECTION_FALSIFIED"
        )
    result = {
        "phase": "Phase 0-R",
        "R1_pass": r1,
        "R2_pass": r2,
        "R3_pass": r3,
        "holdout_A": hold_a,
        "holdout_B": hold_b,
        "final_decision": decision,
    }
    out = Path(args.out).resolve() if args.out else base / "final_gate_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ROOT / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase0r_final_decision_report.md"
    write_phase0r_report(
        report, "final_decision", Path(args.config).resolve(), ROOT,
        {"final_decision": decision, "gate_summary": str(out)},
        " ".join(sys.argv), gate_path=out,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"REPORT={report}")


if __name__ == "__main__":
    main()
