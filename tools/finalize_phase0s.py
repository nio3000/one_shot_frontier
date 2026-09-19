from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frontier.phase0s_gates import global_decision


def main():
    base = ROOT / "runs" / "phase0s"
    gates = {}
    for p in ["panel_a", "panel_b", "panel_c"]:
        f = base / p / "gate_summary.json"
        if not f.exists():
            raise RuntimeError(f"Missing formal gate summary: {f}")
        gates[p] = json.loads(f.read_text(encoding="utf-8"))
    decision = global_decision(gates["panel_a"], gates["panel_b"], gates["panel_c"])
    out = {"decision": decision, "panels": {p: {"pass": bool(gates[p]["pass"])} for p in gates}}
    path = base / "phase0s_global_gate_summary.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"GLOBAL={path}")

if __name__ == "__main__":
    main()
