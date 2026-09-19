from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "runs" / "phase1" / "gate_summary.json"
if not path.exists():
    raise RuntimeError("Phase 1 gate summary is missing")
print(json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
