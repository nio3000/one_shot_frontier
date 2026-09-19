from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_phase0s_report(
    path: Path,
    stage: str,
    protocol_path: Path,
    summary: dict[str, Any],
    command: str,
    gate_path: Path | None = None,
) -> None:
    gate = None
    if gate_path and gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    lines = [
        f"# Experiment Report — Phase 0-S {stage}", "",
        "## 1. Scientific purpose", "",
        "Locked confirmation of the pre-specified smooth estimability surface. No Phase 0-S refit or feature selection is permitted.", "",
        "## 2. Reproducibility command", "", f"```text\n{command}\n```", "",
        "## 3. Protocol", "", f"- Protocol: `{protocol_path}`", f"- Protocol SHA256: `{summary.get('protocol_sha256')}`", f"- Git HEAD: `{summary.get('git_head')}`", f"- Predictor freeze SHA256: `{summary.get('predictor_freeze_sha256')}`", "",
        "## 4. Run summary", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
    ]
    if gate is not None:
        lines += ["## 5. Gate decision", "", "```json", json.dumps(gate, ensure_ascii=False, indent=2), "```", ""]
    lines += ["## 6. Negative results / failures", "", "All failed cells, unfavorable panels, and negative gate results remain permanent artifacts.", "", "## 7. Paper-use note", "", "Engineering smoke is not scientific evidence. Formal panel claims must trace to canonical Phase 0-S CSV/JSON artifacts."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
