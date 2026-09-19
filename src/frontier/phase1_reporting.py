from __future__ import annotations

import json
from pathlib import Path


def write_report(path: Path, summary: dict, gate: dict | None, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment Report — Phase 1 Representation-Level Empirical Audit",
        "",
        "## Reproducibility command",
        "",
        "```text",
        command,
        "```",
        "",
        "## Run summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    if gate is not None:
        lines += ["## Gate decision", "", "```json", json.dumps(gate, ensure_ascii=False, indent=2), "```", ""]
    lines += [
        "## Scientific boundary",
        "",
        "Phase 1 tests whether interior partial covariance complexity survives in real frozen learned representations. It does not fit a new universal estimability law and does not claim communication savings from the tau continuum.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
