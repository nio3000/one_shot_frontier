from __future__ import annotations

import json
from pathlib import Path


def write_phase2r_report(path: Path, summary: dict, gate: dict | None, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Experiment Report — Phase 2-R Selector Objective Repair\n\n"
    text += "## Reproducibility command\n\n```text\n" + command + "\n```\n\n"
    text += "## Run summary\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n\n"
    if gate is not None:
        text += "## Gate decision\n\n```json\n" + json.dumps(gate, indent=2) + "\n```\n\n"
    text += "## Scientific boundary\n\n"
    text += (
        "Phase 2-R is development on the four previously exposed Phase 1 datasets. "
        "It may select one global train-object-only selector objective. "
        "The Nature-target external validation pool remains outcome-blinded until a winner is frozen. "
        "No Phase 2-R result is independent external confirmation.\n"
    )
    path.write_text(text, encoding="utf-8")
