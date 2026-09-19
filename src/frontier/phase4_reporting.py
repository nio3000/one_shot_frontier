from __future__ import annotations

import json
from pathlib import Path


def write_phase4_report(path: Path, summary: dict, gate: dict | None, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Experiment Report — Phase 4 Real-Representation Identifiability Diagnostics\n\n"
    text += "## Reproducibility command\n\n```text\n" + command + "\n```\n\n"
    text += "## Run summary\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n\n"
    if gate is not None:
        text += "## Gate decision\n\n```json\n" + json.dumps(gate, indent=2) + "\n```\n\n"
    text += "## Scientific boundary\n\n"
    text += (
        "Phase 4 is a development/mechanism study on the four previously exposed feature-bank datasets. "
        "It uses prespecified class-reflection counterfactuals that preserve the communicated second-order training summary while altering odd/higher-order geometry. "
        "These counterfactuals are causal diagnostics, not natural external datasets. "
        "The third-order summaries are collision-separating candidates only; Phase 4 does not claim that third moments are universally sufficient or that a deployable selector has been solved. "
        "The Nature-target external outcome blackout remains active.\n"
    )
    path.write_text(text, encoding="utf-8")
