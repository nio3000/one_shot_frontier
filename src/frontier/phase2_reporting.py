from __future__ import annotations

import json
from pathlib import Path


def write_report(path: Path, summary: dict, gate: dict | None, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Experiment Report — Phase 2 Adaptive Partial-Covariance One-Shot FL\n\n"
    text += "## Reproducibility command\n\n```text\n" + command + "\n```\n\n"
    text += "## Run summary\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n\n"
    if gate is not None:
        text += "## Gate decision\n\n```json\n" + json.dumps(gate, indent=2) + "\n```\n\n"
    text += "## Scientific boundary\n\n"
    text += "Track A evaluates a train-observable one-shot selector on frozen representations. External model-level baselines remain a separate Track B until their adapters are formally qualified. Test-label oracle quantities are upper bounds only.\n"
    path.write_text(text, encoding="utf-8")
