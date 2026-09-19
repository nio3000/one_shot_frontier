
from __future__ import annotations
import json
from pathlib import Path


def write_phase3_report(path: Path, run_summary: dict, gate: dict, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Experiment Report — Phase 3 Summary-Object Decision Identifiability\n\n"
    text += "## Reproducibility command\n\n```text\n" + command + "\n```\n\n"
    text += "## Run summary\n\n```json\n" + json.dumps(run_summary, indent=2) + "\n```\n\n"
    text += "## Gate decision\n\n```json\n" + json.dumps(gate, indent=2) + "\n```\n\n"
    text += "## Scientific boundary\n\n"
    text += (
        "Phase 3 establishes an explicit second-order summary collision and the resulting "
        "decision-identifiability gap for a frozen Gaussian covariance-complexity action family. "
        "It is not a universal impossibility theorem for all one-shot federated learning, and it does "
        "not show that every richer summary is sufficient. The third-order sketch is a constructive "
        "escape witness for the collision, not yet a deployable general selector.\n"
    )
    path.write_text(text, encoding="utf-8")
