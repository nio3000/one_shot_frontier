from __future__ import annotations
import json
from pathlib import Path


def write_report(path: Path, summary: dict, gate: dict | None, command: str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    s="# Experiment Report — Phase 5 Natural-Domain Decision-Identifiability\n\n"
    s+="## Reproducibility command\n\n```text\n"+command+"\n```\n\n"
    s+="## Run summary\n\n```json\n"+json.dumps(summary,indent=2)+"\n```\n\n"
    if gate is not None:
        s+="## Gate decision\n\n```json\n"+json.dumps(gate,indent=2)+"\n```\n\n"
    s+="## Scientific boundary\n\n"
    s+=("This is a custom frozen-representation natural-domain study, not a WILDS leaderboard claim. "
        "Pair selection is frozen from fit-only second-order geometry before risk evaluation. "
        "All selected negative pairs remain part of the evidence. Rademacher-M3 is diagnostic only unless Decision A is reached.\n")
    path.write_text(s,encoding="utf-8")
