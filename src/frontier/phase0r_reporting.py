from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import environment_snapshot, file_sha256, get_git_head


def write_phase0r_report(
    path: Path,
    stage: str,
    protocol_path: Path,
    root: Path,
    summary: dict[str, Any],
    command: str,
    cells_path: Path | None = None,
    gate_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = environment_snapshot()
    lines = [
        f"# Experiment Report — Phase 0-R {stage}", "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        "- Study phase: `Phase 0-R — Regularization-Aware Continuous-Complexity Falsification`",
        f"- Run stage: `{stage}`",
        f"- Protocol: `{protocol_path}`",
        f"- Protocol SHA256: `{file_sha256(protocol_path)}`",
        f"- Code Git HEAD: `{get_git_head(root)}`", "",
        "## 1. Scientific purpose", "",
        "Test the separately frozen post-falsification mechanism hypothesis. This experiment does not retroactively rescue Phase 0-A.", "",
        "## 2. Reproducibility command", "", "```text", command, "```", "",
        "## 3. Environment", "",
    ]
    for k, v in env.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## 4. Run summary", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""]
    if cells_path and cells_path.exists():
        try:
            df = pd.read_csv(cells_path)
            cols = [c for c in [
                "target_H_sigma", "requested_effective_support", "alpha", "n_runs",
                "tau_0_5", "best_tau", "interior_gain", "interior_gain_ci_low", "interior_gain_ci_high",
                "endpoint_delta_tau1_minus_tau0"
            ] if c in df.columns]
            lines += ["## 4.1 Canonical cell summary", "", "```text", df[cols].to_string(index=False), "```", ""]
        except Exception as e:
            lines += ["## 4.1 Canonical cell summary", "", f"Could not render: `{e}`", ""]
    if gate_path and gate_path.exists():
        gates = json.loads(gate_path.read_text(encoding="utf-8"))
        lines += ["## 5. Gate decision", "", "```json", json.dumps(gates, ensure_ascii=False, indent=2), "```", ""]
    else:
        lines += ["## 5. Gate decision", "", "No formal gate evaluated for this engineering/incomplete run.", ""]
    lines += [
        "## 6. Negative results / failures", "",
        "All negative cells, failed runs, non-supportive effects and protocol deviations must remain archived.", "",
        "## 7. Paper-use note", "",
        "This is a permanent experiment artifact. Manuscript claims must trace to canonical CSV/JSON outputs, not to chat summaries.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
