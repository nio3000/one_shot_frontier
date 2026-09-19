from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import environment_snapshot, file_sha256, get_git_head


def write_phase0h_report(
    path: Path,
    stage: str,
    protocol_path: Path,
    root: Path,
    summary: dict[str, Any],
    command: str,
    cells_path: Path | None = None,
    gate_path: Path | None = None,
    sanity_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = environment_snapshot()
    lines = [
        f"# Experiment Report — Phase 0-H {stage}", "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        "- Study phase: `Phase 0-H — High-Dimensional Covariance-Complexity Transition`",
        f"- Run stage: `{stage}`",
        f"- Protocol: `{protocol_path}`",
        f"- Protocol SHA256: `{file_sha256(protocol_path)}`",
        f"- Code Git HEAD: `{get_git_head(root)}`", "",
        "## 1. Scientific purpose", "",
        "Test the pre-frozen post-Phase-0-R hypothesis that usable covariance complexity depends on separating sample-to-dimension aspect ratio and spectral effective-rank ratio, with a fixed hinge at rho=1.", "",
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
                "d", "q_target", "target_H_sigma", "requested_rho", "n_runs",
                "tau_0_5", "best_tau", "interior_gain", "interior_gain_ci_low",
                "interior_gain_ci_high", "oracle_tau_0_5", "oracle_best_tau",
            ] if c in df.columns]
            lines += ["## 4.1 Canonical cell summary", "", "```text", df[cols].to_string(index=False), "```", ""]
        except Exception as e:
            lines += ["## 4.1 Canonical cell summary", "", f"Could not render: `{e}`", ""]

    if sanity_path and sanity_path.exists():
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        lines += ["## 5. Numerical sanity", "", "```json", json.dumps(sanity, ensure_ascii=False, indent=2), "```", ""]
    else:
        lines += ["## 5. Numerical sanity", "", "No formal sanity summary was generated for this engineering/incomplete run.", ""]

    if gate_path and gate_path.exists():
        gates = json.loads(gate_path.read_text(encoding="utf-8"))
        lines += ["## 6. Gate decision", "", "```json", json.dumps(gates, ensure_ascii=False, indent=2), "```", ""]
    else:
        lines += ["## 6. Gate decision", "", "No formal scientific gate evaluated for this engineering/incomplete run.", ""]

    lines += [
        "## 7. Negative results / failures", "",
        "All negative cells, failed runs, sanity failures, non-supportive effects, and protocol deviations must remain archived. No seed or structural cell may be removed.", "",
        "## 8. Paper-use note", "",
        "This report is a permanent experiment artifact. Manuscript claims must trace to canonical CSV/JSON outputs, not to chat summaries. Engineering-smoke values are not scientific evidence.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
