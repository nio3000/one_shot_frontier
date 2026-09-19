from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import environment_snapshot, file_sha256, get_git_head


def write_experiment_report(
    report_path: Path,
    experiment_name: str,
    phase: str,
    protocol_path: Path,
    results_path: Path,
    command: str,
    summary: dict[str, Any],
    gates_path: Path | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    env = environment_snapshot()
    git_head = get_git_head(report_path.parents[1] if len(report_path.parents) > 1 else None)
    lines = [
        f"# Experiment Report — {experiment_name}",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Phase: `{phase}`",
        f"- Protocol: `{protocol_path}`",
        f"- Protocol SHA256: `{file_sha256(protocol_path)}`",
        f"- Code Git HEAD: `{git_head}`",
        f"- Results: `{results_path}`",
        "",
        "## 1. Scientific purpose",
        "",
        "Test the frozen Phase 0 estimability-frontier hypothesis without post-hoc seed, threshold, or condition selection.",
        "",
        "## 2. Reproducibility command",
        "",
        "```text",
        command,
        "```",
        "",
        "## 3. Environment",
        "",
    ]
    for k, v in env.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## 4. Results summary", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""]
    try:
        df = pd.read_csv(results_path)
        if len(df):
            preferred = [c for c in [
                "target_H_sigma", "requested_effective_support", "effective_support_mean",
                "n_runs", "oracle_o4_acc_mean", "oracle_o5_acc_mean",
                "estimated_o4_acc_mean", "estimated_o5_acc_mean",
                "delta_mean", "delta_ci_low", "delta_ci_high",
                "oracle_expressivity_gain_mean", "extra_o5_estimation_burden_mean"
            ] if c in df.columns]
            view = df[preferred] if preferred else df
            lines += ["## 4.1 Canonical result table", "", "```text", view.to_string(index=False), "```", ""]
    except Exception as e:
        lines += ["## 4.1 Canonical result table", "", f"Could not render result table: `{e}`", ""]
    if gates_path and gates_path.exists():
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        lines += ["## 5. Gate decision", "", "```json", json.dumps(gates, ensure_ascii=False, indent=2), "```", ""]
    else:
        lines += ["## 5. Gate decision", "", "No formal scientific gate was evaluated in this run (e.g. smoke or incomplete discovery/holdout).", ""]
    lines += [
        "## 6. Negative results / failures",
        "",
        "All failed cells, exceptions, non-supportive effects, and negative findings must remain in the canonical artifacts. Do not delete or cherry-pick them.",
        "",
        "## 7. Paper-use note",
        "",
        "This report is a permanent stage artifact. Numerical claims in the manuscript should be traced to the canonical CSV/JSON artifacts rather than copied from chat summaries.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
