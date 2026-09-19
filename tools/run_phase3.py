
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase3_moment_collision import (
    WitnessConfig,
    task_distributions,
    summary_signature,
    exact_collision_check,
    risk_curves,
    deterministic_minimax_regret,
    randomized_minimax_regret,
    epsilon_sensitivity,
    third_order_escape,
    arbitrary_dimension_summary,
)
from frontier.phase3_gates import evaluate_phase3_gates
from frontier.phase3_reporting import write_phase3_report


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "NO_GIT_HEAD"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "formal"], required=True)
    ap.add_argument("--protocol", default=str(ROOT / "configs" / "phase3_protocol.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "phase3"))
    args = ap.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))

    if args.mode == "formal":
        if protocol.get("version") != "1.0.0-FROZEN" or protocol.get("status") != "FROZEN":
            raise RuntimeError("Phase 3 formal blocked: protocol not 1.0.0-FROZEN / FROZEN")
        if git_head() == "NO_GIT_HEAD":
            raise RuntimeError("Phase 3 formal blocked: real Git HEAD required")

    frozen = protocol["construction"]
    cfg = WitnessConfig(
        epsilon=float(frozen["primary_epsilon"]),
        alpha0=float(frozen["alpha0"]),
    )
    taus = [float(x) for x in frozen["tau_grid"]]
    tasks = task_distributions(cfg)

    sigA = summary_signature(tasks["A"])
    sigB = summary_signature(tasks["B"])
    pop_diff = float(np.max(np.abs(np.array(sigA, dtype=float) - np.array(sigB, dtype=float))))

    curves = risk_curves(cfg, taus)
    bestA = float(curves["A"].max())
    bestB = float(curves["B"].max())
    optA = [float(taus[i]) for i, x in enumerate(curves["A"]) if abs(float(x) - bestA) <= 1e-12]
    optB = [float(taus[i]) for i, x in enumerate(curves["B"]) if abs(float(x) - bestB) <= 1e-12]
    det = deterministic_minimax_regret(curves)
    rand = randomized_minimax_regret(curves)

    exact = exact_collision_check(cfg, int(frozen["n_per_class_exact_witness"]))
    # Convert Fractions for JSON.
    exact_json = {
        "class0_exact": exact["class0_exact"],
        "class1_exact": exact["class1_exact"],
        "class0_A": {
            "n": exact["class0_A"]["n"],
            "sum": str(exact["class0_A"]["sum"]),
            "second_sum": str(exact["class0_A"]["second_sum"]),
            "counts": list(exact["class0_A"]["counts"]),
        },
        "class0_B": {
            "n": exact["class0_B"]["n"],
            "sum": str(exact["class0_B"]["sum"]),
            "second_sum": str(exact["class0_B"]["second_sum"]),
            "counts": list(exact["class0_B"]["counts"]),
        },
        "class1_A": {
            "n": exact["class1_A"]["n"],
            "sum": str(exact["class1_A"]["sum"]),
            "second_sum": str(exact["class1_A"]["second_sum"]),
            "counts": list(exact["class1_A"]["counts"]),
        },
        "class1_B": {
            "n": exact["class1_B"]["n"],
            "sum": str(exact["class1_B"]["sum"]),
            "second_sum": str(exact["class1_B"]["second_sum"]),
            "counts": list(exact["class1_B"]["counts"]),
        },
    }

    sens_eps = [float(x) for x in frozen["epsilon_sensitivity"]]
    sens = epsilon_sensitivity(sens_eps, taus, float(frozen["alpha0"]))

    escape = third_order_escape(cfg)
    escape["extra_scalars"] = 2
    dims = [1] if args.mode == "smoke" else [int(x) for x in frozen["dimension_extension"]]
    dim_rows = [arbitrary_dimension_summary(cfg, d) for d in dims]
    dim_summary = {
        "dims": dims,
        "max_summary_difference": float(max(r["max_summary_difference"] for r in dim_rows)),
        "all_full_rank_nuisance_extensions": bool(all(r["full_rank_nuisance_extension"] for r in dim_rows)),
    }

    scientific = {
        "exact_collision": exact_json,
        "population_summary_max_abs_diff": pop_diff,
        "primary": {
            "tau_grid": taus,
            "bacc_task_A": curves["A"].tolist(),
            "bacc_task_B": curves["B"].tolist(),
            "optimal_tau_set_A": optA,
            "optimal_tau_set_B": optB,
            "optimal_sets_disjoint": len(set(optA).intersection(set(optB))) == 0,
            "deterministic_minimax_regret": float(det),
            "randomized_minimax_regret": float(rand["value"]),
            "randomized_minimax_strategy": rand,
        },
        "epsilon_sensitivity": sens,
        "third_order_escape": escape,
        "dimension_extension": dim_summary,
    }
    gate = evaluate_phase3_gates(scientific, protocol)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scientific_summary.json").write_text(json.dumps(scientific, indent=2), encoding="utf-8")
    (out / "gate_summary.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    run_summary = {
        "mode": args.mode,
        "formal": args.mode == "formal",
        "complete": True,
        "git_head": git_head(),
        "protocol_sha256": sha256(protocol_path),
        "protocol_status": protocol.get("status"),
        "protocol_version": protocol.get("version"),
        "nature_direction_freeze_sha256": sha256(ROOT / protocol["nature_direction_freeze"]["path"]),
        "external_v2_addendum_sha256": sha256(ROOT / protocol["external_validation_addendum"]["path"]),
    }
    (out / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    (out / "environment_manifest.json").write_text(
        json.dumps({"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}, indent=2),
        encoding="utf-8",
    )
    (out / "protocol_snapshot.yaml").write_text(protocol_path.read_text(encoding="utf-8"), encoding="utf-8")

    report = ROOT / "reports" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase3_{args.mode}_experiment_report.md"
    write_phase3_report(report, run_summary, gate, " ".join(sys.argv))

    print(json.dumps(run_summary, indent=2))
    print(json.dumps(gate, indent=2))
    print(f"REPORT={report}")


if __name__ == "__main__":
    main()
