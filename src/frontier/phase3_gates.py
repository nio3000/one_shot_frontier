
from __future__ import annotations
import numpy as np


def evaluate_phase3_gates(summary: dict, protocol: dict) -> dict:
    g = protocol["gates"]

    t1 = bool(
        summary["exact_collision"]["class0_exact"]
        and summary["exact_collision"]["class1_exact"]
        and summary["population_summary_max_abs_diff"] <= float(g["T1"]["max_summary_abs_diff"])
    )

    t2 = bool(
        summary["primary"]["deterministic_minimax_regret"] >= float(g["T2"]["deterministic_minimax_regret_min"])
        and summary["primary"]["randomized_minimax_regret"] >= float(g["T2"]["randomized_minimax_regret_min"])
        and summary["primary"]["optimal_sets_disjoint"]
    )

    sens = summary["epsilon_sensitivity"]
    eps = np.array([r["epsilon"] for r in sens], dtype=float)
    det = np.array([r["deterministic_minimax_regret"] for r in sens], dtype=float)
    slope = float(np.polyfit(eps, det, 1)[0]) if len(eps) >= 2 else float("nan")
    max_dev = float(np.max(np.abs(det - 2.0 * eps)))
    t3 = bool(
        np.isfinite(slope)
        and abs(slope - float(g["T3"]["expected_slope"])) <= float(g["T3"]["slope_tolerance"])
        and max_dev <= float(g["T3"]["max_abs_deviation_from_2epsilon"])
    )

    t4 = bool(
        summary["third_order_escape"]["task_identifiable_by_sign_pattern"]
        and summary["third_order_escape"]["extra_scalars"] == int(g["T4"]["expected_extra_scalars"])
    )

    t5 = bool(
        summary["dimension_extension"]["max_summary_difference"] <= float(g["T5"]["max_summary_abs_diff"])
        and summary["dimension_extension"]["all_full_rank_nuisance_extensions"]
    )

    decision = (
        protocol["decisions"]["A"]
        if all([t1, t2, t3, t4, t5])
        else protocol["decisions"]["B"]
    )
    return {
        "T1_exact_second_order_collision": {"pass": t1},
        "T2_decision_nonidentifiability_gap": {"pass": t2},
        "T3_strength_scaling": {"pass": t3, "slope": slope, "max_abs_deviation_from_2epsilon": max_dev},
        "T4_third_order_escape_witness": {"pass": t4},
        "T5_arbitrary_dimension_extension": {"pass": t5},
        "decision": decision,
    }
