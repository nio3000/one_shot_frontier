from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_phase4_gates(pairs: pd.DataFrame, protocol: dict) -> dict:
    g = protocol["gates"]

    d1 = bool(
        pairs["counts_equal"].all()
        and pairs["t2_comm_norm_diff"].max() <= float(g["D1"]["max_t2_norm_diff"])
        and pairs["t2_model_norm_diff"].max() <= float(g["D1"]["max_t2_norm_diff"])
    )

    d2 = bool(pairs["max_head_parameter_norm_diff"].max() <= float(g["D2"]["max_head_parameter_norm_diff"]))

    # Structural condition qualifies if at least one frozen higher-order intervention
    # yields a material non-zero pairwise decision-identifiability gap.
    by_condition = (
        pairs.groupby(["condition_id", "dataset_id", "modality", "encoder_id", "projection_dim"], as_index=False)
        .agg(max_pair_gap=("deterministic_pair_minimax_regret", "max"),
             max_curve_shift=("curve_linf_shift", "max"),
             any_disjoint=("optimal_sets_disjoint", "max"))
    )
    q = by_condition[by_condition["max_pair_gap"] >= float(g["D3"]["min_pair_gap"])]
    d3 = bool(
        len(q) >= int(g["D3"]["min_qualifying_conditions"])
        and q["dataset_id"].nunique() >= int(g["D3"]["min_datasets"])
        and (not g["D3"]["require_medical"] or "medical" in set(q["modality"]))
        and (not g["D3"]["require_nonmedical"] or "nonmedical" in set(q["modality"]))
        and (not g["D3"]["require_both_encoders"] or q["encoder_id"].nunique() >= 2)
        and (not g["D3"]["require_both_dims"] or q["projection_dim"].nunique() >= 2)
    )

    conflict = pairs[pairs["deterministic_pair_minimax_regret"] >= float(g["D3"]["min_pair_gap"])]
    diag_sep_all = float((pairs["diag_m3_distance"] > float(g["D4"]["min_third_summary_distance"])).mean())
    sketch_sep_all = float((pairs["sketch_m3_distance"] > float(g["D4"]["min_third_summary_distance"])).mean())
    if len(conflict):
        diag_sep_conflict = float((conflict["diag_m3_distance"] > float(g["D4"]["min_third_summary_distance"])).mean())
        sketch_sep_conflict = float((conflict["sketch_m3_distance"] > float(g["D4"]["min_third_summary_distance"])).mean())
    else:
        diag_sep_conflict = float("nan")
        sketch_sep_conflict = float("nan")
    fourth_ok = bool(pairs["sketch_m4_distance"].max() <= float(g["D4"]["max_fourth_control_distance"]))
    d4 = bool(
        diag_sep_all >= float(g["D4"]["min_diag_separation_fraction"])
        and sketch_sep_all >= float(g["D4"]["min_sketch_separation_fraction"])
        and fourth_ok
    )

    d5 = bool(
        pairs["diag_m3_aggregation_norm_error"].max() <= float(g["D5"]["max_aggregation_norm_error"])
        and pairs["sketch_m3_aggregation_norm_error"].max() <= float(g["D5"]["max_aggregation_norm_error"])
        and pairs["diag_m3_upload_ratio_vs_t2"].max() <= float(g["D5"]["max_diag_upload_ratio_vs_t2"])
        and pairs["sketch_m3_upload_ratio_vs_t2"].max() <= float(g["D5"]["max_sketch_upload_ratio_vs_t2"])
    )

    diag_eligible = bool(diag_sep_all >= float(g["D4"]["min_diag_separation_fraction"]) and d5)
    sketch_eligible = bool(sketch_sep_all >= float(g["D4"]["min_sketch_separation_fraction"]) and d5)

    if all([d1, d2, d3, d4, d5]):
        decision = protocol["decisions"]["A"]
        promoted = "rademacher_m3_k16" if sketch_eligible else ("diag_m3" if diag_eligible else None)
    elif all([d1, d2, d4, d5]) and not d3:
        decision = protocol["decisions"]["B"]
        promoted = None
    else:
        decision = protocol["decisions"]["C"]
        promoted = None

    return {
        "D1_second_order_invariance": {"pass": d1, "max_t2_comm_norm_diff": float(pairs["t2_comm_norm_diff"].max()), "max_t2_model_norm_diff": float(pairs["t2_model_norm_diff"].max())},
        "D2_gaussian_head_family_invariance": {"pass": d2, "max_head_parameter_norm_diff": float(pairs["max_head_parameter_norm_diff"].max())},
        "D3_real_representation_decision_gap": {
            "pass": d3,
            "n_qualifying_conditions": int(len(q)),
            "datasets": sorted(set(q["dataset_id"].astype(str))),
            "modalities": sorted(set(q["modality"].astype(str))),
            "encoders": sorted(set(q["encoder_id"].astype(str))),
            "projection_dims": sorted(set(q["projection_dim"].astype(int))),
            "median_max_pair_gap": float(by_condition["max_pair_gap"].median()),
            "median_max_curve_shift": float(by_condition["max_curve_shift"].median()),
        },
        "D4_escape_summary_specificity": {
            "pass": d4,
            "diag_m3_separation_fraction_all_pairs": diag_sep_all,
            "sketch_m3_separation_fraction_all_pairs": sketch_sep_all,
            "diag_m3_separation_fraction_on_conflicts": diag_sep_conflict,
            "sketch_m3_separation_fraction_on_conflicts": sketch_sep_conflict,
            "max_sketch_m4_negative_control_distance": float(pairs["sketch_m4_distance"].max()),
        },
        "D5_aggregation_and_communication": {
            "pass": d5,
            "max_diag_m3_aggregation_norm_error": float(pairs["diag_m3_aggregation_norm_error"].max()),
            "max_sketch_m3_aggregation_norm_error": float(pairs["sketch_m3_aggregation_norm_error"].max()),
            "max_diag_m3_upload_ratio_vs_t2": float(pairs["diag_m3_upload_ratio_vs_t2"].max()),
            "max_sketch_m3_upload_ratio_vs_t2": float(pairs["sketch_m3_upload_ratio_vs_t2"].max()),
        },
        "escape_candidate_promotion": promoted,
        "decision": decision,
    }
