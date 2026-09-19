from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_phase2a_gates(conditions: pd.DataFrame, protocol: dict) -> dict:
    g = protocol["phase2a_gates"]
    a1 = bool(
        conditions["object_recovery_pass"].all()
        and conditions["selector_test_independence_pass"].all()
        and conditions["all_finite"].all()
    )

    q = conditions[
        (conditions["adaptive_gain_vs_best_endpoint"] >= float(g["A2"]["min_gain"]))
        & (conditions["adaptive_gain_ci95_low"] > 0.0)
    ].copy()
    datasets = set(q["dataset_id"].astype(str))
    modalities = set(q["modality"].astype(str))
    encoders = set(q["encoder_id"].astype(str))
    dims = set(q["projection_dim"].astype(int))
    a2 = bool(
        len(q) >= int(g["A2"]["min_qualifying_conditions"])
        and len(datasets) >= int(g["A2"]["min_datasets"])
        and (not g["A2"]["require_medical"] or "medical" in modalities)
        and (not g["A2"]["require_nonmedical"] or "nonmedical" in modalities)
        and (not g["A2"]["require_both_encoders"] or len(encoders) >= 2)
        and (not g["A2"]["require_both_projection_dims"] or len(dims) >= 2)
    )

    opp = conditions[conditions["oracle_opportunity"]].copy()
    rec = opp["oracle_gain_recovery_fraction"].replace([np.inf, -np.inf], np.nan).dropna()
    median_rec = float(rec.median()) if len(rec) else float("nan")
    frac_half = float((rec >= 0.50).mean()) if len(rec) else 0.0
    a3 = bool(
        len(rec) > 0
        and median_rec >= float(g["A3"]["min_median_recovery_fraction_on_oracle_opportunity"])
        and frac_half >= float(g["A3"]["min_fraction_opportunity_conditions_recovery_ge_0_50"])
    )

    gains = conditions["adaptive_gain_vs_best_endpoint"].to_numpy(dtype=float)
    frac_loss_gt_2pp = float((gains < -0.02).mean())
    p10 = float(np.quantile(gains, 0.10))
    a4 = bool(
        frac_loss_gt_2pp <= float(g["A4"]["max_fraction_conditions_loss_gt_0_02"])
        and p10 >= float(g["A4"]["min_p10_adaptive_gain"])
    )

    max_ratio = float(conditions["adaptive_upload_ratio_vs_class_specific_object"].max())
    a5 = bool(max_ratio <= float(g["A5"]["max_adaptive_upload_ratio_vs_class_specific_object"]))

    if all([a1, a2, a3, a4, a5]):
        decision = protocol["phase2a_decisions"]["A"]
    elif a1 and a5 and (a2 or a3):
        decision = protocol["phase2a_decisions"]["B"]
    else:
        decision = protocol["phase2a_decisions"]["C"]

    return {
        "A1": {"pass": a1},
        "A2": {
            "pass": a2,
            "n_qualifying_conditions": int(len(q)),
            "datasets": sorted(datasets),
            "modalities": sorted(modalities),
            "encoders": sorted(encoders),
            "projection_dims": sorted(dims),
        },
        "A3": {
            "pass": a3,
            "n_oracle_opportunity_conditions": int(len(opp)),
            "median_recovery_fraction": median_rec,
            "fraction_recovery_ge_0_50": frac_half,
        },
        "A4": {
            "pass": a4,
            "fraction_conditions_loss_gt_0_02": frac_loss_gt_2pp,
            "p10_adaptive_gain": p10,
        },
        "A5": {"pass": a5, "max_upload_ratio_vs_class_specific_object": max_ratio},
        "decision": decision,
    }
