from __future__ import annotations

import numpy as np
import pandas as pd


def _objective_metrics(df: pd.DataFrame, protocol: dict, objective_id: str) -> dict:
    x = df[df["objective_id"] == objective_id].copy()
    g = protocol["gates"]

    r1 = bool(
        x["object_recovery_pass"].all()
        and x["selector_test_independence_pass"].all()
        and x["all_finite"].all()
    )

    q = x[
        (x["adaptive_gain_vs_best_endpoint"] >= float(protocol["primary_evaluation"]["qualifying_min_gain"]))
        & (x["adaptive_gain_ci95_low"] > 0.0)
    ]
    r2 = bool(
        len(q) >= int(g["R2"]["min_qualifying_conditions"])
        and q["dataset_id"].nunique() >= int(g["R2"]["min_datasets"])
        and (not g["R2"]["require_medical"] or "medical" in set(q["modality"]))
        and (not g["R2"]["require_nonmedical"] or "nonmedical" in set(q["modality"]))
        and (not g["R2"]["require_both_encoders"] or q["encoder_id"].nunique() >= 2)
        and (not g["R2"]["require_both_projection_dims"] or q["projection_dim"].nunique() >= 2)
    )

    opp = x[x["oracle_opportunity"]]
    rec = opp["oracle_gain_recovery_fraction"].replace([np.inf, -np.inf], np.nan).dropna()
    med_rec = float(rec.median()) if len(rec) else float("nan")
    frac_half = float((rec >= 0.50).mean()) if len(rec) else 0.0
    r3 = bool(
        len(rec) > 0
        and med_rec >= float(g["R3"]["median_recovery_min"])
        and frac_half >= float(g["R3"]["fraction_recovery_ge_half_min"])
    )

    gains = x["adaptive_gain_vs_best_endpoint"].to_numpy(dtype=float)
    frac_loss = float((gains < -0.02).mean())
    p10 = float(np.quantile(gains, 0.10))
    r4 = bool(
        frac_loss <= float(g["R4"]["max_fraction_loss_gt_0_02"])
        and p10 >= float(g["R4"]["p10_gain_min"])
    )

    max_ratio = float(x["adaptive_upload_ratio_vs_class_specific_object"].max())
    r5 = bool(max_ratio <= float(g["R5"]["max_upload_ratio_vs_class_specific_object"]))

    return {
        "objective_id": objective_id,
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "R4": r4,
        "R5": r5,
        "pass_all": bool(all([r1, r2, r3, r4, r5])),
        "n_qualifying_conditions": int(len(q)),
        "qualifying_datasets": sorted(set(q["dataset_id"].astype(str))),
        "qualifying_modalities": sorted(set(q["modality"].astype(str))),
        "qualifying_encoders": sorted(set(q["encoder_id"].astype(str))),
        "qualifying_dims": sorted(set(q["projection_dim"].astype(int))),
        "n_oracle_opportunities": int(len(opp)),
        "median_recovery_fraction": med_rec,
        "fraction_recovery_ge_0_50": frac_half,
        "fraction_conditions_loss_gt_0_02": frac_loss,
        "p10_adaptive_gain": p10,
        "mean_adaptive_gain": float(np.mean(gains)),
        "max_upload_ratio_vs_class_specific_object": max_ratio,
    }


def evaluate_phase2r_gates(results: pd.DataFrame, protocol: dict) -> dict:
    objective_ids = list(protocol["candidate_objectives"])
    metrics = {oid: _objective_metrics(results, protocol, oid) for oid in objective_ids}

    eligible = list(protocol["winner_eligible_objectives"])
    passing = [metrics[o] for o in eligible if metrics[o]["pass_all"]]

    winner = None
    if passing:
        tie_rank = {o: i for i, o in enumerate(protocol["winner_tie_order"])}
        passing.sort(
            key=lambda m: (
                m["fraction_conditions_loss_gt_0_02"],
                -m["p10_adaptive_gain"],
                -m["median_recovery_fraction"],
                -m["n_qualifying_conditions"],
                tie_rank[m["objective_id"]],
            )
        )
        winner = passing[0]["objective_id"]
        decision = protocol["decisions"]["A"]
    else:
        nll = metrics["nll_control"]
        improved = []
        for o in eligible:
            m = metrics[o]
            if (
                np.isfinite(m["median_recovery_fraction"])
                and np.isfinite(nll["median_recovery_fraction"])
                and m["median_recovery_fraction"] > nll["median_recovery_fraction"]
                and m["p10_adaptive_gain"] > nll["p10_adaptive_gain"]
            ):
                improved.append(o)
        decision = protocol["decisions"]["B"] if improved else protocol["decisions"]["C"]

    return {
        "decision": decision,
        "winner_objective_id": winner,
        "objective_metrics": metrics,
        "winner_rule": "safety-first frozen lexicographic rule",
    }
