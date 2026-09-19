from __future__ import annotations

import pandas as pd


def evaluate_phase1_gates(conditions: pd.DataFrame, protocol: dict) -> dict:
    g = protocol["phase1_gates"]
    q = conditions[conditions["qualifying_interior"]].copy()
    datasets = set(q["dataset_id"].astype(str))
    modalities = set(q["modality"].astype(str))
    encoders = set(q["encoder_id"].astype(str))
    dims = set(q["projection_dim"].astype(int))
    p1 = (
        len(q) >= int(g["P1"]["min_qualifying_conditions"])
        and len(datasets) >= int(g["P1"]["min_datasets"])
        and (not g["P1"]["require_medical"] or "medical" in modalities)
        and (not g["P1"]["require_nonmedical"] or "nonmedical" in modalities)
        and (not g["P1"]["require_both_encoders"] or len(encoders) >= 2)
        and (not g["P1"]["require_both_projection_dims"] or len(dims) >= 2)
    )
    dataset_has = conditions.groupby(["dataset_id", "modality"])["qualifying_interior"].any().reset_index()
    passed_datasets = dataset_has[dataset_has["qualifying_interior"]]
    p2 = (
        len(passed_datasets) >= int(g["P2"]["min_datasets_with_any_qualifying_condition"])
        and int((passed_datasets["modality"] == "medical").sum()) >= int(g["P2"]["min_medical_datasets_with_any_qualifying_condition"])
        and int((passed_datasets["modality"] == "nonmedical").sum()) >= int(g["P2"]["min_nonmedical_datasets_with_any_qualifying_condition"])
    )
    endpoint_fraction = float(conditions["tau_0_5"].isin([0.0, 1.0]).mean())
    p3 = endpoint_fraction <= float(g["P3"]["max_fraction_conditions_endpoint_selected"])
    p4 = bool(conditions["object_recovery_pass"].all() and conditions["all_finite"].all())
    if p1 and p2 and p3 and p4:
        decision = protocol["formal_decisions"]["A"]
    elif p4 and len(q) >= 3:
        decision = protocol["formal_decisions"]["B"]
    else:
        decision = protocol["formal_decisions"]["C"]
    return {
        "P1": {"pass": bool(p1), "n_qualifying_conditions": int(len(q)), "datasets": sorted(datasets), "modalities": sorted(modalities), "encoders": sorted(encoders), "projection_dims": sorted(dims)},
        "P2": {"pass": bool(p2), "n_datasets_with_any_qualifying_condition": int(len(passed_datasets)), "datasets": passed_datasets[["dataset_id", "modality"]].to_dict("records")},
        "P3": {"pass": bool(p3), "endpoint_tau_0_5_fraction": endpoint_fraction, "threshold": float(g["P3"]["max_fraction_conditions_endpoint_selected"])},
        "P4": {"pass": bool(p4), "all_object_recovery_pass": bool(conditions["object_recovery_pass"].all()), "all_finite": bool(conditions["all_finite"].all())},
        "decision": decision,
    }
