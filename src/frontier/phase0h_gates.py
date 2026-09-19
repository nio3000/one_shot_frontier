from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .phase0h_predictors import predict_from_spec


def evaluate_h1(cells: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    q = cells[(cells["interior_gain"] >= float(cfg["min_interior_gain"])) & (cells["interior_gain_ci_low"] > 0)].copy()
    dims = sorted(int(x) for x in q["d"].unique())
    qs = sorted(float(x) for x in q["q_target"].unique())
    rhos = sorted(float(x) for x in q["requested_rho"].unique())
    hs = sorted(float(x) for x in q.loc[q["target_H_sigma"] > 0, "target_H_sigma"].unique())
    passed = (
        len(q) >= int(cfg["min_qualifying_cells"])
        and len(dims) >= int(cfg["min_dimensions"])
        and len(qs) >= int(cfg["min_q_levels"])
        and len(rhos) >= int(cfg["min_rho_levels"])
        and len(hs) >= int(cfg["min_nonzero_H_levels"])
    )
    return {
        "pass": bool(passed),
        "n_qualifying_cells": int(len(q)),
        "qualifying_dimensions": dims,
        "qualifying_q_levels": qs,
        "qualifying_rho_levels": rhos,
        "qualifying_nonzero_H_levels": hs,
        "thresholds": dict(cfg),
    }


def evaluate_h2(fit: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    old = fit["models"]["M_old"]
    ratio = fit["models"]["M_ratio"]
    mae_ratio = float(ratio["oof_mae"] / old["oof_mae"]) if old["oof_mae"] > 0 else float("inf")
    r2_gain = float(ratio["oof_r2"] - old["oof_r2"])
    passed = mae_ratio <= float(cfg["mae_ratio_max"]) and r2_gain >= float(cfg["r2_gain_min"])
    return {
        "pass": bool(passed),
        "M_old_mae": float(old["oof_mae"]),
        "M_ratio_mae": float(ratio["oof_mae"]),
        "mae_ratio": mae_ratio,
        "M_old_r2": float(old["oof_r2"]),
        "M_ratio_r2": float(ratio["oof_r2"]),
        "r2_gain": r2_gain,
        "mae_ratio_threshold": float(cfg["mae_ratio_max"]),
        "r2_gain_threshold": float(cfg["r2_gain_min"]),
    }


def evaluate_h3(fit: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    ratio = fit["models"]["M_ratio"]
    hinge = fit["models"]["M_hinge"]
    mae_ratio = float(hinge["oof_mae"] / ratio["oof_mae"]) if ratio["oof_mae"] > 0 else float("inf")
    r2_gain = float(hinge["oof_r2"] - ratio["oof_r2"])
    passed = mae_ratio <= float(cfg["mae_ratio_max"]) and r2_gain >= float(cfg["r2_gain_min"])
    return {
        "pass": bool(passed),
        "M_ratio_mae": float(ratio["oof_mae"]),
        "M_hinge_mae": float(hinge["oof_mae"]),
        "mae_ratio": mae_ratio,
        "M_ratio_r2": float(ratio["oof_r2"]),
        "M_hinge_r2": float(hinge["oof_r2"]),
        "r2_gain": r2_gain,
        "mae_ratio_threshold": float(cfg["mae_ratio_max"]),
        "r2_gain_threshold": float(cfg["r2_gain_min"]),
        "hinge_rho": 1.0,
    }


def evaluate_h4(fit: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    hinge = fit["models"]["M_hinge"]
    plusd = fit["models"]["M_hinge_plus_d"]
    mae_improvement = float((hinge["oof_mae"] - plusd["oof_mae"]) / hinge["oof_mae"]) if hinge["oof_mae"] > 0 else float("inf")
    r2_gain = float(plusd["oof_r2"] - hinge["oof_r2"])
    base_ok = float(hinge["oof_mae"]) <= float(cfg["M_hinge_mae_max"]) and float(hinge["oof_r2"]) >= float(cfg["M_hinge_r2_min"])
    collapse_ok = mae_improvement <= float(cfg["explicit_d_mae_improvement_max_fraction"]) and r2_gain <= float(cfg["explicit_d_r2_gain_max"])
    return {
        "pass": bool(base_ok and collapse_ok),
        "M_hinge_mae": float(hinge["oof_mae"]),
        "M_hinge_r2": float(hinge["oof_r2"]),
        "M_hinge_plus_d_mae": float(plusd["oof_mae"]),
        "M_hinge_plus_d_r2": float(plusd["oof_r2"]),
        "explicit_d_mae_improvement_fraction": mae_improvement,
        "explicit_d_r2_gain": r2_gain,
        "base_predictive_quality_pass": bool(base_ok),
        "ratio_collapse_pass": bool(collapse_ok),
        "thresholds": dict(cfg),
    }


def discovery_decision(h1: dict[str, Any], h2: dict[str, Any], h3: dict[str, Any], h4: dict[str, Any]) -> tuple[str, bool]:
    if h1["pass"] and h2["pass"] and h3["pass"] and h4["pass"]:
        return "DISCOVERY_SUPPORTS_LOCKED_CONFIRMATION", True
    if h1["pass"] and (not h2["pass"] or not h3["pass"]):
        return "CONTINUOUS_COMPLEXITY_CONFIRMED_BUT_RANK_TRANSITION_NOT_SUPPORTED", False
    if h1["pass"] and h2["pass"] and h3["pass"] and not h4["pass"]:
        return "AMBIENT_DIMENSION_REMAINS_AN_INDEPENDENT_AXIS", False
    return "HIGH_DIMENSIONAL_TRANSITION_DIRECTION_FALSIFIED_AT_DISCOVERY", False


def _regime(tau_grid: float) -> str:
    t = float(tau_grid)
    if t <= 0.2 + 1e-12:
        return "LOW"
    if t >= 0.8 - 1e-12:
        return "HIGH"
    return "MID"


def evaluate_holdout(
    cells: pd.DataFrame,
    curve_means: pd.DataFrame,
    frozen_hinge_spec: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    out = cells.copy()
    pred = predict_from_spec(out, frozen_hinge_spec)
    grid = np.sort(curve_means["tau"].unique().astype(float))
    pred_grid = np.asarray([float(grid[np.argmin(np.abs(grid - p))]) for p in pred])
    out["predicted_tau"] = pred
    out["predicted_tau_grid"] = pred_grid
    regrets = []
    for (_, row), t in zip(out.iterrows(), pred_grid):
        mask = (
            (curve_means["d"].astype(int) == int(row["d"]))
            & np.isclose(curve_means["q_target"], float(row["q_target"]))
            & np.isclose(curve_means["target_H_sigma"], float(row["target_H_sigma"]))
            & np.isclose(curve_means["requested_rho"], float(row["requested_rho"]))
        )
        g = curve_means[mask]
        best = float(g["accuracy_mean"].max())
        chosen = float(g.loc[np.isclose(g["tau"], t), "accuracy_mean"].iloc[0])
        regrets.append(best - chosen)
    out["regret"] = np.asarray(regrets)
    out["observed_regime"] = [_regime(t) for t in out["tau_0_5"]]
    out["predicted_regime"] = [_regime(t) for t in pred_grid]
    out["regime_correct"] = out["observed_regime"] == out["predicted_regime"]

    mae = float(mean_absolute_error(out["tau_0_5"], out["predicted_tau"]))
    med = float(np.median(out["regret"]))
    p90 = float(np.quantile(out["regret"], 0.90))
    acc = float(out["regime_correct"].mean())
    q = out[(out["interior_gain"] >= 0.005) & (out["interior_gain_ci_low"] > 0)]
    nonzero_h = sorted(float(x) for x in q.loc[q["target_H_sigma"] > 0, "target_H_sigma"].unique())
    rhos = sorted(float(x) for x in q["requested_rho"].unique())

    c1 = mae <= float(cfg["C1_tau_mae_max"])
    c2 = med <= float(cfg["C2_median_regret_max"]) and p90 <= float(cfg["C2_p90_regret_max"])
    c3 = acc >= float(cfg["C3_complexity_regime_accuracy_min"])
    c4 = (
        len(q) >= int(cfg["C4_min_interior_cells"])
        and len(nonzero_h) >= int(cfg["C4_min_nonzero_H_levels"])
        and len(rhos) >= int(cfg["C4_min_rho_levels"])
    )
    result = {
        "C1": {"pass": bool(c1), "tau_mae": mae, "threshold": float(cfg["C1_tau_mae_max"])},
        "C2": {"pass": bool(c2), "median_regret": med, "p90_regret": p90,
               "median_threshold": float(cfg["C2_median_regret_max"]), "p90_threshold": float(cfg["C2_p90_regret_max"])},
        "C3": {"pass": bool(c3), "complexity_regime_accuracy": acc, "threshold": float(cfg["C3_complexity_regime_accuracy_min"])},
        "C4": {"pass": bool(c4), "n_interior_cells": int(len(q)), "nonzero_H_levels": nonzero_h, "rho_levels": rhos,
               "min_cells": int(cfg["C4_min_interior_cells"])},
        "pass": bool(c1 and c2 and c3 and c4),
    }
    return result, out


def evaluate_sanity(
    runs: pd.DataFrame,
    cells: pd.DataFrame,
    oracle_curve_means: pd.DataFrame,
    protocol: dict[str, Any],
    expected_runs: int,
    expected_curve_rows: int,
    actual_curve_rows: int,
) -> dict[str, Any]:
    cfg = protocol["sanity"]
    max_reff = float(runs["effective_rank_abs_error"].max()) if len(runs) else float("inf")
    max_h = float(runs["heterogeneity_abs_error"].max()) if len(runs) else float("inf")
    max_pool = float(runs["population_pooled_cov_relative_fro_error"].max()) if len(runs) else float("inf")
    max_tau0 = float(runs["tau0_class_cov_identity_max_abs"].max()) if len(runs) else float("inf")
    finite = bool(runs["all_curve_values_finite"].astype(bool).all()) if len(runs) else False

    # Oracle tau=1 should not be materially worse than tau=0 for nonzero-H Gaussian conditions.
    bad_oracle = []
    if len(oracle_curve_means):
        piv = oracle_curve_means.pivot_table(
            index=["d", "q_target", "target_H_sigma", "requested_rho"],
            columns="tau", values="oracle_accuracy_mean", aggfunc="first"
        ).reset_index()
        if 0.0 in piv.columns and 1.0 in piv.columns:
            q = piv[piv["target_H_sigma"] > 0].copy()
            q["delta"] = q[1.0] - q[0.0]
            bad_oracle = q[q["delta"] < -0.005].to_dict(orient="records")

    h0 = cells[np.isclose(cells["target_H_sigma"], 0.0)]
    h0_fraction = float((h0["tau_0_5"] <= float(cfg["H0_tau_0_5_max"]) + 1e-12).mean()) if len(h0) else 0.0
    checks = {
        "effective_rank": max_reff <= float(cfg["effective_rank_abs_tolerance"]),
        "heterogeneity": max_h <= float(cfg["H_sigma_abs_tolerance"]),
        "pooled_covariance_preservation": max_pool <= float(cfg["pooled_cov_relative_fro_error_max"]),
        "tau0_identity": max_tau0 <= float(cfg["tau0_class_cov_identity_abs_max"]),
        "all_finite": finite,
        "oracle_tau1_not_materially_worse": len(bad_oracle) == 0,
        "H0_negative_control": h0_fraction >= float(cfg["H0_fraction_cells_required"]),
        "run_count": len(runs) == int(expected_runs),
        "curve_count": int(actual_curve_rows) == int(expected_curve_rows),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "max_effective_rank_abs_error": max_reff,
        "max_H_sigma_abs_error": max_h,
        "max_pooled_cov_relative_fro_error": max_pool,
        "max_tau0_identity_abs": max_tau0,
        "H0_tau_0_5_pass_fraction": h0_fraction,
        "n_oracle_tau1_worse_by_gt_0_5pp": int(len(bad_oracle)),
        "oracle_review_examples": bad_oracle[:10],
    }
