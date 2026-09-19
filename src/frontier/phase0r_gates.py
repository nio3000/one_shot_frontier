from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold


FULL_FEATURES = [
    "H_sigma", "log2_effective_support", "log10_alpha",
    "H_x_log2S", "H_x_log10alpha", "log2S_x_log10alpha",
]
REDUCED_FEATURES = ["H_sigma", "log2_effective_support"]


def predictor_frame(cells: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=cells.index)
    d["H_sigma"] = cells["target_H_sigma"].astype(float)
    d["log2_effective_support"] = np.log2(cells["effective_support_mean"].astype(float))
    d["log10_alpha"] = np.log10(cells["alpha"].astype(float))
    d["H_x_log2S"] = d["H_sigma"] * d["log2_effective_support"]
    d["H_x_log10alpha"] = d["H_sigma"] * d["log10_alpha"]
    d["log2S_x_log10alpha"] = d["log2_effective_support"] * d["log10_alpha"]
    return d


def _fit_model(X: pd.DataFrame, y: np.ndarray, features: list[str]) -> LinearRegression:
    m = LinearRegression()
    m.fit(X[features], y)
    return m


def _predict_clipped(model: LinearRegression, X: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.clip(model.predict(X[features]), 0.0, 1.0)


def fit_predictor_with_grouped_cv(cells: pd.DataFrame, folds: int = 5) -> dict[str, Any]:
    X = predictor_frame(cells)
    y = cells["tau_0_5"].to_numpy(float)
    # Every alpha for the same (H,S) must remain in the same fold.
    groups = (cells["target_H_sigma"].astype(str) + "__" + cells["requested_effective_support"].astype(str)).to_numpy()
    gkf = GroupKFold(n_splits=int(folds))
    pred_full = np.empty_like(y)
    pred_reduced = np.empty_like(y)
    fold_rows = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        mf = _fit_model(X.iloc[tr], y[tr], FULL_FEATURES)
        mr = _fit_model(X.iloc[tr], y[tr], REDUCED_FEATURES)
        pred_full[va] = _predict_clipped(mf, X.iloc[va], FULL_FEATURES)
        pred_reduced[va] = _predict_clipped(mr, X.iloc[va], REDUCED_FEATURES)
        fold_rows.append({
            "fold": fold,
            "n_train": int(len(tr)),
            "n_valid": int(len(va)),
            "full_mae": float(mean_absolute_error(y[va], pred_full[va])),
            "reduced_mae": float(mean_absolute_error(y[va], pred_reduced[va])),
        })
    full_mae = float(mean_absolute_error(y, pred_full))
    red_mae = float(mean_absolute_error(y, pred_reduced))
    full_r2 = float(r2_score(y, pred_full))
    red_r2 = float(r2_score(y, pred_reduced))
    mf = _fit_model(X, y, FULL_FEATURES)
    mr = _fit_model(X, y, REDUCED_FEATURES)
    return {
        "target": "tau_0_5",
        "n_cells": int(len(cells)),
        "grouped_cv_folds": int(folds),
        "cv": {
            "full_mae": full_mae,
            "reduced_mae": red_mae,
            "full_r2": full_r2,
            "reduced_r2": red_r2,
            "mae_ratio_full_over_reduced": float(full_mae / red_mae) if red_mae > 0 else float("inf"),
            "r2_gain": float(full_r2 - red_r2),
            "folds": fold_rows,
        },
        "full_model": {
            "features": FULL_FEATURES,
            "intercept": float(mf.intercept_),
            "coefficients": {k: float(v) for k, v in zip(FULL_FEATURES, mf.coef_)},
        },
        "reduced_model": {
            "features": REDUCED_FEATURES,
            "intercept": float(mr.intercept_),
            "coefficients": {k: float(v) for k, v in zip(REDUCED_FEATURES, mr.coef_)},
        },
    }


def predict_from_frozen_fit(cells: pd.DataFrame, fit: dict[str, Any]) -> np.ndarray:
    X = predictor_frame(cells)
    spec = fit["full_model"]
    z = np.full(len(cells), float(spec["intercept"]), dtype=float)
    for f in spec["features"]:
        z += float(spec["coefficients"][f]) * X[f].to_numpy(float)
    return np.clip(z, 0.0, 1.0)


def evaluate_r1(cells: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    q = cells[
        (cells["interior_gain"] >= float(cfg["min_interior_gain"]))
        & (cells["interior_gain_ci_low"] > 0)
    ].copy()
    nonzero_h = sorted(q.loc[q["target_H_sigma"] > 0, "target_H_sigma"].unique())
    supports = sorted(q["requested_effective_support"].unique())
    passed = (
        len(q) >= int(cfg["min_qualifying_cells"])
        and len(nonzero_h) >= int(cfg["min_nonzero_H_levels"])
        and len(supports) >= int(cfg["min_support_levels"])
    )
    return {
        "pass": bool(passed),
        "n_qualifying_cells": int(len(q)),
        "n_nonzero_H_levels": int(len(nonzero_h)),
        "n_support_levels": int(len(supports)),
        "qualifying_H_levels": [float(x) for x in nonzero_h],
        "qualifying_support_levels": [float(x) for x in supports],
        "min_qualifying_cells": int(cfg["min_qualifying_cells"]),
        "min_interior_gain": float(cfg["min_interior_gain"]),
    }


def evaluate_r2(fit: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    cv = fit["cv"]
    ratio = float(cv["mae_ratio_full_over_reduced"])
    gain = float(cv["r2_gain"])
    passed = ratio <= float(cfg["full_mae_relative_to_reduced_max"]) and gain >= float(cfg["min_R2_gain"])
    return {
        "pass": bool(passed),
        "full_mae": float(cv["full_mae"]),
        "reduced_mae": float(cv["reduced_mae"]),
        "mae_ratio": ratio,
        "full_r2": float(cv["full_r2"]),
        "reduced_r2": float(cv["reduced_r2"]),
        "r2_gain": gain,
        "mae_ratio_threshold": float(cfg["full_mae_relative_to_reduced_max"]),
        "r2_gain_threshold": float(cfg["min_R2_gain"]),
    }


def evaluate_r3(lw_cells: pd.DataFrame, phase0a_cells: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    left = lw_cells[["target_H_sigma", "requested_effective_support", "endpoint_delta_tau1_minus_tau0"]].copy()
    right = phase0a_cells[["target_H_sigma", "requested_effective_support", "delta_mean"]].copy()
    merged = left.merge(right, on=["target_H_sigma", "requested_effective_support"], how="inner")
    if len(merged) == 0:
        return {"pass": False, "error": "No matched Phase 0-A cells"}
    r = float(pearsonr(merged["endpoint_delta_tau1_minus_tau0"], merged["delta_mean"]).statistic)
    a = np.sign(merged["endpoint_delta_tau1_minus_tau0"].to_numpy(float))
    b = np.sign(merged["delta_mean"].to_numpy(float))
    agreement = float((a == b).mean())
    passed = r >= float(cfg["min_phase0A_delta_pearson_r"]) and agreement >= float(cfg["min_endpoint_sign_agreement"])
    return {
        "pass": bool(passed),
        "n_matched": int(len(merged)),
        "pearson_r": r,
        "sign_agreement": agreement,
        "min_pearson_r": float(cfg["min_phase0A_delta_pearson_r"]),
        "min_sign_agreement": float(cfg["min_endpoint_sign_agreement"]),
    }


def evaluate_holdout(cells: pd.DataFrame, curve_means: pd.DataFrame, fit: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    out = cells.copy()
    out["predicted_tau"] = predict_from_frozen_fit(out, fit)
    grid = np.sort(curve_means.loc[curve_means["estimator_family"] == "fixed_ridge", "tau"].unique().astype(float))
    out["predicted_tau_grid"] = [float(grid[np.argmin(np.abs(grid - p))]) for p in out["predicted_tau"]]
    regrets = []
    endpoint_regime_correct = []
    endpoint_eligible = []
    threshold = float(cfg.get("endpoint_regime_tau_threshold", 0.5))
    for _, row in out.iterrows():
        mask = (
            np.isclose(curve_means["target_H_sigma"], row["target_H_sigma"])
            & np.isclose(curve_means["requested_effective_support"], row["requested_effective_support"])
            & np.isclose(curve_means["alpha"], row["alpha"])
            & (curve_means["estimator_family"] == "fixed_ridge")
        )
        g = curve_means[mask]
        best = float(g["accuracy_mean"].max())
        chosen = float(g.loc[np.isclose(g["tau"], row["predicted_tau_grid"]), "accuracy_mean"].iloc[0])
        regrets.append(best - chosen)
        effect = float(row["endpoint_delta_tau1_minus_tau0"])
        eligible = abs(effect) >= float(cfg["C3_endpoint_effect_min"])
        endpoint_eligible.append(eligible)
        if eligible:
            true_o5 = effect > 0
            pred_o5 = float(row["predicted_tau"] ) >= threshold
            endpoint_regime_correct.append(bool(true_o5 == pred_o5))
        else:
            endpoint_regime_correct.append(np.nan)
    out["regret"] = regrets
    out["endpoint_regime_eligible"] = endpoint_eligible
    out["endpoint_regime_correct"] = endpoint_regime_correct
    mae = float(np.mean(np.abs(out["predicted_tau"] - out["tau_0_5"])))
    median_regret = float(np.median(out["regret"]))
    p90_regret = float(np.quantile(out["regret"], 0.90))
    eligible = out[out["endpoint_regime_eligible"]]
    sign_acc = float(eligible["endpoint_regime_correct"].astype(float).mean()) if len(eligible) else float("nan")
    C1 = mae <= float(cfg["C1_tau_mae_max"])
    C2 = median_regret <= float(cfg["C2_median_regret_max"]) and p90_regret <= float(cfg["C2_p90_regret_max"])
    C3 = len(eligible) > 0 and sign_acc >= float(cfg["C3_endpoint_sign_accuracy_min"])
    return {
        "C1": {"pass": bool(C1), "tau_mae": mae, "threshold": float(cfg["C1_tau_mae_max"])},
        "C2": {"pass": bool(C2), "median_regret": median_regret, "p90_regret": p90_regret,
               "median_threshold": float(cfg["C2_median_regret_max"]), "p90_threshold": float(cfg["C2_p90_regret_max"])},
        "C3": {"pass": bool(C3), "sign_accuracy": sign_acc, "n_eligible": int(len(eligible)),
               "threshold": float(cfg["C3_endpoint_sign_accuracy_min"])},
        "pass": bool(C1 and C2 and C3),
    }, out
