from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


MODEL_FEATURES = {
    "M_old": ["H", "log2_Seff", "H_x_log2_Seff"],
    "M_ratio": ["H", "log2_rho", "log2_q", "H_x_log2_rho", "H_x_log2_q", "log2_rho_x_log2_q"],
    "M_hinge": ["H", "u_minus", "u_plus", "log2_q", "H_x_u_minus", "H_x_u_plus", "H_x_log2_q"],
    "M_hinge_plus_d": ["H", "u_minus", "u_plus", "log2_q", "H_x_u_minus", "H_x_u_plus", "H_x_log2_q", "log2_d_over_64"],
}


def feature_frame(cells: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=cells.index)
    x["H"] = cells["actual_H_sigma_mean"].astype(float)
    rho = cells["actual_rho_mean"].astype(float)
    q = cells["q_actual_mean"].astype(float)
    x["log2_rho"] = np.log2(rho)
    x["log2_q"] = np.log2(q)
    x["log2_Seff"] = np.log2(rho / q)
    x["H_x_log2_Seff"] = x["H"] * x["log2_Seff"]
    x["H_x_log2_rho"] = x["H"] * x["log2_rho"]
    x["H_x_log2_q"] = x["H"] * x["log2_q"]
    x["log2_rho_x_log2_q"] = x["log2_rho"] * x["log2_q"]
    x["u_minus"] = np.minimum(x["log2_rho"], 0.0)
    x["u_plus"] = np.maximum(x["log2_rho"], 0.0)
    x["H_x_u_minus"] = x["H"] * x["u_minus"]
    x["H_x_u_plus"] = x["H"] * x["u_plus"]
    x["log2_d_over_64"] = np.log2(cells["d"].astype(float) / 64.0)
    return x


def _fit(X: pd.DataFrame, y: np.ndarray, features: list[str]) -> LinearRegression:
    m = LinearRegression()
    m.fit(X[features], y)
    return m


def _predict(m: LinearRegression, X: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.clip(m.predict(X[features]), 0.0, 1.0)


def _model_spec(model: LinearRegression, features: list[str]) -> dict[str, Any]:
    return {
        "features": list(features),
        "intercept": float(model.intercept_),
        "coefficients": {f: float(v) for f, v in zip(features, model.coef_)},
    }


def predict_from_spec(cells: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    X = feature_frame(cells)
    z = np.full(len(cells), float(spec["intercept"]), dtype=float)
    for f in spec["features"]:
        z += float(spec["coefficients"][f]) * X[f].to_numpy(float)
    return np.clip(z, 0.0, 1.0)


def _regret_for_predictions(cells: pd.DataFrame, curve_means: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    grid = np.sort(curve_means["tau"].unique().astype(float))
    regrets = []
    for (_, row), p in zip(cells.iterrows(), pred):
        chosen_tau = float(grid[np.argmin(np.abs(grid - float(p)))])
        mask = (
            (curve_means["d"].astype(int) == int(row["d"]))
            & np.isclose(curve_means["q_target"], float(row["q_target"]))
            & np.isclose(curve_means["target_H_sigma"], float(row["target_H_sigma"]))
            & np.isclose(curve_means["requested_rho"], float(row["requested_rho"]))
        )
        g = curve_means[mask]
        if len(g) == 0:
            raise KeyError("Missing curve for prediction-regret calculation")
        best = float(g["accuracy_mean"].max())
        chosen = float(g.loc[np.isclose(g["tau"], chosen_tau), "accuracy_mean"].iloc[0])
        regrets.append(best - chosen)
    return np.asarray(regrets, dtype=float)


def fit_leave_one_dimension_out(cells: pd.DataFrame, curve_means: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    dims = sorted(int(x) for x in cells["d"].unique())
    if len(dims) != 3:
        raise ValueError(f"Frozen discovery requires exactly three dimensions; got {dims}")
    X = feature_frame(cells)
    y = cells["tau_0_5"].to_numpy(float)
    predictions = pd.DataFrame(index=cells.index)
    predictions["observed_tau_0_5"] = y
    predictions["d"] = cells["d"].astype(int)
    for c in ["q_target", "target_H_sigma", "requested_rho"]:
        predictions[c] = cells[c].to_numpy()

    model_metrics: dict[str, Any] = {}
    for model_name, features in MODEL_FEATURES.items():
        oof = np.empty(len(cells), dtype=float)
        folds = []
        for fold, held_d in enumerate(dims):
            va = (cells["d"].astype(int).to_numpy() == held_d)
            tr = ~va
            m = _fit(X.loc[tr], y[tr], features)
            oof[va] = _predict(m, X.loc[va], features)
            folds.append({
                "fold": int(fold),
                "held_dimension": int(held_d),
                "n_train": int(tr.sum()),
                "n_valid": int(va.sum()),
                "mae": float(mean_absolute_error(y[va], oof[va])),
                "r2": float(r2_score(y[va], oof[va])),
            })
        regrets = _regret_for_predictions(cells, curve_means, oof)
        mae = float(mean_absolute_error(y, oof))
        r2 = float(r2_score(y, oof))
        predictions[f"pred_{model_name}"] = oof
        predictions[f"regret_{model_name}"] = regrets
        final_model = _fit(X, y, features)
        model_metrics[model_name] = {
            "features": list(features),
            "oof_mae": mae,
            "oof_r2": r2,
            "oof_median_regret": float(np.median(regrets)),
            "oof_p90_regret": float(np.quantile(regrets, 0.90)),
            "folds": folds,
            "all_discovery_fit": _model_spec(final_model, features),
        }

    return {
        "target": "tau_0_5",
        "cv": "leave_one_dimension_out",
        "dimensions": dims,
        "n_cells": int(len(cells)),
        "models": model_metrics,
    }, predictions


def fit_all_discovery_hinge(cells: pd.DataFrame) -> dict[str, Any]:
    X = feature_frame(cells)
    y = cells["tau_0_5"].to_numpy(float)
    features = MODEL_FEATURES["M_hinge"]
    return _model_spec(_fit(X, y, features), features)
