from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(np.mean(x)) if n else float("nan")
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n <= 1:
        return mean, sd, float("nan"), float("nan")
    se = sd / math.sqrt(n)
    crit = float(stats.t.ppf(0.5 + level / 2.0, n - 1))
    return mean, sd, mean - crit * se, mean + crit * se


def aggregate_curve_points(curves: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    cols = [
        "run_stage", "estimator_family", "generator_family", "C", "d",
        "target_reff", "target_H_sigma", "requested_effective_support", "alpha", "tau"
    ]
    rows = []
    for keys, g in curves.groupby(cols, dropna=False, sort=True):
        row = dict(zip(cols, keys))
        mean, sd, lo, hi = _mean_ci(g["accuracy"].to_numpy(float), ci_level)
        row.update({
            "n_runs": int(len(g)),
            "accuracy_mean": mean,
            "accuracy_sd": sd,
            "accuracy_ci_low": lo,
            "accuracy_ci_high": hi,
            "actual_H_sigma_mean": float(g["actual_H_sigma"].mean()),
            "effective_support_mean": float(g["effective_support"].mean()),
            "actual_reff_mean": float(g["actual_reff"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_diff_ci(curves: pd.DataFrame, tau_a: float, tau_b: float, ci_level: float = 0.95) -> tuple[float, float, float, int]:
    a = curves[np.isclose(curves["tau"], tau_a)][["seed", "accuracy"]].rename(columns={"accuracy": "a"})
    b = curves[np.isclose(curves["tau"], tau_b)][["seed", "accuracy"]].rename(columns={"accuracy": "b"})
    p = a.merge(b, on="seed", how="inner")
    d = (p["a"] - p["b"]).to_numpy(float)
    mean, _, lo, hi = _mean_ci(d, ci_level)
    return mean, lo, hi, len(d)


def summarize_fixed_ridge_cells(curves: pd.DataFrame, tolerance: float = 0.005, ci_level: float = 0.95) -> pd.DataFrame:
    fixed = curves[curves["estimator_family"] == "fixed_ridge"].copy()
    group_cols = [
        "run_stage", "generator_family", "C", "d", "target_reff",
        "target_H_sigma", "requested_effective_support", "alpha"
    ]
    rows = []
    for keys, g in fixed.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        means = g.groupby("tau")["accuracy"].mean().sort_index()
        max_acc = float(means.max())
        eligible = means[means >= max_acc - float(tolerance)]
        tau_05 = float(eligible.index.min())
        best_tau = float(means.idxmax())
        interior = means.loc[(means.index > 0) & (means.index < 1)]
        endpoints = means.loc[(np.isclose(means.index, 0)) | (np.isclose(means.index, 1))]
        best_interior_tau = float(interior.idxmax())
        best_endpoint_tau = float(endpoints.idxmax())
        gain = float(interior.max() - endpoints.max())
        dmean, dlo, dhi, n_pair = _paired_diff_ci(g, best_interior_tau, best_endpoint_tau, ci_level)
        endpoint_delta, edlo, edhi, _ = _paired_diff_ci(g, 1.0, 0.0, ci_level)
        row.update({
            "n_runs": int(g["seed"].nunique()),
            "actual_H_sigma_mean": float(g["actual_H_sigma"].mean()),
            "effective_support_mean": float(g["effective_support"].mean()),
            "actual_reff_mean": float(g["actual_reff"].mean()),
            "best_accuracy_mean": max_acc,
            "best_tau": best_tau,
            "tau_0_5": tau_05,
            "best_interior_tau": best_interior_tau,
            "best_endpoint_tau": best_endpoint_tau,
            "interior_gain": gain,
            "interior_gain_paired_mean": dmean,
            "interior_gain_ci_low": dlo,
            "interior_gain_ci_high": dhi,
            "paired_n": n_pair,
            "endpoint_delta_tau1_minus_tau0": endpoint_delta,
            "endpoint_delta_ci_low": edlo,
            "endpoint_delta_ci_high": edhi,
            "tau0_accuracy_mean": float(means.loc[0.0]),
            "tau1_accuracy_mean": float(means.loc[1.0]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_lw_cells(curves: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    lw = curves[curves["estimator_family"] == "adaptive_lw"].copy()
    group_cols = [
        "run_stage", "generator_family", "C", "d", "target_reff",
        "target_H_sigma", "requested_effective_support"
    ]
    rows = []
    for keys, g in lw.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        dmean, dlo, dhi, n = _paired_diff_ci(g, 1.0, 0.0, ci_level)
        means = g.groupby("tau")["accuracy"].mean().sort_index()
        row.update({
            "n_runs": int(n),
            "actual_H_sigma_mean": float(g["actual_H_sigma"].mean()),
            "effective_support_mean": float(g["effective_support"].mean()),
            "endpoint_delta_tau1_minus_tau0": dmean,
            "endpoint_delta_ci_low": dlo,
            "endpoint_delta_ci_high": dhi,
            "tau0_accuracy_mean": float(means.loc[0.0]),
            "tau1_accuracy_mean": float(means.loc[1.0]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def accuracy_for_predicted_tau(curve_means: pd.DataFrame, target_tau: float) -> float:
    taus = curve_means["tau"].to_numpy(float)
    idx = int(np.argmin(np.abs(taus - float(target_tau))))
    chosen = float(taus[idx])
    return float(curve_means.loc[np.isclose(curve_means["tau"], chosen), "accuracy_mean"].iloc[0])
