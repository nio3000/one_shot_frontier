from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import stats

STRUCT_COLS = [
    "panel", "distribution", "covariance_geometry", "C", "d", "q_target", "target_reff",
    "target_H_sigma", "requested_effective_support",
]


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


def aggregate_curve_points(curves: pd.DataFrame, oracle: bool = False, ci_level: float = 0.95) -> pd.DataFrame:
    value = "oracle_accuracy" if oracle else "accuracy"
    prefix = "oracle_accuracy" if oracle else "accuracy"
    rows = []
    for keys, g in curves.groupby([*STRUCT_COLS, "tau"], dropna=False, sort=True):
        row = dict(zip([*STRUCT_COLS, "tau"], keys))
        mean, sd, lo, hi = _mean_ci(g[value].to_numpy(float), ci_level)
        row.update({
            "n_runs": int(g["seed"].nunique()),
            f"{prefix}_mean": mean,
            f"{prefix}_sd": sd,
            f"{prefix}_ci_low": lo,
            f"{prefix}_ci_high": hi,
            "actual_reff_mean": float(g["actual_reff"].mean()),
            "actual_H_sigma_mean": float(g["actual_H_sigma"].mean()),
            "effective_support_mean": float(g["effective_support"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_diff_ci(curves: pd.DataFrame, tau_a: float, tau_b: float, ci_level: float) -> tuple[float, float, float, int]:
    a = curves[np.isclose(curves["tau"], tau_a)][["seed", "accuracy"]].rename(columns={"accuracy": "a"})
    b = curves[np.isclose(curves["tau"], tau_b)][["seed", "accuracy"]].rename(columns={"accuracy": "b"})
    p = a.merge(b, on="seed", how="inner")
    diff = (p["a"] - p["b"]).to_numpy(float)
    mean, _, lo, hi = _mean_ci(diff, ci_level)
    return mean, lo, hi, int(len(diff))


def _curve_target(means: pd.Series, tolerance: float) -> tuple[float, float, float, float, float]:
    means = means.sort_index()
    max_acc = float(means.max())
    eligible = means[means >= max_acc - float(tolerance)]
    tau05 = float(eligible.index.min())
    best_tau = float(means.idxmax())
    interior = means.loc[(means.index > 0) & (means.index < 1)]
    endpoints = means.loc[(np.isclose(means.index, 0)) | (np.isclose(means.index, 1))]
    bi = float(interior.idxmax())
    be = float(endpoints.idxmax())
    return tau05, best_tau, bi, be, float(interior.max() - endpoints.max())


def summarize_cells(curves: pd.DataFrame, oracle_curves: pd.DataFrame, tolerance: float = 0.005, ci_level: float = 0.95) -> pd.DataFrame:
    oracle_map = {k: g for k, g in oracle_curves.groupby(STRUCT_COLS, dropna=False, sort=True)}
    rows = []
    for keys, g in curves.groupby(STRUCT_COLS, dropna=False, sort=True):
        row = dict(zip(STRUCT_COLS, keys))
        means = g.groupby("tau")["accuracy"].mean().sort_index()
        tau05, best_tau, bi, be, gain = _curve_target(means, tolerance)
        dmean, dlo, dhi, n = _paired_diff_ci(g, bi, be, ci_level)
        og = oracle_map[keys]
        omeans = og.groupby("tau")["oracle_accuracy"].mean().sort_index()
        otau05, obest, _, _, _ = _curve_target(omeans, tolerance)
        row.update({
            "n_runs": int(g["seed"].nunique()),
            "actual_reff_mean": float(g["actual_reff"].mean()),
            "actual_H_sigma_mean": float(g["actual_H_sigma"].mean()),
            "effective_support_mean": float(g["effective_support"].mean()),
            "q_actual_mean": float(g["actual_reff"].mean() / float(row["d"])),
            "best_accuracy_mean": float(means.max()),
            "tau_0_5": tau05,
            "best_tau": best_tau,
            "best_interior_tau": bi,
            "best_endpoint_tau": be,
            "interior_gain": gain,
            "interior_gain_paired_mean": dmean,
            "interior_gain_ci_low": dlo,
            "interior_gain_ci_high": dhi,
            "paired_n": n,
            "tau0_accuracy_mean": float(means.loc[0.0]),
            "tau1_accuracy_mean": float(means.loc[1.0]),
            "oracle_tau_0_5": otau05,
            "oracle_best_tau": obest,
            "oracle_best_accuracy_mean": float(omeans.max()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_mechanism_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "class_cov_fro_rel_error_mean", "class_cov_op_rel_error_mean",
        "pooled_cov_fro_rel_error", "pooled_cov_op_rel_error",
        "sample_cov_rank_fraction_mean", "regularized_class_condition_mean",
        "regularized_class_condition_max", "regularized_pooled_condition",
        "population_pooled_cov_relative_fro_error",
    ]
    rows = []
    for keys, g in metrics.groupby(STRUCT_COLS, dropna=False, sort=True):
        row = dict(zip(STRUCT_COLS, keys))
        row["n_runs"] = int(g["seed"].nunique())
        for c in numeric:
            row[c + "_mean"] = float(g[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)
