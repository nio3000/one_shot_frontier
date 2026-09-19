from __future__ import annotations

import math
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


STRUCT_COLS = [
    "run_stage", "generator_family", "C", "d", "q_target", "target_reff",
    "target_H_sigma", "requested_rho",
]


def aggregate_curve_points(curves: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    rows = []
    for keys, g in curves.groupby([*STRUCT_COLS, "tau"], dropna=False, sort=True):
        row = dict(zip([*STRUCT_COLS, "tau"], keys))
        mean, sd, lo, hi = _mean_ci(g["accuracy"].to_numpy(float), ci_level)
        row.update({
            "n_runs": int(g["seed"].nunique()),
            "accuracy_mean": mean,
            "accuracy_sd": sd,
            "accuracy_ci_low": lo,
            "accuracy_ci_high": hi,
            "actual_reff_mean": float(g["actual_reff"].mean()),
            "actual_H_sigma_mean": float(g["actual_H_sigma"].mean()),
            "actual_rho_mean": float(g["actual_rho"].mean()),
            "effective_support_mean": float(g["effective_support"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_oracle_curve_points(curves: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    rows = []
    for keys, g in curves.groupby([*STRUCT_COLS, "tau"], dropna=False, sort=True):
        row = dict(zip([*STRUCT_COLS, "tau"], keys))
        mean, sd, lo, hi = _mean_ci(g["oracle_accuracy"].to_numpy(float), ci_level)
        row.update({
            "n_runs": int(g["seed"].nunique()),
            "oracle_accuracy_mean": mean,
            "oracle_accuracy_sd": sd,
            "oracle_accuracy_ci_low": lo,
            "oracle_accuracy_ci_high": hi,
            "actual_reff_mean": float(g["actual_reff"].mean()),
            "actual_H_sigma_mean": float(g["actual_H_sigma"].mean()),
            "actual_rho_mean": float(g["actual_rho"].mean()),
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
    tau_05 = float(eligible.index.min())
    best_tau = float(means.idxmax())
    interior = means.loc[(means.index > 0) & (means.index < 1)]
    endpoints = means.loc[(np.isclose(means.index, 0)) | (np.isclose(means.index, 1))]
    best_interior_tau = float(interior.idxmax())
    best_endpoint_tau = float(endpoints.idxmax())
    return tau_05, best_tau, best_interior_tau, best_endpoint_tau, float(interior.max() - endpoints.max())


def summarize_cells(
    curves: pd.DataFrame,
    oracle_curves: pd.DataFrame,
    tolerance: float = 0.005,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    oracle_grouped = {k: g for k, g in oracle_curves.groupby(STRUCT_COLS, dropna=False, sort=True)}
    rows = []
    for keys, g in curves.groupby(STRUCT_COLS, dropna=False, sort=True):
        row = dict(zip(STRUCT_COLS, keys))
        means = g.groupby("tau")["accuracy"].mean().sort_index()
        tau05, best_tau, bi, be, gain = _curve_target(means, tolerance)
        dmean, dlo, dhi, n = _paired_diff_ci(g, bi, be, ci_level)
        endpoint_delta, edlo, edhi, _ = _paired_diff_ci(g, 1.0, 0.0, ci_level)

        og = oracle_grouped[keys]
        omeans = og.groupby("tau")["oracle_accuracy"].mean().sort_index()
        otau05, obest, _, _, _ = _curve_target(omeans, tolerance)

        row.update({
            "n_runs": int(g["seed"].nunique()),
            "actual_reff_mean": float(g["actual_reff"].mean()),
            "actual_H_sigma_mean": float(g["actual_H_sigma"].mean()),
            "actual_rho_mean": float(g["actual_rho"].mean()),
            "q_actual_mean": float(g["actual_reff"].mean() / float(row["d"])),
            "effective_support_mean": float(g["effective_support"].mean()),
            "best_accuracy_mean": float(means.max()),
            "best_tau": best_tau,
            "tau_0_5": tau05,
            "best_interior_tau": bi,
            "best_endpoint_tau": be,
            "interior_gain": gain,
            "interior_gain_paired_mean": dmean,
            "interior_gain_ci_low": dlo,
            "interior_gain_ci_high": dhi,
            "paired_n": n,
            "endpoint_delta_tau1_minus_tau0": endpoint_delta,
            "endpoint_delta_ci_low": edlo,
            "endpoint_delta_ci_high": edhi,
            "tau0_accuracy_mean": float(means.loc[0.0]),
            "tau1_accuracy_mean": float(means.loc[1.0]),
            "oracle_tau_0_5": otau05,
            "oracle_best_tau": obest,
            "oracle_best_accuracy_mean": float(omeans.max()),
            "oracle_tau0_accuracy_mean": float(omeans.loc[0.0]),
            "oracle_tau1_accuracy_mean": float(omeans.loc[1.0]),
            "estimated_minus_oracle_tau_0_5": float(tau05 - otau05),
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
            if c in g:
                row[c + "_mean"] = float(g[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)
