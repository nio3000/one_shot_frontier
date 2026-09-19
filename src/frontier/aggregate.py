from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def aggregate_runs(df: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    group_cols = [
        "phase", "generator_family", "condition_id", "C", "d", "target_reff",
        "target_H_sigma", "requested_effective_support"
    ]
    rows = []
    alpha = 1.0 - ci_level
    for keys, g in df.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        n = len(g)
        delta = g["delta_o5_minus_o4"].to_numpy(float)
        mean = float(delta.mean())
        sd = float(delta.std(ddof=1)) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 0 else np.nan
        crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1)) if n > 1 else np.nan
        row.update({
            "n_runs": n,
            "actual_reff_mean": float(g["actual_reff"].mean()),
            "actual_H_sigma_mean": float(g["actual_H_sigma"].mean()),
            "effective_support_mean": float(g["effective_support"].mean()),
            "delta_mean": mean,
            "delta_sd": sd,
            "delta_ci_low": mean - crit * se if n > 1 else np.nan,
            "delta_ci_high": mean + crit * se if n > 1 else np.nan,
            "oracle_expressivity_gain_mean": float(g["oracle_expressivity_gain"].mean()),
            "extra_o5_estimation_burden_mean": float(g["extra_o5_estimation_burden"].mean()),
            "oracle_o4_acc_mean": float(g["oracle_o4_acc"].mean()),
            "oracle_o5_acc_mean": float(g["oracle_o5_acc"].mean()),
            "estimated_o4_acc_mean": float(g["estimated_o4_acc"].mean()),
            "estimated_o5_acc_mean": float(g["estimated_o5_acc"].mean()),
            "object_recovery_max_abs_max": float(g["object_recovery_max_abs"].max()),
            "decomposition_residual_max_abs": float(np.abs(g["decomposition_residual"]).max()),
            "o4_condition_number_mean": float(g["o4_condition_number"].mean()),
            "o5_condition_number_mean": float(g["o5_condition_number_mean"].mean()),
            "o5_condition_number_max": float(g["o5_condition_number_max"].max()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def linear_zero_crossing(s: np.ndarray, y: np.ndarray) -> float | None:
    order = np.argsort(s)
    s, y = s[order], y[order]
    for i in range(len(s) - 1):
        if y[i] == 0:
            return float(s[i])
        if y[i] < 0 <= y[i + 1]:
            if y[i + 1] == y[i]:
                return float(s[i + 1])
            t = -y[i] / (y[i + 1] - y[i])
            return float(s[i] + t * (s[i + 1] - s[i]))
    return None
