from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .phase0s_predictor import predict_spec


def _nearest_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.asarray([float(grid[np.argmin(np.abs(grid - float(v)))]) for v in values])


def _regime(t: float) -> str:
    if t <= 0.2 + 1e-12:
        return "LOW"
    if t >= 0.8 - 1e-12:
        return "HIGH"
    return "MID"


def _prediction_table(cells: pd.DataFrame, curve_means: pd.DataFrame, spec: dict[str, Any], label: str) -> pd.DataFrame:
    out = cells.copy()
    pred = predict_spec(out, spec)
    grid = np.sort(curve_means["tau"].unique().astype(float))
    pred_grid = _nearest_grid(pred, grid)
    out[f"pred_{label}"] = pred
    out[f"pred_grid_{label}"] = pred_grid
    regrets = []
    for (_, row), t in zip(out.iterrows(), pred_grid):
        mask = (
            (curve_means["panel"] == row["panel"])
            & (curve_means["d"].astype(int) == int(row["d"]))
            & np.isclose(curve_means["q_target"], float(row["q_target"]))
            & np.isclose(curve_means["target_H_sigma"], float(row["target_H_sigma"]))
            & np.isclose(curve_means["requested_effective_support"], float(row["requested_effective_support"]))
        )
        g = curve_means[mask]
        best = float(g["accuracy_mean"].max())
        chosen = float(g.loc[np.isclose(g["tau"], t), "accuracy_mean"].iloc[0])
        regrets.append(best - chosen)
    out[f"regret_{label}"] = np.asarray(regrets, dtype=float)
    return out


def _bootstrap_mae_ci(obs: np.ndarray, pred: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    n = len(obs)
    vals = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        vals[i] = np.mean(np.abs(obs[idx] - pred[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def evaluate_panel(
    cells: pd.DataFrame,
    curve_means: pd.DataFrame,
    predictor_freeze: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    models = predictor_freeze["models"]
    table = cells.copy()
    metrics = {}
    for label, spec in models.items():
        tmp = _prediction_table(cells, curve_means, spec, label)
        table[f"pred_{label}"] = tmp[f"pred_{label}"]
        table[f"pred_grid_{label}"] = tmp[f"pred_grid_{label}"]
        table[f"regret_{label}"] = tmp[f"regret_{label}"]
        obs = table["tau_0_5"].to_numpy(float)
        pred = table[f"pred_{label}"].to_numpy(float)
        metrics[label] = {
            "mae": float(mean_absolute_error(obs, pred)),
            "r2": float(r2_score(obs, pred)),
            "median_regret": float(np.median(table[f"regret_{label}"])),
            "p90_regret": float(np.quantile(table[f"regret_{label}"], 0.90)),
            "max_regret": float(np.max(table[f"regret_{label}"])),
        }

    primary = "M_old"
    obs = table["tau_0_5"].to_numpy(float)
    pred = table[f"pred_{primary}"].to_numpy(float)
    bcfg = protocol["bootstrap"]
    ci_lo, ci_hi = _bootstrap_mae_ci(obs, pred, int(bcfg["structural_cell_replicates"]), int(bcfg["seed"]))
    pred_regime = np.asarray([_regime(t) for t in table[f"pred_grid_{primary}"]])
    obs_regime = np.asarray([_regime(t) for t in table["tau_0_5"]])
    regime_acc = float(np.mean(pred_regime == obs_regime))
    table["observed_regime"] = obs_regime
    table["predicted_regime"] = pred_regime

    g = protocol["panel_gates"]
    m = metrics[primary]
    s1 = m["mae"] <= float(g["S1"]["tau_mae_max"]) and ci_hi <= float(g["S1"]["tau_mae_ci95_upper_max"]) and m["r2"] >= float(g["S1"]["r2_min"])
    s2 = m["median_regret"] <= float(g["S2"]["median_regret_max"]) and m["p90_regret"] <= float(g["S2"]["p90_regret_max"])
    s3 = regime_acc >= float(g["S3"]["low_mid_high_accuracy_min"])
    best_single = min(metrics["B1_H_only"]["mae"], metrics["B2_Seff_only"]["mae"])
    s4 = m["mae"] <= float(g["S4"]["Mold_mae_relative_to_best_single_axis_max"]) * best_single and m["mae"] < metrics["B0_constant"]["mae"]

    result = {
        "S1": {"pass": bool(s1), "mae": m["mae"], "mae_bootstrap_ci95": [ci_lo, ci_hi], "r2": m["r2"], "thresholds": g["S1"]},
        "S2": {"pass": bool(s2), "median_regret": m["median_regret"], "p90_regret": m["p90_regret"], "max_regret": m["max_regret"], "thresholds": g["S2"]},
        "S3": {"pass": bool(s3), "regime_accuracy": regime_acc, "threshold": float(g["S3"]["low_mid_high_accuracy_min"])},
        "S4": {"pass": bool(s4), "M_old_mae": m["mae"], "B0_mae": metrics["B0_constant"]["mae"], "B1_mae": metrics["B1_H_only"]["mae"], "B2_mae": metrics["B2_Seff_only"]["mae"], "best_single_mae": best_single, "ratio_to_best_single": float(m["mae"] / best_single) if best_single > 0 else float("inf"), "thresholds": g["S4"]},
        "models": metrics,
        "pass": bool(s1 and s2 and s3 and s4),
    }
    return result, table


def global_decision(panel_a: dict[str, Any], panel_b: dict[str, Any], panel_c: dict[str, Any]) -> str:
    a, b, c = bool(panel_a["pass"]), bool(panel_b["pass"]), bool(panel_c["pass"])
    if a and b and c:
        return "SMOOTH_ESTIMABILITY_SURFACE_TRANSFER_SUPPORTED"
    if a and b and not c:
        return "GAUSSIAN_GEOMETRY_TRANSFER_SUPPORTED_NON_GAUSSIAN_FAIL"
    if a and not b:
        return "SCALE_TRANSFER_SUPPORTED_GEOMETRY_NOT_GENERAL"
    return "SMOOTH_SURFACE_NOT_TRANSFERABLE"
