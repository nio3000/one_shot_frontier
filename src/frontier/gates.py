from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .aggregate import linear_zero_crossing


def _features(df: pd.DataFrame) -> np.ndarray:
    s = np.log2(df["effective_support_mean"].to_numpy(float))
    h = np.log1p(10.0 * df["actual_H_sigma_mean"].to_numpy(float))
    return np.column_stack([s, h, s * h])


def evaluate_f1(agg: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    min_effect = float(cfg["min_effect_pp"]) / 100.0
    low_max = float(cfg["low_support_max"])
    high_min = float(cfg["high_support_min"])
    need = int(cfg["min_consecutive_nonzero_H_levels"])
    passed_levels = []
    details = []
    for h, g in agg[agg["target_H_sigma"] > 0].groupby("target_H_sigma"):
        low = g[g["effective_support_mean"] <= low_max]
        high = g[g["effective_support_mean"] >= high_min]
        o4_low = low[(low["delta_mean"] <= -min_effect) & (low["delta_ci_high"] < 0)]
        o5_high = high[(high["delta_mean"] >= min_effect) & (high["delta_ci_low"] > 0)]
        ok = (len(o4_low) > 0) and (len(o5_high) > 0)
        details.append({"H": float(h), "pass": bool(ok), "n_o4_low": len(o4_low), "n_o5_high": len(o5_high)})
        if ok:
            passed_levels.append(float(h))
    hs = sorted(agg.loc[agg["target_H_sigma"] > 0, "target_H_sigma"].unique())
    max_consecutive = 0
    cur = 0
    pset = set(passed_levels)
    for h in hs:
        if float(h) in pset:
            cur += 1
            max_consecutive = max(max_consecutive, cur)
        else:
            cur = 0
    return {"pass": max_consecutive >= need, "max_consecutive_levels": max_consecutive, "details": details}


def evaluate_f2(agg: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    hs, stars = [], []
    detail = []
    for h, g in agg[agg["target_H_sigma"] > 0].groupby("target_H_sigma"):
        s = g["effective_support_mean"].to_numpy(float)
        y = g["delta_mean"].to_numpy(float)
        star = linear_zero_crossing(s, y)
        detail.append({"H": float(h), "S_star": star})
        if star is not None and star > 0:
            hs.append(float(h)); stars.append(float(star))
    min_n = int(cfg["min_finite_crossovers"])
    if len(hs) < min_n:
        return {"pass": False, "rho": None, "n_finite_crossovers": len(hs), "details": detail}
    rho = float(spearmanr(hs, np.log2(stars)).statistic)
    threshold = float(cfg["max_spearman_rho_H_vs_log2_Sstar"])
    return {"pass": rho <= threshold, "rho": rho, "threshold": threshold, "n_finite_crossovers": len(hs), "details": detail}


def fit_predictor(discovery_agg: pd.DataFrame) -> LogisticRegression:
    X = _features(discovery_agg)
    y = (discovery_agg["delta_mean"].to_numpy(float) > 0).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Discovery cells contain only one winner class; frontier predictor cannot be fit.")
    model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=0)
    model.fit(X, y)
    return model


def evaluate_f3(discovery_agg: pd.DataFrame, holdout_agg: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        model = fit_predictor(discovery_agg)
    except Exception as e:
        return {"pass": False, "error": str(e), "auroc": None, "sign_accuracy": None}
    Xh = _features(holdout_agg)
    yh = (holdout_agg["delta_mean"].to_numpy(float) > 0).astype(int)
    prob = model.predict_proba(Xh)[:, 1]
    pred = (prob >= 0.5).astype(int)
    sign_acc = float((pred == yh).mean())
    auroc = float(roc_auc_score(yh, prob)) if len(np.unique(yh)) == 2 else float("nan")
    passed = (not math.isnan(auroc)) and auroc >= float(cfg["min_auroc"]) and sign_acc >= float(cfg["min_sign_accuracy"])
    return {
        "pass": passed,
        "auroc": auroc,
        "sign_accuracy": sign_acc,
        "min_auroc": float(cfg["min_auroc"]),
        "min_sign_accuracy": float(cfg["min_sign_accuracy"]),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }


def sanity_summary(agg: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    s = protocol["sanity_checks"]
    h0 = agg[np.isclose(agg["target_H_sigma"], 0.0)]
    oracle_gap_pp = float((h0["oracle_o5_acc_mean"] - h0["oracle_o4_acc_mean"]).abs().max() * 100) if len(h0) else float("nan")
    obj_max = float(agg["object_recovery_max_abs_max"].max())
    decomp_max = float(agg["decomposition_residual_max_abs"].max())
    return {
        "S0_1_oracle_equal_H0_pass": oracle_gap_pp <= float(s["oracle_equal_H0_max_abs_pp"]),
        "S0_1_oracle_gap_H0_max_abs_pp": oracle_gap_pp,
        "S0_3_object_recovery_pass": obj_max <= float(s["object_recovery_max_abs"]),
        "S0_3_object_recovery_max_abs": obj_max,
        "decomposition_identity_pass": decomp_max <= 1e-12,
        "decomposition_residual_max_abs": decomp_max,
    }
