from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from .phase1_eval import metrics, stratified_paired_bootstrap_bacc_delta


def evaluate_adaptive_condition(
    y: np.ndarray,
    pred_by_tau: dict[float, np.ndarray],
    selected_tau: float,
    endpoint_candidates: list[float],
    oracle_interior_candidates: list[float],
    n_boot: int,
    bootstrap_seed: int,
    oracle_opportunity_min_gain: float,
) -> dict[str, object]:
    y = np.asarray(y, dtype=np.int64)
    selected_tau = float(selected_tau)
    adaptive_pred = pred_by_tau[selected_tau]
    adaptive_m = metrics(y, adaptive_pred)
    endpoint_scores = {float(t): float(balanced_accuracy_score(y, pred_by_tau[float(t)])) for t in endpoint_candidates}
    best_endpoint_tau = min(t for t, s in endpoint_scores.items() if s == max(endpoint_scores.values()))
    best_endpoint_pred = pred_by_tau[best_endpoint_tau]
    best_endpoint_bacc = endpoint_scores[best_endpoint_tau]
    all_scores = {float(t): float(balanced_accuracy_score(y, p)) for t, p in pred_by_tau.items()}
    oracle_tau = min(t for t, s in all_scores.items() if s == max(all_scores.values()))
    oracle_bacc = all_scores[oracle_tau]
    interior_scores = {float(t): all_scores[float(t)] for t in oracle_interior_candidates}
    oracle_interior_tau = min(t for t, s in interior_scores.items() if s == max(interior_scores.values()))
    oracle_interior_bacc = interior_scores[oracle_interior_tau]
    adaptive_gain = float(adaptive_m.balanced_accuracy - best_endpoint_bacc)
    oracle_gain = float(oracle_interior_bacc - best_endpoint_bacc)
    opportunity = bool(oracle_gain >= float(oracle_opportunity_min_gain))
    recovery = float(adaptive_gain / oracle_gain) if opportunity and oracle_gain > 0 else float("nan")
    ci_lo, ci_hi = stratified_paired_bootstrap_bacc_delta(
        y, adaptive_pred, best_endpoint_pred, int(n_boot), int(bootstrap_seed)
    )
    return {
        "selected_tau": selected_tau,
        "adaptive_bacc": float(adaptive_m.balanced_accuracy),
        "adaptive_accuracy": float(adaptive_m.accuracy),
        "adaptive_macro_f1": float(adaptive_m.macro_f1),
        "best_endpoint_tau_test_oracle": float(best_endpoint_tau),
        "best_endpoint_bacc_test_oracle": float(best_endpoint_bacc),
        "adaptive_gain_vs_best_endpoint": adaptive_gain,
        "adaptive_gain_ci95_low": float(ci_lo),
        "adaptive_gain_ci95_high": float(ci_hi),
        "oracle_best_tau_test": float(oracle_tau),
        "oracle_best_bacc_test": float(oracle_bacc),
        "oracle_interior_tau_test": float(oracle_interior_tau),
        "oracle_interior_bacc_test": float(oracle_interior_bacc),
        "oracle_interior_gain_vs_best_endpoint": oracle_gain,
        "oracle_opportunity": opportunity,
        "oracle_gain_recovery_fraction": recovery,
    }
