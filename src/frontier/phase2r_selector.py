from __future__ import annotations

import numpy as np

from .phase1_models import covariance_family
from .phase1_objects import SufficientStats
from .phase2_selector import FoldedObject, select_tau_folded_summary_cv


OBJECTIVES = (
    "nll_control",
    "mean_pairwise_expected_margin",
    "hard_competitor_expected_margin",
)


def _expected_discriminant_matrix(
    train: SufficientStats,
    val: SufficientStats,
    tau: float,
    alpha0: float,
) -> np.ndarray:
    """E[g_k(X) | true class c] for every true class c and model class k.

    Uses only held-out count/sum/second-moment sufficient statistics.
    """
    covs = covariance_family(train, float(tau), float(alpha0))
    C = len(train.counts)
    out = np.empty((C, C), dtype=np.float64)

    invs = []
    logdets = []
    constants = []
    for k in range(C):
        eig, Q = np.linalg.eigh(covs[k])
        if eig.min() <= 0 or not np.isfinite(eig).all():
            return np.full((C, C), np.nan)
        inv = (Q / eig) @ Q.T
        logdet = float(np.log(eig).sum())
        mu = train.means[k]
        const = (
            -0.5 * float(mu @ inv @ mu)
            -0.5 * logdet
            + np.log(max(float(train.priors[k]), 1e-15))
        )
        invs.append(inv)
        logdets.append(logdet)
        constants.append(const)

    for c in range(C):
        n = int(val.counts[c])
        if n <= 0:
            out[c].fill(np.nan)
            continue
        ex = val.sums[c] / n
        exx = val.second[c] / n
        for k in range(C):
            inv = invs[k]
            mu = train.means[k]
            # E[-.5 (x-mu)' inv (x-mu) -.5 logdet + log prior]
            quad = float(np.sum(inv * exx.T) - 2.0 * (mu @ inv @ ex) + mu @ inv @ mu)
            out[c, k] = -0.5 * quad - 0.5 * logdets[k] + np.log(max(float(train.priors[k]), 1e-15))
    return out


def _margin_score(train: SufficientStats, val: SufficientStats, tau: float, alpha0: float, hard: bool) -> float:
    E = _expected_discriminant_matrix(train, val, tau, alpha0)
    if not np.isfinite(E).all():
        return float("-inf")

    per_class = []
    C = E.shape[0]
    for c in range(C):
        margins = np.array([E[c, c] - E[c, j] for j in range(C) if j != c], dtype=np.float64)
        if hard:
            per_class.append(float(margins.min()))
        else:
            per_class.append(float(margins.mean()))
    return float(np.mean(per_class))


def objective_curve(
    obj: FoldedObject,
    taus: list[float],
    alpha0: float,
    objective_id: str,
) -> dict[float, float]:
    if len(obj.fold_stats) != 2:
        raise ValueError("Phase 2-R requires exactly two folds")
    a, b = obj.fold_stats

    if objective_id == "nll_control":
        _, scores = select_tau_folded_summary_cv(obj, taus, alpha0)
        return {float(t): float(v) for t, v in scores.items()}

    if objective_id not in OBJECTIVES:
        raise ValueError(f"Unknown objective_id={objective_id}")

    hard = objective_id == "hard_competitor_expected_margin"
    out = {}
    for tau in map(float, taus):
        out[tau] = 0.5 * (
            _margin_score(a, b, tau, alpha0, hard)
            + _margin_score(b, a, tau, alpha0, hard)
        )
    return out


def select_tau_objective(
    obj: FoldedObject,
    taus: list[float],
    alpha0: float,
    objective_id: str,
) -> tuple[float, dict[float, float]]:
    scores = objective_curve(obj, taus, alpha0, objective_id)
    if objective_id == "nll_control":
        best = min(scores.values())
        selected = min(t for t in sorted(scores) if abs(scores[t] - best) <= 1e-12)
    else:
        best = max(scores.values())
        selected = min(t for t in sorted(scores) if abs(scores[t] - best) <= 1e-12)
    return float(selected), scores
