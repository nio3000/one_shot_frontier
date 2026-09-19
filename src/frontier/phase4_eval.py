from __future__ import annotations

import numpy as np

from .phase1_models import covariance_family


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    vals = []
    for c in np.unique(y):
        idx = y == c
        vals.append(float(np.mean(pred[idx] == c)))
    return float(np.mean(vals))


def _head_for_tau(stats, tau: float, alpha0: float):
    covs = covariance_family(stats, float(tau), float(alpha0))
    heads = []
    for c in range(len(stats.counts)):
        eig, Q = np.linalg.eigh(covs[c])
        if eig.min() <= 0 or not np.isfinite(eig).all():
            raise ValueError(f"non-SPD covariance for class {c}")
        heads.append((eig, Q, stats.means[c], float(np.log(eig).sum()), float(np.log(max(stats.priors[c], 1e-15)))))
    return heads


def _predict_with_head(X: np.ndarray, heads) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    scores = np.empty((len(X), len(heads)), dtype=np.float64)
    for c, (eig, Q, mu, logdet, logprior) in enumerate(heads):
        z = (X - mu) @ Q
        mahal = np.sum((z * z) / eig, axis=1)
        scores[:, c] = -0.5 * (mahal + logdet) + logprior
    return scores.argmax(axis=1).astype(np.int64)


def paired_risk_curves(stats, variants: dict[str, tuple[np.ndarray, np.ndarray]], taus: list[float], alpha0: float) -> dict[str, np.ndarray]:
    curves = {name: np.zeros(len(taus), dtype=np.float64) for name in variants}
    for i, tau in enumerate(taus):
        heads = _head_for_tau(stats, float(tau), float(alpha0))
        for name, (X, y) in variants.items():
            pred = _predict_with_head(X, heads)
            curves[name][i] = balanced_accuracy(y, pred)
    return curves


def optimal_tau_set(curve: np.ndarray, taus: list[float], tolerance: float = 1e-12) -> list[float]:
    curve = np.asarray(curve, dtype=np.float64)
    best = float(curve.max())
    return [float(taus[i]) for i, v in enumerate(curve) if best - float(v) <= float(tolerance)]


def deterministic_minimax_regret(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    a = np.asarray(curve_a, dtype=np.float64)
    b = np.asarray(curve_b, dtype=np.float64)
    ra = float(a.max()) - a
    rb = float(b.max()) - b
    return float(np.min(np.maximum(ra, rb)))


def randomized_minimax_regret(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    a = np.asarray(curve_a, dtype=np.float64)
    b = np.asarray(curve_b, dtype=np.float64)
    ra = float(a.max()) - a
    rb = float(b.max()) - b
    best = float("inf")
    m = len(ra)
    for i in range(m):
        best = min(best, max(float(ra[i]), float(rb[i])))
    for i in range(m):
        for j in range(i + 1, m):
            den = (ra[i] - ra[j]) - (rb[i] - rb[j])
            if abs(float(den)) <= 1e-15:
                continue
            p = float((rb[j] - ra[j]) / den)
            if -1e-12 <= p <= 1.0 + 1e-12:
                p = min(1.0, max(0.0, p))
                ea = p * float(ra[i]) + (1.0 - p) * float(ra[j])
                eb = p * float(rb[i]) + (1.0 - p) * float(rb[j])
                best = min(best, max(ea, eb))
    return float(best)


def pair_decision_metrics(curve_a: np.ndarray, curve_b: np.ndarray, taus: list[float]) -> dict[str, object]:
    opt_a = optimal_tau_set(curve_a, taus)
    opt_b = optimal_tau_set(curve_b, taus)
    return {
        "optimal_tau_set_original": opt_a,
        "optimal_tau_set_counterfactual": opt_b,
        "optimal_sets_disjoint": bool(len(set(opt_a).intersection(opt_b)) == 0),
        "deterministic_pair_minimax_regret": deterministic_minimax_regret(curve_a, curve_b),
        "randomized_pair_minimax_regret": randomized_minimax_regret(curve_a, curve_b),
        "curve_linf_shift": float(np.max(np.abs(np.asarray(curve_a) - np.asarray(curve_b)))),
        "best_bacc_original": float(np.max(curve_a)),
        "best_bacc_counterfactual": float(np.max(curve_b)),
    }
