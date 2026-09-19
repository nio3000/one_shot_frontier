from __future__ import annotations

import numpy as np

from .phase1_objects import SufficientStats


def covariance_family(stats: SufficientStats, tau: float, alpha0: float) -> np.ndarray:
    d = stats.pooled_cov.shape[0]
    ridge = float(alpha0) * max(float(stats.mean_variance_pool), 1e-12)
    I = np.eye(d)
    covs = (1.0 - tau) * stats.pooled_cov[None, :, :] + tau * stats.class_covs
    covs = covs + ridge * I[None, :, :]
    return 0.5 * (covs + np.swapaxes(covs, 1, 2))


def gaussian_scores(X: np.ndarray, means: np.ndarray, priors: np.ndarray, covs: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    C = means.shape[0]
    out = np.empty((len(X), C), dtype=np.float64)
    for c in range(C):
        eig, Q = np.linalg.eigh(covs[c])
        if eig.min() <= 0 or not np.isfinite(eig).all():
            raise ValueError(f"Non-SPD covariance for class {c}")
        z = (X - means[c]) @ Q
        mahal = np.sum((z * z) / eig, axis=1)
        logdet = float(np.log(eig).sum())
        out[:, c] = -0.5 * (mahal + logdet) + np.log(max(float(priors[c]), 1e-15))
    return out


def predict_tau_path(stats: SufficientStats, X: np.ndarray, taus: list[float], alpha0: float) -> dict[float, np.ndarray]:
    pred: dict[float, np.ndarray] = {}
    for tau in taus:
        covs = covariance_family(stats, float(tau), float(alpha0))
        pred[float(tau)] = gaussian_scores(X, stats.means, stats.priors, covs).argmax(axis=1).astype(np.int64)
    return pred


def predict_ncm(stats: SufficientStats, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    dist = ((X[:, None, :] - stats.means[None, :, :]) ** 2).sum(axis=2)
    return dist.argmin(axis=1).astype(np.int64)


def predict_diagonal_gaussian(stats: SufficientStats, X: np.ndarray, alpha0: float) -> np.ndarray:
    d = stats.pooled_cov.shape[0]
    ridge = float(alpha0) * max(float(stats.mean_variance_pool), 1e-12)
    vars_ = np.diagonal(stats.class_covs, axis1=1, axis2=2) + ridge
    vars_ = np.maximum(vars_, 1e-12)
    out = np.empty((len(X), len(stats.counts)), dtype=np.float64)
    for c in range(len(stats.counts)):
        z = X - stats.means[c]
        out[:, c] = -0.5 * (np.sum(z * z / vars_[c], axis=1) + np.log(vars_[c]).sum()) + np.log(max(stats.priors[c], 1e-15))
    return out.argmax(axis=1).astype(np.int64)
