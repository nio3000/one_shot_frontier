from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf


@dataclass
class LDAHead:
    means: np.ndarray
    cov: np.ndarray
    priors: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    shrinkage: float | None = None


@dataclass
class QDAHead:
    means: np.ndarray
    covs: np.ndarray
    priors: np.ndarray
    chols: list[np.ndarray]
    logdets: np.ndarray
    shrinkages: np.ndarray | None = None


def _floor_cov(cov: np.ndarray, rel_floor: float) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    scale = float(np.trace(cov) / cov.shape[0])
    floor = max(rel_floor * max(scale, 1e-12), np.finfo(float).eps)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, floor)
    return (eigvecs * eigvals) @ eigvecs.T


def build_population_lda(means: np.ndarray, cov: np.ndarray, priors: np.ndarray, rel_floor: float = 1e-12) -> LDAHead:
    cov = _floor_cov(cov, rel_floor)
    coef = np.linalg.solve(cov, means.T).T
    intercept = -0.5 * np.einsum("cd,cd->c", means, coef) + np.log(priors)
    return LDAHead(means, cov, priors, coef, intercept, None)


def build_population_qda(means: np.ndarray, covs: np.ndarray, priors: np.ndarray, rel_floor: float = 1e-12) -> QDAHead:
    clean = []
    chols = []
    logdets = []
    for cov in covs:
        c = _floor_cov(cov, rel_floor)
        chol = np.linalg.cholesky(c)
        clean.append(c)
        chols.append(chol)
        logdets.append(2.0 * np.log(np.diag(chol)).sum())
    return QDAHead(means, np.stack(clean), priors, chols, np.asarray(logdets), None)


def fit_o4_ledoit_wolf(X: np.ndarray, y: np.ndarray, C: int, rel_floor: float) -> LDAHead:
    means = np.stack([X[y == c].mean(axis=0) for c in range(C)])
    residuals = np.vstack([X[y == c] - means[c] for c in range(C)])
    lw = LedoitWolf(assume_centered=True).fit(residuals)
    cov = _floor_cov(lw.covariance_, rel_floor)
    priors = np.asarray([(y == c).mean() for c in range(C)], dtype=float)
    coef = np.linalg.solve(cov, means.T).T
    intercept = -0.5 * np.einsum("cd,cd->c", means, coef) + np.log(priors)
    return LDAHead(means, cov, priors, coef, intercept, float(lw.shrinkage_))


def fit_o5_ledoit_wolf(X: np.ndarray, y: np.ndarray, C: int, rel_floor: float) -> QDAHead:
    means, covs, chols, logdets, shrinkages = [], [], [], [], []
    priors = []
    for c in range(C):
        xc = X[y == c]
        mu = xc.mean(axis=0)
        res = xc - mu
        lw = LedoitWolf(assume_centered=True).fit(res)
        cov = _floor_cov(lw.covariance_, rel_floor)
        chol = np.linalg.cholesky(cov)
        means.append(mu)
        covs.append(cov)
        chols.append(chol)
        logdets.append(2.0 * np.log(np.diag(chol)).sum())
        shrinkages.append(float(lw.shrinkage_))
        priors.append(len(xc) / len(X))
    return QDAHead(
        np.stack(means), np.stack(covs), np.asarray(priors), chols,
        np.asarray(logdets), np.asarray(shrinkages)
    )


def predict_lda(head: LDAHead, X: np.ndarray) -> np.ndarray:
    scores = X @ head.coef.T + head.intercept
    return scores.argmax(axis=1)


def predict_qda(head: QDAHead, X: np.ndarray) -> np.ndarray:
    scores = np.empty((len(X), len(head.means)), dtype=float)
    for c, (mu, chol, logdet, prior) in enumerate(zip(head.means, head.chols, head.logdets, head.priors)):
        diff_t = (X - mu).T
        whitened = solve_triangular(chol, diff_t, lower=True, check_finite=False, overwrite_b=False)
        quad = np.einsum("dn,dn->n", whitened, whitened)
        scores[:, c] = -0.5 * (quad + logdet) + np.log(prior)
    return scores.argmax(axis=1)


def accuracy_lda(X: np.ndarray, y: np.ndarray, means: np.ndarray, cov: np.ndarray) -> float:
    priors = np.ones(len(means)) / len(means)
    head = build_population_lda(means, cov, priors)
    return float((predict_lda(head, X) == y).mean())


def condition_number(cov: np.ndarray) -> float:
    eigs = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    eigs = np.maximum(eigs, np.finfo(float).tiny)
    return float(eigs.max() / eigs.min())
