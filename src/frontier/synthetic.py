from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class PopulationGeometry:
    C: int
    d: int
    means: np.ndarray
    covs: np.ndarray
    priors: np.ndarray
    pooled_cov: np.ndarray
    actual_h_sigma: float
    actual_reff: float
    target_h_sigma: float
    target_reff: float
    mean_scale: float
    generator_family: str


def effective_rank(cov: np.ndarray) -> float:
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.clip(eigs, 0.0, None)
    op = float(eigs.max())
    if op <= 0:
        return 0.0
    return float(eigs.sum() / op)


def covariance_heterogeneity(covs: np.ndarray, priors: np.ndarray) -> tuple[float, np.ndarray]:
    pooled = np.einsum("c,cij->ij", priors, covs)
    denom = np.linalg.norm(pooled, ord="fro")
    if denom == 0:
        return 0.0, pooled
    diffs = covs - pooled[None, :, :]
    num2 = float(np.einsum("c,cij,cij->", priors, diffs, diffs))
    return float(np.sqrt(max(num2, 0.0)) / denom), pooled


def _tau_for_target_reff(d: int, target_reff: float) -> float:
    if not (1.0 < target_reff <= d):
        raise ValueError("target_reff must lie in (1, d]")
    lo, hi = 0.0, 10.0
    idx = np.arange(d, dtype=float)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        s = float(np.exp(-mid * idx).sum())
        if s > target_reff:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def base_eigenvalues(d: int, target_reff: float) -> np.ndarray:
    tau = _tau_for_target_reff(d, target_reff)
    vals = np.exp(-tau * np.arange(d, dtype=float))
    vals /= vals.max()
    return vals


def random_orthogonal(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d))
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def regular_simplex(C: int, d: int) -> np.ndarray:
    if C < 2 or d < C - 1:
        raise ValueError("Need d >= C-1 and C >= 2")
    centered = np.eye(C) - np.ones((C, C)) / C
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, : C - 1] * np.sqrt(C / (C - 1))
    out = np.zeros((C, d), dtype=float)
    out[:, : C - 1] = coords
    return out


def _shared_eigenvalue_covs(
    C: int,
    d: int,
    target_reff: float,
    target_h: float,
    geometry_seed: int,
    perturbation_seed: int,
) -> np.ndarray:
    q = random_orthogonal(d, geometry_seed)
    base = base_eigenvalues(d, target_reff)
    rng = np.random.default_rng(perturbation_seed)
    u = rng.normal(size=(C, d))
    u -= u.mean(axis=0, keepdims=True)
    u /= np.maximum(u.std(axis=0, keepdims=True), 1e-12)

    def build(eta: float) -> np.ndarray:
        covs = []
        for c in range(C):
            vals = base * np.exp(eta * u[c])
            vals *= base.sum() / vals.sum()
            cov = (q * vals) @ q.T
            covs.append(0.5 * (cov + cov.T))
        return np.stack(covs)

    if target_h <= 0:
        return build(0.0)
    priors = np.ones(C) / C
    lo, hi = 0.0, 0.25
    while covariance_heterogeneity(build(hi), priors)[0] < target_h and hi < 8.0:
        hi *= 2.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        h = covariance_heterogeneity(build(mid), priors)[0]
        if h < target_h:
            lo = mid
        else:
            hi = mid
    return build(0.5 * (lo + hi))


def _rotation_covs(
    C: int,
    d: int,
    target_reff: float,
    target_h: float,
    geometry_seed: int,
    perturbation_seed: int,
) -> np.ndarray:
    """Held-out covariance family with class-specific eigenvector structure.

    Each class receives a fixed randomly rotated version of the same base spectrum.
    A scalar eta mixes the common covariance with that rotated covariance. This keeps
    every matrix SPD, changes eigenvector structure rather than eigenvalues alone, and
    makes calibration cheap enough for locked holdout validation.
    """
    q = random_orthogonal(d, geometry_seed)
    base = base_eigenvalues(d, target_reff)
    base_cov = (q * base) @ q.T
    rotated = []
    for c in range(C):
        rc = random_orthogonal(d, perturbation_seed + 1009 * (c + 1))
        cov = rc @ base_cov @ rc.T
        rotated.append(0.5 * (cov + cov.T))
    rotated = np.stack(rotated)

    def build(eta: float) -> np.ndarray:
        covs = (1.0 - eta) * base_cov[None, :, :] + eta * rotated
        return 0.5 * (covs + np.transpose(covs, (0, 2, 1)))

    if target_h <= 0:
        return np.repeat(base_cov[None, :, :], C, axis=0)
    priors = np.ones(C) / C
    max_h = covariance_heterogeneity(build(1.0), priors)[0]
    if target_h > max_h + 1e-10:
        raise ValueError(
            f"Requested H_sigma={target_h} exceeds holdout generator maximum {max_h:.6f}"
        )
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        h = covariance_heterogeneity(build(mid), priors)[0]
        if h < target_h:
            lo = mid
        else:
            hi = mid
    return build(0.5 * (lo + hi))

def make_covariances(
    generator_family: str,
    C: int,
    d: int,
    target_reff: float,
    target_h: float,
    geometry_seed: int,
    perturbation_seed: int,
) -> np.ndarray:
    if generator_family == "shared_eigenvectors_class_specific_eigenvalues":
        return _shared_eigenvalue_covs(C, d, target_reff, target_h, geometry_seed, perturbation_seed)
    if generator_family == "class_specific_eigenvector_rotations":
        return _rotation_covs(C, d, target_reff, target_h, geometry_seed, perturbation_seed)
    raise ValueError(f"Unknown generator family: {generator_family}")


def sample_balanced(
    means: np.ndarray,
    covs: np.ndarray,
    n_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c, (mu, cov) in enumerate(zip(means, covs)):
        x = rng.multivariate_normal(mu, cov, size=n_per_class, check_valid="raise")
        xs.append(x)
        ys.append(np.full(n_per_class, c, dtype=np.int64))
    return np.vstack(xs), np.concatenate(ys)


def _sample_n_total_balanced(
    means: np.ndarray,
    covs: np.ndarray,
    n_total: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    C = len(means)
    per = max(1, n_total // C)
    return sample_balanced(means, covs, per, seed)


def calibrate_mean_scale(
    simplex: np.ndarray,
    pooled_cov: np.ndarray,
    covs_h0: np.ndarray,
    target_acc: float,
    seed: int,
    n_total: int,
    accuracy_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
) -> float:
    # Calibration data are dedicated generator-design samples, never reused for scientific evaluation.
    lo, hi = 0.0, 1.0
    priors = np.ones(simplex.shape[0]) / simplex.shape[0]
    while True:
        means = hi * simplex
        x, y = _sample_n_total_balanced(means, covs_h0, n_total, seed)
        acc = accuracy_fn(x, y, means, pooled_cov)
        if acc >= target_acc or hi > 128:
            break
        hi *= 2.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        means = mid * simplex
        x, y = _sample_n_total_balanced(means, covs_h0, n_total, seed)
        acc = accuracy_fn(x, y, means, pooled_cov)
        if acc < target_acc:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_geometry(
    generator_family: str,
    C: int,
    d: int,
    target_reff: float,
    target_h: float,
    geometry_seed: int,
    perturbation_seed: int,
    mean_scale: float,
) -> PopulationGeometry:
    priors = np.ones(C, dtype=float) / C
    covs = make_covariances(
        generator_family, C, d, target_reff, target_h, geometry_seed, perturbation_seed
    )
    actual_h, pooled = covariance_heterogeneity(covs, priors)
    means = mean_scale * regular_simplex(C, d)
    return PopulationGeometry(
        C=C,
        d=d,
        means=means,
        covs=covs,
        priors=priors,
        pooled_cov=pooled,
        actual_h_sigma=actual_h,
        actual_reff=effective_rank(pooled),
        target_h_sigma=target_h,
        target_reff=target_reff,
        mean_scale=mean_scale,
        generator_family=generator_family,
    )
