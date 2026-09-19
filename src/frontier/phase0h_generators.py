from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Phase0HGeometry:
    C: int
    d: int
    q: float
    target_reff: float
    actual_reff: float
    target_h_sigma: float
    actual_h_sigma: float
    base_eigenvalues: np.ndarray
    basis: np.ndarray
    sigma0: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    priors: np.ndarray
    pooled_cov: np.ndarray
    generator_family: str
    mean_scale: float


def stable_seed(base: int, *parts: Any) -> int:
    payload = "|".join([str(base), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False) & ((1 << 63) - 1)


def effective_rank(cov: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    eig = np.clip(eig, 0.0, None)
    op = float(eig.max()) if eig.size else 0.0
    return 0.0 if op <= 0 else float(eig.sum() / op)


def covariance_heterogeneity(covs: np.ndarray, priors: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    covs = np.asarray(covs, dtype=float)
    C = covs.shape[0]
    if priors is None:
        priors = np.ones(C, dtype=float) / C
    priors = np.asarray(priors, dtype=float)
    pooled = np.einsum("c,cij->ij", priors, covs)
    denom = float(np.linalg.norm(pooled, ord="fro"))
    if denom <= 0:
        return 0.0, pooled
    diffs = covs - pooled[None, :, :]
    num2 = float(np.einsum("c,cij,cij->", priors, diffs, diffs))
    return float(np.sqrt(max(num2, 0.0)) / denom), pooled


def random_orthogonal(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    a = rng.normal(size=(d, d))
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def regular_simplex(C: int, d: int) -> np.ndarray:
    if C < 2 or d < C - 1:
        raise ValueError("Need C>=2 and d>=C-1")
    centered = np.eye(C) - np.ones((C, C)) / C
    u, _, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, : C - 1] * np.sqrt(C / (C - 1))
    z = np.zeros((C, d), dtype=float)
    z[:, : C - 1] = coords
    return z


def exponential_spectrum(d: int, target_reff: float, tol: float = 1e-12) -> tuple[np.ndarray, float]:
    """Calibrate kappa in exp(-kappa*j/(d-1)) to a target effective rank.

    Maximum eigenvalue is exactly one, therefore effective rank is simply the sum.
    """
    target_reff = float(target_reff)
    if not (1.0 < target_reff <= d):
        raise ValueError(f"target_reff must lie in (1,d], got {target_reff} for d={d}")
    if abs(target_reff - d) <= tol:
        return np.ones(d, dtype=float), 0.0
    x = np.arange(d, dtype=float) / max(d - 1, 1)
    lo, hi = 0.0, 1.0
    def reff(k: float) -> float:
        return float(np.exp(-k * x).sum())
    while reff(hi) > target_reff:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Could not bracket exponential-spectrum effective rank")
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if reff(mid) > target_reff:
            lo = mid
        else:
            hi = mid
    kappa = 0.5 * (lo + hi)
    eig = np.exp(-kappa * x)
    eig /= eig.max()
    return eig, float(kappa)


def _base_geometry(d: int, q_ratio: float, geometry_seed: int, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    target_reff = float(q_ratio) * int(d)
    eig, kappa = exponential_spectrum(d, target_reff, tol=tol)
    Q = random_orthogonal(d, geometry_seed)
    sigma0 = (Q * eig) @ Q.T
    sigma0 = 0.5 * (sigma0 + sigma0.T)
    return eig, Q, sigma0, kappa


def _mean_preserving_eigenvalue_covs(
    C: int,
    eig: np.ndarray,
    Q: np.ndarray,
    sigma0: np.ndarray,
    target_h: float,
    perturbation_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(perturbation_seed))
    u = rng.normal(size=(C, len(eig)))
    u -= u.mean(axis=0, keepdims=True)
    col_sd = u.std(axis=0, keepdims=True)
    u /= np.maximum(col_sd, 1e-12)

    def build(eta: float) -> np.ndarray:
        raw = np.exp(float(eta) * u)
        multipliers = raw / raw.mean(axis=0, keepdims=True)
        vals = eig[None, :] * multipliers
        covs = np.einsum("ij,cj,kj->cik", Q, vals, Q, optimize=True)
        return 0.5 * (covs + np.transpose(covs, (0, 2, 1)))

    if target_h <= 0:
        return np.repeat(sigma0[None, :, :], C, axis=0)
    priors = np.ones(C) / C
    lo, hi = 0.0, 0.25
    while covariance_heterogeneity(build(hi), priors)[0] < target_h:
        hi *= 2.0
        if hi > 64:
            raise ValueError(f"Could not reach H_sigma={target_h} with eigenvalue generator")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if covariance_heterogeneity(build(mid), priors)[0] < target_h:
            lo = mid
        else:
            hi = mid
    return build(0.5 * (lo + hi))


def _noncommuting_A(C: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    mats = rng.normal(size=(C, d, d))
    mats = 0.5 * (mats + np.transpose(mats, (0, 2, 1)))
    mats -= mats.mean(axis=0, keepdims=True)
    # Remove the diagonal in the Sigma0 eigenbasis representation before rotation.
    # This guarantees genuinely noncommuting structure once Sigma0 is non-isotropic.
    idx = np.arange(d)
    mats[:, idx, idx] = 0.0
    mats -= mats.mean(axis=0, keepdims=True)
    max_op = max(float(np.max(np.abs(np.linalg.eigvalsh(m)))) for m in mats)
    if max_op <= 0:
        raise RuntimeError("Degenerate noncommuting perturbation")
    return mats / max_op


def _sym_matrix_exp(a: np.ndarray, scale: float = 1.0) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (a + a.T))
    return (V * np.exp(float(scale) * w)) @ V.T


def _spd_inv_sqrt(a: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (a + a.T))
    if float(w.min()) <= 0:
        raise np.linalg.LinAlgError("Matrix is not SPD")
    return (V * (1.0 / np.sqrt(w))) @ V.T


def _mean_preserving_noncommuting_covs(
    C: int,
    eig: np.ndarray,
    Q: np.ndarray,
    sigma0: np.ndarray,
    target_h: float,
    perturbation_seed: int,
) -> np.ndarray:
    """Mean-preserving, noncommuting SPD covariance family for Holdout B.

    The pre-freeze affine form Sigma0^(1/2)(I+eta A_c)Sigma0^(1/2) cannot
    reach H_sigma=0.8 for all frozen q values while preserving SPD. Before any
    formal Phase 0-H execution, the implementation therefore uses an SPD
    exponential perturbation followed by a deterministic pooled-covariance
    normalization. This preserves the intended scientific properties:
    class-specific noncommuting eigenvectors, exact pooled covariance Sigma0,
    continuous H_sigma calibration, and unconditional SPD.
    """
    if target_h <= 0:
        return np.repeat(sigma0[None, :, :], C, axis=0)

    # Generate symmetric class perturbations in the Sigma0 eigenbasis, center
    # them across classes, then rotate them to ambient coordinates. Retaining
    # both diagonal and off-diagonal components gives a broad noncommuting family.
    rng = np.random.default_rng(int(perturbation_seed))
    A_eig = rng.normal(size=(C, len(eig), len(eig)))
    A_eig = 0.5 * (A_eig + np.transpose(A_eig, (0, 2, 1)))
    A_eig -= A_eig.mean(axis=0, keepdims=True)
    max_op = max(float(np.max(np.abs(np.linalg.eigvalsh(a)))) for a in A_eig)
    if max_op <= 0:
        raise RuntimeError("Degenerate noncommuting perturbation")
    A_eig /= max_op
    A = np.einsum("ij,cjk,lk->cil", Q, A_eig, Q, optimize=True)
    A = 0.5 * (A + np.transpose(A, (0, 2, 1)))
    sqrt_sigma = (Q * np.sqrt(eig)) @ Q.T

    def build(eta: float) -> np.ndarray:
        raw = np.stack([
            sqrt_sigma @ _sym_matrix_exp(a, eta) @ sqrt_sigma
            for a in A
        ])
        raw = 0.5 * (raw + np.transpose(raw, (0, 2, 1)))
        raw_bar = raw.mean(axis=0)
        # Congruence T raw_bar T^T = Sigma0 exactly up to numerical precision.
        T = sqrt_sigma @ _spd_inv_sqrt(raw_bar)
        covs = np.stack([T @ r @ T.T for r in raw])
        return 0.5 * (covs + np.transpose(covs, (0, 2, 1)))

    priors = np.ones(C) / C
    lo, hi = 0.0, 0.5
    while covariance_heterogeneity(build(hi), priors)[0] < target_h:
        hi *= 2.0
        if hi > 64:
            raise ValueError(f"Could not reach H_sigma={target_h} in noncommuting generator")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if covariance_heterogeneity(build(mid), priors)[0] < target_h:
            lo = mid
        else:
            hi = mid
    return build(0.5 * (lo + hi))

def build_phase0h_geometry(
    *,
    generator_family: str,
    C: int,
    d: int,
    q_ratio: float,
    target_h: float,
    seed: int,
    mean_scale: float,
    reff_tolerance: float = 1e-6,
) -> Phase0HGeometry:
    geometry_seed = stable_seed(10_000_001, generator_family, d, q_ratio, seed, "basis")
    perturbation_seed = stable_seed(10_000_003, generator_family, d, q_ratio, seed, "perturbation")
    eig, Q, sigma0, _ = _base_geometry(d, q_ratio, geometry_seed)
    target_reff = float(q_ratio) * d
    actual_reff = effective_rank(sigma0)
    if abs(actual_reff - target_reff) > reff_tolerance:
        raise AssertionError(f"Effective-rank calibration failed: target={target_reff}, actual={actual_reff}")

    if generator_family == "mean_preserving_shared_eigenvector_eigenvalue_heterogeneity":
        covs = _mean_preserving_eigenvalue_covs(C, eig, Q, sigma0, target_h, perturbation_seed)
    elif generator_family == "mean_preserving_noncommuting_covariance_perturbation":
        covs = _mean_preserving_noncommuting_covs(C, eig, Q, sigma0, target_h, perturbation_seed)
    else:
        raise ValueError(f"Unknown Phase 0-H generator: {generator_family}")

    priors = np.ones(C, dtype=float) / C
    actual_h, pooled = covariance_heterogeneity(covs, priors)
    z = regular_simplex(C, d)
    sqrt_sigma = (Q * np.sqrt(eig)) @ Q.T
    means = float(mean_scale) * (z @ sqrt_sigma.T)

    return Phase0HGeometry(
        C=C,
        d=d,
        q=float(q_ratio),
        target_reff=target_reff,
        actual_reff=float(actual_reff),
        target_h_sigma=float(target_h),
        actual_h_sigma=float(actual_h),
        base_eigenvalues=eig,
        basis=Q,
        sigma0=sigma0,
        means=means,
        covs=covs,
        priors=priors,
        pooled_cov=pooled,
        generator_family=generator_family,
        mean_scale=float(mean_scale),
    )


def sample_class_pools(
    geom: Phase0HGeometry,
    n_per_class: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    pools = []
    for c in range(geom.C):
        pools.append(rng.multivariate_normal(
            geom.means[c], geom.covs[c], size=int(n_per_class), check_valid="raise"
        ))
    return pools


def prefixes_to_xy(pools: list[np.ndarray], n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for c, pool in enumerate(pools):
        n = int(n_per_class)
        if n > len(pool):
            raise ValueError("Requested prefix exceeds master pool")
        xs.append(pool[:n])
        ys.append(np.full(n, c, dtype=np.int64))
    return np.vstack(xs), np.concatenate(ys)


def sample_balanced_test(geom: Phase0HGeometry, n_total: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    per = max(1, int(n_total) // geom.C)
    pools = sample_class_pools(geom, per, seed)
    return prefixes_to_xy(pools, per)
