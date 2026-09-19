from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf

from .classifiers import _floor_cov, predict_qda, QDAHead
from .synthetic import build_geometry, sample_balanced


@dataclass(frozen=True)
class EmpiricalCovarianceStats:
    means: np.ndarray
    priors: np.ndarray
    pooled_cov: np.ndarray
    class_covs: np.ndarray
    n_per_class: np.ndarray
    mean_variance_pool: float


def _stable_seed(base: int, *parts: Any) -> int:
    payload = "|".join([str(base), *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False) & ((1 << 63) - 1)


def unbiased_covariance_stats(X: np.ndarray, y: np.ndarray, C: int) -> EmpiricalCovarianceStats:
    """Unbiased within-class covariance estimates used by the fixed-ridge family.

    Class covariance uses denominator n_c-1. The pooled within-class covariance uses
    denominator N-C. With balanced classes this makes the pooled covariance the
    arithmetic mean of class-wise unbiased covariance estimates.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    means = []
    class_covs = []
    counts = []
    scatter = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    for c in range(C):
        xc = X[y == c]
        if len(xc) < 2:
            raise ValueError(f"Class {c} has fewer than 2 samples")
        mu = xc.mean(axis=0)
        res = xc - mu
        sc = res.T @ res
        cov = sc / (len(xc) - 1)
        means.append(mu)
        class_covs.append(0.5 * (cov + cov.T))
        counts.append(len(xc))
        scatter += sc
    n_total = len(X)
    if n_total <= C:
        raise ValueError("Need N > C for pooled within-class covariance")
    pooled = scatter / (n_total - C)
    pooled = 0.5 * (pooled + pooled.T)
    priors = np.asarray(counts, dtype=float) / n_total
    vbar = float(np.trace(pooled) / pooled.shape[0])
    if not np.isfinite(vbar) or vbar <= 0:
        raise ValueError("Non-positive pooled mean variance")
    return EmpiricalCovarianceStats(
        means=np.stack(means),
        priors=priors,
        pooled_cov=pooled,
        class_covs=np.stack(class_covs),
        n_per_class=np.asarray(counts, dtype=int),
        mean_variance_pool=vbar,
    )


def _build_qda_head(means: np.ndarray, covs: np.ndarray, priors: np.ndarray) -> QDAHead:
    clean, chols, logdets = [], [], []
    for cov in covs:
        c = 0.5 * (cov + cov.T)
        chol = np.linalg.cholesky(c)
        clean.append(c)
        chols.append(chol)
        logdets.append(2.0 * np.log(np.diag(chol)).sum())
    return QDAHead(
        means=np.asarray(means),
        covs=np.stack(clean),
        priors=np.asarray(priors),
        chols=chols,
        logdets=np.asarray(logdets),
        shrinkages=None,
    )


def explicit_fixed_ridge_covs(stats: EmpiricalCovarianceStats, tau: float, alpha: float) -> np.ndarray:
    I = np.eye(stats.pooled_cov.shape[0])
    ridge = float(alpha) * stats.mean_variance_pool * I
    return (
        (1.0 - float(tau)) * stats.pooled_cov[None, :, :]
        + float(tau) * stats.class_covs
        + ridge[None, :, :]
    )


def explicit_fixed_ridge_accuracy(
    stats: EmpiricalCovarianceStats,
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    alpha: float,
) -> float:
    covs = explicit_fixed_ridge_covs(stats, tau, alpha)
    head = _build_qda_head(stats.means, covs, stats.priors)
    return float((predict_qda(head, X) == y).mean())


def fit_ledoit_wolf_endpoints(X: np.ndarray, y: np.ndarray, C: int, rel_floor: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.stack([X[y == c].mean(axis=0) for c in range(C)])
    priors = np.asarray([(y == c).mean() for c in range(C)], dtype=float)
    residuals = np.vstack([X[y == c] - means[c] for c in range(C)])
    lw_pool = LedoitWolf(assume_centered=True).fit(residuals)
    pooled = _floor_cov(lw_pool.covariance_, rel_floor)
    class_covs = []
    shrinkages = []
    for c in range(C):
        res = X[y == c] - means[c]
        lw = LedoitWolf(assume_centered=True).fit(res)
        class_covs.append(_floor_cov(lw.covariance_, rel_floor))
        shrinkages.append(float(lw.shrinkage_))
    return means, priors, pooled, np.stack(class_covs), np.asarray([float(lw_pool.shrinkage_), *shrinkages])


def _evaluate_covariance_path(
    means: np.ndarray,
    priors: np.ndarray,
    pooled_cov: np.ndarray,
    class_covs: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    taus: Iterable[float],
    ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Efficiently evaluate a covariance interpolation path.

    For each class, Sigma(tau)=A+tau*D where
      A = pooled_cov + ridge*I
      D = class_cov - pooled_cov.
    A generalized eigendecomposition D v = lambda A v gives
      Sigma(tau)^-1 = V diag(1/(1+tau*lambda)) V^T,
    and logdet(Sigma(tau)) = logdet(A)+sum log(1+tau*lambda).
    This avoids rebuilding/inverting a full covariance matrix for each tau.
    """
    taus = np.asarray(list(taus), dtype=float)
    n_tau = len(taus)
    n = len(X)
    C = len(means)
    d = X.shape[1]
    I = np.eye(d)
    scores = np.empty((n, n_tau, C), dtype=float)
    min_factor = float("inf")
    min_a_eig = float("inf")
    for c in range(C):
        A = 0.5 * (pooled_cov + pooled_cov.T) + float(ridge) * I
        D = 0.5 * ((class_covs[c] - pooled_cov) + (class_covs[c] - pooled_cov).T)
        a_eigs = np.linalg.eigvalsh(A)
        min_a_eig = min(min_a_eig, float(a_eigs.min()))
        if a_eigs.min() <= 0:
            raise np.linalg.LinAlgError("Base covariance A is not SPD")
        lam, V = eigh(D, A, check_finite=False)
        factors = 1.0 + np.outer(lam, taus)  # d x tau
        mf = float(factors.min())
        min_factor = min(min_factor, mf)
        if mf <= 0:
            raise np.linalg.LinAlgError(f"Interpolated covariance lost SPD: min factor={mf}")
        sign, logdetA = np.linalg.slogdet(A)
        if sign <= 0:
            raise np.linalg.LinAlgError("Base covariance has non-positive determinant")
        centered = X - means[c]
        Z = centered @ V
        quad = (Z * Z) @ (1.0 / factors)
        logdets = float(logdetA) + np.log(factors).sum(axis=0)
        scores[:, :, c] = -0.5 * (quad + logdets[None, :]) + math.log(float(priors[c]))
    pred = scores.argmax(axis=2)
    acc = (pred == y[:, None]).mean(axis=0).astype(float)
    return acc, {
        "min_generalized_factor": float(min_factor),
        "min_base_eigenvalue": float(min_a_eig),
    }


def evaluate_fixed_ridge_grid(
    stats: EmpiricalCovarianceStats,
    X: np.ndarray,
    y: np.ndarray,
    taus: Iterable[float],
    alphas: Iterable[float],
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for alpha in alphas:
        ridge = float(alpha) * stats.mean_variance_pool
        acc, diag = _evaluate_covariance_path(
            stats.means, stats.priors, stats.pooled_cov, stats.class_covs,
            X, y, taus, ridge,
        )
        for tau, a in zip(taus, acc):
            out.append({
                "estimator_family": "fixed_ridge",
                "alpha": float(alpha),
                "tau": float(tau),
                "accuracy": float(a),
                **diag,
            })
    return out


def evaluate_adaptive_lw_grid(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    C: int,
    taus: Iterable[float],
    rel_floor: float = 1e-8,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    means, priors, pooled, class_covs, shrink = fit_ledoit_wolf_endpoints(Xtr, ytr, C, rel_floor)
    acc, diag = _evaluate_covariance_path(means, priors, pooled, class_covs, Xte, yte, taus, ridge=0.0)
    rows = [{
        "estimator_family": "adaptive_lw",
        "alpha": np.nan,
        "tau": float(t),
        "accuracy": float(a),
        **diag,
    } for t, a in zip(taus, acc)]
    info = {
        "lw_pooled_shrinkage": float(shrink[0]),
        "lw_class_shrinkage_mean": float(shrink[1:].mean()),
        "lw_class_shrinkage_min": float(shrink[1:].min()),
        "lw_class_shrinkage_max": float(shrink[1:].max()),
    }
    return rows, info


def _section_for_mode(protocol: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode in {"discovery", "smoke"}:
        return dict(protocol["discovery"])
    if mode == "holdout_a":
        return dict(protocol["holdout_A"])
    if mode == "holdout_b":
        return dict(protocol["holdout_B"])
    raise ValueError(mode)


def run_phase0r_replicate(
    protocol: dict[str, Any],
    mode: str,
    target_h: float,
    requested_support: float,
    seed: int,
    *,
    test_n_override: int | None = None,
    tau_override: list[float] | None = None,
    alpha_override: list[float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t0 = time.perf_counter()
    section = _section_for_mode(protocol, mode)
    C = int(section["C"])
    d = int(section["d"])
    target_reff = float(section["target_effective_rank"])
    mean_scale = float(protocol["discovery"]["mean_scale"])
    generator = str(section["generator"])
    generator_impl = "class_specific_eigenvector_rotations" if generator == "class_specific_eigenvector_rotation" else generator
    geometry_seed = int(section.get("geometry_seed", 314159 if mode != "holdout_b" else 314160))
    perturbation_seed = int(section.get("perturbation_seed", 161803 if mode != "holdout_b" else 161804))
    geom = build_geometry(generator_impl, C, d, target_reff, float(target_h), geometry_seed, perturbation_seed, mean_scale)

    n_per_class = max(3, int(math.ceil(float(requested_support) * geom.actual_reff)) + 1)
    Xtr, ytr = sample_balanced(geom.means, geom.covs, n_per_class, int(seed))
    test_n = int(test_n_override if test_n_override is not None else section["test_n"])
    per_test = max(1, test_n // C)
    base = int(section.get("test_seed_base", 9_200_000 if mode == "discovery" else 9_300_000))
    test_seed = _stable_seed(base, mode, d, generator, f"{target_h:.8f}", f"{requested_support:.8f}", int(seed))
    Xte, yte = sample_balanced(geom.means, geom.covs, per_test, test_seed)

    stats = unbiased_covariance_stats(Xtr, ytr, C)
    taus = tau_override if tau_override is not None else list(map(float, section["tau"]))
    alphas = alpha_override if alpha_override is not None else list(map(float, section["alpha"]))
    fixed = evaluate_fixed_ridge_grid(stats, Xte, yte, taus, alphas)
    rel_floor = float(protocol.get("numerical", {}).get("lw_eigenvalue_floor_relative", 1e-8))
    lw_rows, lw_info = evaluate_adaptive_lw_grid(Xtr, ytr, Xte, yte, C, taus, rel_floor)

    condition_id = f"{generator}__d{d}__H{float(target_h):.4f}__S{float(requested_support):.4f}"
    common = {
        "run_stage": mode,
        "generator_family": generator,
        "condition_id": condition_id,
        "seed": int(seed),
        "C": C,
        "d": d,
        "target_reff": target_reff,
        "actual_reff": float(geom.actual_reff),
        "target_H_sigma": float(target_h),
        "actual_H_sigma": float(geom.actual_h_sigma),
        "requested_effective_support": float(requested_support),
        "effective_support": float((n_per_class - 1) / geom.actual_reff),
        "n_per_class": int(n_per_class),
        "train_n": int(len(ytr)),
        "test_n": int(len(yte)),
        "mean_scale": mean_scale,
    }
    curves = []
    for row in [*fixed, *lw_rows]:
        curves.append({**common, **row})

    tau0_covs = explicit_fixed_ridge_covs(stats, 0.0, float(alphas[0]))
    tau1_covs = explicit_fixed_ridge_covs(stats, 1.0, float(alphas[0]))
    tau0_identity = float(np.max(np.abs(tau0_covs - tau0_covs[0])))
    min_eig_tau1 = float(min(np.linalg.eigvalsh(c).min() for c in tau1_covs))
    run_row = {
        **common,
        "heterogeneity_abs_error": float(abs(geom.actual_h_sigma - float(target_h))),
        "tau0_class_cov_identity_max_abs": tau0_identity,
        "tau1_min_eigenvalue_alpha_min": min_eig_tau1,
        "all_curve_values_finite": bool(all(np.isfinite(r["accuracy"]) for r in curves)),
        **lw_info,
        "wall_time_sec": float(time.perf_counter() - t0),
    }
    return run_row, curves
