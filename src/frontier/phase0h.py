from __future__ import annotations

import math
import time
from typing import Any, Iterable

import numpy as np

from .phase0h_generators import (
    Phase0HGeometry,
    build_phase0h_geometry,
    prefixes_to_xy,
    sample_balanced_test,
    sample_class_pools,
    stable_seed,
)
from .phase0r import _evaluate_covariance_path, unbiased_covariance_stats


def _section(protocol: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode in {"discovery", "smoke"}:
        return dict(protocol["discovery"])
    if mode == "holdout_a":
        return dict(protocol["holdout_A"])
    if mode == "holdout_b":
        return dict(protocol["holdout_B"])
    raise ValueError(mode)


def _generator_for_mode(section: dict[str, Any]) -> str:
    return str(section["generator"])


def n_per_class_from_rho(rho: float, d: int) -> int:
    return max(3, int(math.ceil(float(rho) * int(d))) + 1)


def _condition_id(generator: str, d: int, q: float, h: float, rho: float) -> str:
    return f"{generator}__d{d}__q{q:.6f}__H{h:.4f}__rho{rho:.4f}"


def _population_id(generator: str, d: int, q: float, h: float, seed: int) -> str:
    return f"{generator}__d{d}__q{q:.6f}__H{h:.4f}__seed{seed}"


def _op_norm(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (a + a.T)))))


def _relative_cov_errors(estimated: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    frob = []
    op = []
    for e, t in zip(estimated, truth):
        diff = 0.5 * ((e - t) + (e - t).T)
        frob.append(float(np.linalg.norm(diff, ord="fro") / max(np.linalg.norm(t, ord="fro"), 1e-15)))
        op.append(float(_op_norm(diff) / max(_op_norm(t), 1e-15)))
    return float(np.mean(frob)), float(np.mean(op))


def _regularized_condition_metrics(class_covs: np.ndarray, pooled: np.ndarray, ridge: float) -> tuple[float, float, float]:
    d = pooled.shape[0]
    I = np.eye(d)
    conds = []
    for cov in class_covs:
        eig = np.linalg.eigvalsh(0.5 * (cov + cov.T) + ridge * I)
        conds.append(float(eig.max() / eig.min()))
    peig = np.linalg.eigvalsh(0.5 * (pooled + pooled.T) + ridge * I)
    return float(np.mean(conds)), float(np.max(conds)), float(peig.max() / peig.min())


def evaluate_phase0h_path(
    means: np.ndarray,
    priors: np.ndarray,
    pooled_cov: np.ndarray,
    class_covs: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    taus: Iterable[float],
    alpha0: float,
) -> tuple[np.ndarray, dict[str, float]]:
    vbar = float(np.trace(pooled_cov) / pooled_cov.shape[0])
    if not np.isfinite(vbar) or vbar <= 0:
        raise ValueError("Non-positive mean variance")
    ridge = float(alpha0) * vbar
    return _evaluate_covariance_path(
        means, priors, pooled_cov, class_covs, X, y, taus, ridge=ridge
    )


def run_phase0h_population_replicate(
    protocol: dict[str, Any],
    mode: str,
    d: int,
    q_ratio: float,
    target_h: float,
    seed: int,
    *,
    rho_override: list[float] | None = None,
    tau_override: list[float] | None = None,
    test_n_override: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one independent population replicate across the full nested rho path.

    Returns:
      run_rows: one row per rho
      curve_rows: finite-sample estimated accuracy curves, one row per rho/tau
      oracle_rows: population-oracle curves, one row per rho/tau (same oracle curve repeated across rho)
      mechanism_rows: one row per rho
    """
    t0 = time.perf_counter()
    section = _section(protocol, mode)
    fixed = protocol["fixed"]
    C = int(fixed["C"])
    alpha0 = float(fixed["alpha0"])
    mean_scale = float(fixed["mean_scale_whitened_simplex"])
    taus = list(map(float, tau_override if tau_override is not None else fixed["tau"]))
    rhos = list(map(float, rho_override if rho_override is not None else section["rho_nminus1_over_d"]))
    generator = _generator_for_mode(section)
    reff_tol = float(protocol["spectrum"]["effective_rank_tolerance"])

    geom = build_phase0h_geometry(
        generator_family=generator,
        C=C,
        d=int(d),
        q_ratio=float(q_ratio),
        target_h=float(target_h),
        seed=int(seed),
        mean_scale=mean_scale,
        reff_tolerance=reff_tol,
    )

    test_n = int(test_n_override if test_n_override is not None else section.get("test_n", fixed["test_n"]))
    test_seed = stable_seed(20_000_001, mode, d, q_ratio, target_h, seed, "test")
    Xte, yte = sample_balanced_test(geom, test_n, test_seed)

    rho_max = max(rhos)
    max_n = n_per_class_from_rho(rho_max, d)
    train_seed = stable_seed(20_000_003, mode, d, q_ratio, target_h, seed, "train")
    master = sample_class_pools(geom, max_n, train_seed)

    oracle_acc, oracle_diag = evaluate_phase0h_path(
        geom.means, geom.priors, geom.pooled_cov, geom.covs, Xte, yte, taus, alpha0
    )

    run_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    pop_id = _population_id(generator, d, q_ratio, target_h, seed)
    pooled_rel = float(np.linalg.norm(geom.pooled_cov - geom.sigma0, ord="fro") / max(np.linalg.norm(geom.sigma0, ord="fro"), 1e-15))

    for rho in rhos:
        rt0 = time.perf_counter()
        n_per_class = n_per_class_from_rho(rho, d)
        Xtr, ytr = prefixes_to_xy(master, n_per_class)
        stats = unbiased_covariance_stats(Xtr, ytr, C)
        acc, diag = evaluate_phase0h_path(
            stats.means, stats.priors, stats.pooled_cov, stats.class_covs,
            Xte, yte, taus, alpha0,
        )
        actual_rho = float((n_per_class - 1) / d)
        actual_seff = float((n_per_class - 1) / geom.actual_reff)
        cond_id = _condition_id(generator, d, q_ratio, target_h, rho)
        common = {
            "run_stage": mode,
            "generator_family": generator,
            "population_id": pop_id,
            "condition_id": cond_id,
            "seed": int(seed),
            "C": C,
            "d": int(d),
            "q_target": float(q_ratio),
            "target_reff": float(q_ratio * d),
            "actual_reff": float(geom.actual_reff),
            "target_H_sigma": float(target_h),
            "actual_H_sigma": float(geom.actual_h_sigma),
            "requested_rho": float(rho),
            "actual_rho": actual_rho,
            "effective_support": actual_seff,
            "n_per_class": int(n_per_class),
            "train_n": int(len(ytr)),
            "test_n": int(len(yte)),
            "alpha0": alpha0,
            "mean_scale": mean_scale,
        }
        for tau, a, oa in zip(taus, acc, oracle_acc):
            curve_rows.append({
                **common,
                "tau": float(tau),
                "accuracy": float(a),
                "min_generalized_factor": float(diag["min_generalized_factor"]),
                "min_base_eigenvalue": float(diag["min_base_eigenvalue"]),
            })
            oracle_rows.append({
                **common,
                "tau": float(tau),
                "oracle_accuracy": float(oa),
                "oracle_min_generalized_factor": float(oracle_diag["min_generalized_factor"]),
                "oracle_min_base_eigenvalue": float(oracle_diag["min_base_eigenvalue"]),
            })

        frob_err, op_err = _relative_cov_errors(stats.class_covs, geom.covs)
        pooled_diff = 0.5 * ((stats.pooled_cov - geom.pooled_cov) + (stats.pooled_cov - geom.pooled_cov).T)
        pooled_frob_err = float(np.linalg.norm(pooled_diff, ord="fro") / max(np.linalg.norm(geom.pooled_cov, ord="fro"), 1e-15))
        pooled_op_err = float(_op_norm(pooled_diff) / max(_op_norm(geom.pooled_cov), 1e-15))
        ranks = [np.linalg.matrix_rank(c, tol=max(np.linalg.norm(c, 2), 1.0) * np.finfo(float).eps * d * 10) / d for c in stats.class_covs]
        ridge = alpha0 * stats.mean_variance_pool
        cmean, cmax, pcond = _regularized_condition_metrics(stats.class_covs, stats.pooled_cov, ridge)

        mechanism_rows.append({
            **common,
            "class_cov_fro_rel_error_mean": frob_err,
            "class_cov_op_rel_error_mean": op_err,
            "pooled_cov_fro_rel_error": pooled_frob_err,
            "pooled_cov_op_rel_error": pooled_op_err,
            "sample_cov_rank_fraction_mean": float(np.mean(ranks)),
            "regularized_class_condition_mean": cmean,
            "regularized_class_condition_max": cmax,
            "regularized_pooled_condition": pcond,
            "population_pooled_cov_relative_fro_error": pooled_rel,
        })
        run_rows.append({
            **common,
            "effective_rank_abs_error": float(abs(geom.actual_reff - q_ratio * d)),
            "heterogeneity_abs_error": float(abs(geom.actual_h_sigma - target_h)),
            "population_pooled_cov_relative_fro_error": pooled_rel,
            "tau0_class_cov_identity_max_abs": 0.0,
            "all_curve_values_finite": bool(np.isfinite(acc).all() and np.isfinite(oracle_acc).all()),
            "rho_wall_time_sec": float(time.perf_counter() - rt0),
        })

    total = float(time.perf_counter() - t0)
    for r in run_rows:
        r["population_wall_time_sec"] = total
    return run_rows, curve_rows, oracle_rows, mechanism_rows
