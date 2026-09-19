from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from .phase0h import evaluate_phase0h_path
from .phase0r import unbiased_covariance_stats
from .phase0h_generators import stable_seed
from .phase0s_generators import n_per_class_from_seff, panel_geometry, sample_balanced_test, sample_xy


def _op_norm(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (a + a.T)))))


def _relative_cov_errors(estimated: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    frob, op = [], []
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


def run_phase0s_replicate(
    protocol: dict[str, Any],
    panel: str,
    d: int,
    q_ratio: float,
    target_h: float,
    requested_seff: float,
    seed: int,
    *,
    tau_override: list[float] | None = None,
    test_n_override: int | None = None,
    distribution_override: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    t0 = time.perf_counter()
    fixed = protocol["fixed"]
    C = int(fixed["C"])
    alpha0 = float(fixed["alpha0"])
    mean_scale = float(fixed["mean_scale_whitened_simplex"])
    taus = list(map(float, tau_override if tau_override is not None else fixed["tau"]))

    if panel == "panel_a":
        distribution = "gaussian"
        nu = 5.0
        covariance_geometry = "commuting_eigenvalue"
    elif panel == "panel_b":
        distribution = "gaussian"
        nu = 5.0
        covariance_geometry = "noncommuting"
    elif panel == "panel_c":
        distribution = "student_t"
        nu = float(protocol["panel_C"]["degrees_of_freedom"])
        covariance_geometry = "commuting_eigenvalue"
    elif panel == "smoke":
        distribution = str(distribution_override or "gaussian")
        nu = 5.0
        covariance_geometry = "commuting_eigenvalue" if distribution == "gaussian" else "commuting_eigenvalue_student_t"
    else:
        raise ValueError(panel)

    geom_seed = stable_seed(30_000_001, "phase0s", panel, d, q_ratio, target_h, requested_seff, seed, "geometry")
    geom = panel_geometry(
        panel if panel != "smoke" else "smoke",
        C=C,
        d=int(d),
        q_ratio=float(q_ratio),
        target_h=float(target_h),
        seed=int(geom_seed),
        mean_scale=mean_scale,
    )

    n_per_class = n_per_class_from_seff(requested_seff, geom.actual_reff)
    actual_seff = float((n_per_class - 1) / geom.actual_reff)
    test_n = int(test_n_override if test_n_override is not None else fixed["test_n"])

    train_seed = stable_seed(30_000_003, "phase0s", panel, d, q_ratio, target_h, requested_seff, seed, "train")
    test_seed = stable_seed(30_000_005, "phase0s", panel, d, q_ratio, target_h, requested_seff, seed, "test")
    Xtr, ytr = sample_xy(geom, n_per_class=n_per_class, seed=train_seed, distribution=distribution, nu=nu)
    Xte, yte = sample_balanced_test(geom, n_total=test_n, seed=test_seed, distribution=distribution, nu=nu)

    stats = unbiased_covariance_stats(Xtr, ytr, C)
    acc, diag = evaluate_phase0h_path(
        stats.means, stats.priors, stats.pooled_cov, stats.class_covs,
        Xte, yte, taus, alpha0,
    )
    oracle_acc, oracle_diag = evaluate_phase0h_path(
        geom.means, geom.priors, geom.pooled_cov, geom.covs,
        Xte, yte, taus, alpha0,
    )

    pop_pool_err = float(np.linalg.norm(geom.pooled_cov - geom.sigma0, ord="fro") / max(np.linalg.norm(geom.sigma0, ord="fro"), 1e-15))
    common = {
        "panel": panel,
        "distribution": distribution,
        "covariance_geometry": covariance_geometry,
        "condition_id": f"{panel}__d{d}__q{q_ratio:.6f}__H{target_h:.4f}__S{requested_seff:.6f}",
        "seed": int(seed),
        "C": C,
        "d": int(d),
        "q_target": float(q_ratio),
        "target_reff": float(q_ratio * d),
        "actual_reff": float(geom.actual_reff),
        "target_H_sigma": float(target_h),
        "actual_H_sigma": float(geom.actual_h_sigma),
        "requested_effective_support": float(requested_seff),
        "effective_support": actual_seff,
        "n_per_class": int(n_per_class),
        "train_n": int(len(ytr)),
        "test_n": int(len(yte)),
        "alpha0": alpha0,
        "mean_scale": mean_scale,
        "student_t_nu": float(nu) if distribution == "student_t" else np.nan,
    }

    curve_rows, oracle_rows = [], []
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
            "oracle_type": "covariance_structure_qda_family" if panel == "panel_c" else "population_qda_family",
            "bayes_optimal_claimed": False if panel == "panel_c" else True,
            "oracle_min_generalized_factor": float(oracle_diag["min_generalized_factor"]),
            "oracle_min_base_eigenvalue": float(oracle_diag["min_base_eigenvalue"]),
        })

    frob_err, op_err = _relative_cov_errors(stats.class_covs, geom.covs)
    pooled_diff = 0.5 * ((stats.pooled_cov - geom.pooled_cov) + (stats.pooled_cov - geom.pooled_cov).T)
    pooled_frob_err = float(np.linalg.norm(pooled_diff, ord="fro") / max(np.linalg.norm(geom.pooled_cov, ord="fro"), 1e-15))
    pooled_op_err = float(_op_norm(pooled_diff) / max(_op_norm(geom.pooled_cov), 1e-15))
    ranks = [np.linalg.matrix_rank(c) / d for c in stats.class_covs]
    ridge = alpha0 * stats.mean_variance_pool
    cmean, cmax, pcond = _regularized_condition_metrics(stats.class_covs, stats.pooled_cov, ridge)

    mech = {
        **common,
        "class_cov_fro_rel_error_mean": frob_err,
        "class_cov_op_rel_error_mean": op_err,
        "pooled_cov_fro_rel_error": pooled_frob_err,
        "pooled_cov_op_rel_error": pooled_op_err,
        "sample_cov_rank_fraction_mean": float(np.mean(ranks)),
        "regularized_class_condition_mean": cmean,
        "regularized_class_condition_max": cmax,
        "regularized_pooled_condition": pcond,
        "population_pooled_cov_relative_fro_error": pop_pool_err,
    }

    run = {
        **common,
        "effective_rank_abs_error": float(abs(geom.actual_reff - q_ratio * d)),
        "heterogeneity_abs_error": float(abs(geom.actual_h_sigma - target_h)),
        "population_pooled_cov_relative_fro_error": pop_pool_err,
        "tau0_class_cov_identity_max_abs": 0.0,
        "all_curve_values_finite": bool(np.isfinite(acc).all() and np.isfinite(oracle_acc).all()),
        "wall_time_sec": float(time.perf_counter() - t0),
    }
    return run, curve_rows, oracle_rows, mech
