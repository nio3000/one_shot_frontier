from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np

from .classifiers import (
    accuracy_lda,
    build_population_lda,
    build_population_qda,
    condition_number,
    fit_o4_ledoit_wolf,
    fit_o5_ledoit_wolf,
    predict_lda,
    predict_qda,
)
from .synthetic import (
    build_geometry,
    calibrate_mean_scale,
    make_covariances,
    covariance_heterogeneity,
    regular_simplex,
    sample_balanced,
)
from .utils import condition_id


def compute_mean_scale(section: dict[str, Any]) -> float:
    C, d = int(section["C"]), int(section["d"])
    covs_h0 = make_covariances(
        section["generator_family"], C, d, float(section["target_reff"]), 0.0,
        int(section["geometry_seed"]), int(section["perturbation_seed"]),
    )
    priors = np.ones(C) / C
    _, pooled = covariance_heterogeneity(covs_h0, priors)
    simplex = regular_simplex(C, d)
    return calibrate_mean_scale(
        simplex,
        pooled,
        covs_h0,
        float(section["target_population_o4_accuracy_at_H0"]),
        int(section["calibration_seed"]),
        int(section["calibration_n_total"]),
        accuracy_lda,
    )


def run_one(
    section: dict[str, Any],
    target_h: float,
    effective_support: float,
    train_seed: int,
    mean_scale: float,
    rel_floor: float,
    test_n_override: int | None = None,
) -> dict[str, Any]:
    geom = build_geometry(
        section["generator_family"], int(section["C"]), int(section["d"]),
        float(section["target_reff"]), float(target_h), int(section["geometry_seed"]),
        int(section["perturbation_seed"]), mean_scale,
    )
    n_per_class = max(2, int(math.ceil(float(effective_support) * geom.actual_reff)) + 1)
    Xtr, ytr = sample_balanced(geom.means, geom.covs, n_per_class, train_seed)

    test_n = int(test_n_override or section["test_n_total"])
    per_test = max(1, test_n // geom.C)
    test_seed = int(section["test_seed_base"]) + int(round(target_h * 10000)) * 100 + int(round(effective_support * 10))
    Xte, yte = sample_balanced(geom.means, geom.covs, per_test, test_seed)

    o4_oracle = build_population_lda(geom.means, geom.pooled_cov, geom.priors)
    o5_oracle = build_population_qda(geom.means, geom.covs, geom.priors)
    o4 = fit_o4_ledoit_wolf(Xtr, ytr, geom.C, rel_floor)
    o5 = fit_o5_ledoit_wolf(Xtr, ytr, geom.C, rel_floor)

    oracle_o4_acc = float((predict_lda(o4_oracle, Xte) == yte).mean())
    oracle_o5_acc = float((predict_qda(o5_oracle, Xte) == yte).mean())
    estimated_o4_acc = float((predict_lda(o4, Xte) == yte).mean())
    estimated_o5_acc = float((predict_qda(o5, Xte) == yte).mean())

    expressivity_gain = oracle_o5_acc - oracle_o4_acc
    o4_est_error = oracle_o4_acc - estimated_o4_acc
    o5_est_error = oracle_o5_acc - estimated_o5_acc
    extra_o5_burden = o5_est_error - o4_est_error
    delta = estimated_o5_acc - estimated_o4_acc
    decomposition_residual = delta - (expressivity_gain - extra_o5_burden)

    # Exact object-recovery sanity: B == sum_c S_c from finite training sample.
    B = Xtr.T @ Xtr
    S_sum = np.zeros_like(B)
    for c in range(geom.C):
        xc = Xtr[ytr == c]
        S_sum += xc.T @ xc
    object_recovery_max_abs = float(np.max(np.abs(B - S_sum)))

    o5_conds = np.asarray([condition_number(c) for c in o5.covs])

    return {
        "generator_family": geom.generator_family,
        "condition_id": condition_id(geom.generator_family, geom.d, target_h, effective_support),
        "seed": int(train_seed),
        "C": geom.C,
        "d": geom.d,
        "target_reff": geom.target_reff,
        "actual_reff": geom.actual_reff,
        "target_H_sigma": float(target_h),
        "actual_H_sigma": geom.actual_h_sigma,
        "n_per_class": n_per_class,
        "effective_support": (n_per_class - 1) / geom.actual_reff,
        "requested_effective_support": float(effective_support),
        "mean_scale": mean_scale,
        "oracle_o4_acc": oracle_o4_acc,
        "oracle_o5_acc": oracle_o5_acc,
        "estimated_o4_acc": estimated_o4_acc,
        "estimated_o5_acc": estimated_o5_acc,
        "delta_o5_minus_o4": delta,
        "oracle_expressivity_gain": expressivity_gain,
        "o4_estimation_error": o4_est_error,
        "o5_estimation_error": o5_est_error,
        "extra_o5_estimation_burden": extra_o5_burden,
        "decomposition_residual": decomposition_residual,
        "o4_condition_number": condition_number(o4.cov),
        "o5_condition_number_mean": float(o5_conds.mean()),
        "o5_condition_number_max": float(o5_conds.max()),
        "o4_shrinkage": float(o4.shrinkage if o4.shrinkage is not None else np.nan),
        "o5_shrinkage_mean": float(np.mean(o5.shrinkages)) if o5.shrinkages is not None else np.nan,
        "object_recovery_max_abs": object_recovery_max_abs,
        "test_n": len(yte),
    }
