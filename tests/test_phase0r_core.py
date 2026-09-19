from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase0r import (
    _evaluate_covariance_path,
    evaluate_fixed_ridge_grid,
    explicit_fixed_ridge_accuracy,
    explicit_fixed_ridge_covs,
    run_phase0r_replicate,
    unbiased_covariance_stats,
)
from frontier.synthetic import build_geometry, sample_balanced
from frontier.utils import load_yaml


def _toy():
    geom = build_geometry(
        "shared_eigenvectors_class_specific_eigenvalues",
        C=3, d=8, target_reff=5.0, target_h=0.35,
        geometry_seed=11, perturbation_seed=22, mean_scale=0.8,
    )
    Xtr, ytr = sample_balanced(geom.means, geom.covs, 16, 123)
    Xte, yte = sample_balanced(geom.means, geom.covs, 50, 456)
    stats = unbiased_covariance_stats(Xtr, ytr, 3)
    return stats, Xte, yte


def test_tau0_covariances_identical():
    stats, _, _ = _toy()
    covs = explicit_fixed_ridge_covs(stats, tau=0.0, alpha=0.1)
    assert np.max(np.abs(covs - covs[0])) < 1e-12


def test_tau1_matches_explicit_qda_and_fast_path():
    stats, Xte, yte = _toy()
    rows = evaluate_fixed_ridge_grid(stats, Xte, yte, taus=[1.0], alphas=[0.1])
    fast = rows[0]["accuracy"]
    explicit = explicit_fixed_ridge_accuracy(stats, Xte, yte, tau=1.0, alpha=0.1)
    assert abs(fast - explicit) < 1e-12


def test_generalized_path_matches_explicit_all_taus():
    stats, Xte, yte = _toy()
    taus = [0.0, 0.25, 0.5, 0.75, 1.0]
    alpha = 0.03
    rows = evaluate_fixed_ridge_grid(stats, Xte, yte, taus=taus, alphas=[alpha])
    for row in rows:
        explicit = explicit_fixed_ridge_accuracy(stats, Xte, yte, row["tau"], alpha)
        assert abs(row["accuracy"] - explicit) < 1e-12


def test_all_fixed_ridge_covariances_spd():
    stats, _, _ = _toy()
    for alpha in [0.01, 0.1, 1.0]:
        for tau in [0.0, 0.3, 0.7, 1.0]:
            covs = explicit_fixed_ridge_covs(stats, tau, alpha)
            assert min(np.linalg.eigvalsh(c).min() for c in covs) > 0


def test_ridge_scale_is_alpha_times_pooled_mean_variance():
    stats, _, _ = _toy()
    c1 = explicit_fixed_ridge_covs(stats, 0.4, 0.1)
    c2 = explicit_fixed_ridge_covs(stats, 0.4, 0.3)
    diff = c2 - c1
    expected = 0.2 * stats.mean_variance_pool
    for c in range(len(diff)):
        assert np.max(np.abs(diff[c] - expected * np.eye(diff.shape[-1]))) < 1e-12


def test_phase0r_replicate_deterministic_smoke():
    protocol = load_yaml(ROOT / "configs" / "phase0r_protocol.yaml")
    a_run, a_curves = run_phase0r_replicate(
        protocol, "smoke", 0.4, 1.0, 2000,
        test_n_override=300,
        tau_override=[0.0, 0.5, 1.0],
        alpha_override=[0.03],
    )
    b_run, b_curves = run_phase0r_replicate(
        protocol, "smoke", 0.4, 1.0, 2000,
        test_n_override=300,
        tau_override=[0.0, 0.5, 1.0],
        alpha_override=[0.03],
    )
    ac = [(r["estimator_family"], r["alpha"], r["tau"], r["accuracy"]) for r in a_curves]
    bc = [(r["estimator_family"], r["alpha"], r["tau"], r["accuracy"]) for r in b_curves]
    assert ac == bc
    assert a_run["n_per_class"] == b_run["n_per_class"]
    assert a_run["test_n"] == b_run["test_n"]


def test_rotation_protocol_name_is_mapped_inside_phase0r():
    protocol = load_yaml(ROOT / "configs" / "phase0r_protocol.yaml")
    run, curves = run_phase0r_replicate(
        protocol, "holdout_b", 0.8, 1.0, 4000,
        test_n_override=200,
        tau_override=[0.0, 1.0],
        alpha_override=[0.1],
    )
    assert abs(run["actual_H_sigma"] - 0.8) < 0.01
    assert run["generator_family"] == "class_specific_eigenvector_rotation"
    assert len(curves) == 4
