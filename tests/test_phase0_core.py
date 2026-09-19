from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.classifiers import build_population_lda, build_population_qda, predict_lda, predict_qda
from frontier.synthetic import build_geometry, covariance_heterogeneity, effective_rank, make_covariances, sample_balanced


def test_h0_covariances_equal_and_oracles_agree():
    C, d = 5, 16
    covs = make_covariances("shared_eigenvectors_class_specific_eigenvalues", C, d, 8.0, 0.0, 1, 2)
    assert np.max(np.abs(covs - covs[0])) < 1e-10
    h, pooled = covariance_heterogeneity(covs, np.ones(C) / C)
    assert abs(h) < 1e-12
    geom = build_geometry("shared_eigenvectors_class_specific_eigenvalues", C, d, 8.0, 0.0, 1, 2, 2.0)
    X, y = sample_balanced(geom.means, geom.covs, 500, 123)
    lda = build_population_lda(geom.means, geom.pooled_cov, geom.priors)
    qda = build_population_qda(geom.means, geom.covs, geom.priors)
    assert np.array_equal(predict_lda(lda, X), predict_qda(qda, X))


def test_effective_rank_target_is_close():
    covs = make_covariances("shared_eigenvectors_class_specific_eigenvalues", 5, 32, 10.0, 0.0, 3, 4)
    r = effective_rank(covs[0])
    assert abs(r - 10.0) < 1e-6


def test_requested_h_sigma_is_calibrated():
    covs = make_covariances("shared_eigenvectors_class_specific_eigenvalues", 5, 32, 10.0, 0.4, 3, 4)
    h, _ = covariance_heterogeneity(covs, np.ones(5) / 5)
    assert abs(h - 0.4) < 2e-4


def test_o5_object_recovers_o4_second_moment():
    geom = build_geometry("shared_eigenvectors_class_specific_eigenvalues", 5, 16, 8.0, 0.3, 1, 2, 2.0)
    X, y = sample_balanced(geom.means, geom.covs, 30, 99)
    B = X.T @ X
    S = sum((X[y == c].T @ X[y == c] for c in range(5)), start=np.zeros_like(B))
    assert np.max(np.abs(B - S)) < 1e-10
