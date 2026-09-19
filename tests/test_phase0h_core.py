from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from frontier.phase0h import evaluate_phase0h_path, n_per_class_from_rho, run_phase0h_population_replicate
from frontier.phase0h_generators import (
    build_phase0h_geometry,
    exponential_spectrum,
    prefixes_to_xy,
    sample_class_pools,
)
from frontier.phase0r import explicit_fixed_ridge_accuracy, unbiased_covariance_stats
from frontier.phase0h_predictors import feature_frame, fit_leave_one_dimension_out

ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    import yaml
    with open(ROOT / "configs" / "phase0h_protocol.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_exponential_spectrum_effective_rank_calibration():
    for d in [32, 64, 128]:
        for q in [0.125, 0.25, 0.5]:
            eig, _ = exponential_spectrum(d, q * d)
            assert abs(float(eig.sum() / eig.max()) - q * d) < 1e-6


def test_discovery_generator_preserves_pooled_covariance_and_H():
    g = build_phase0h_geometry(
        generator_family="mean_preserving_shared_eigenvector_eigenvalue_heterogeneity",
        C=5, d=64, q_ratio=0.25, target_h=0.4, seed=5000, mean_scale=1.85,
    )
    rel = np.linalg.norm(g.pooled_cov - g.sigma0, ord="fro") / np.linalg.norm(g.sigma0, ord="fro")
    assert rel < 1e-10
    assert abs(g.actual_h_sigma - 0.4) < 1e-3


def test_holdout_b_generator_reaches_h08_and_preserves_pool():
    for q in [0.1875, 0.375, 0.625]:
        g = build_phase0h_geometry(
            generator_family="mean_preserving_noncommuting_covariance_perturbation",
            C=5, d=96, q_ratio=q, target_h=0.8, seed=7000, mean_scale=1.85,
        )
        rel = np.linalg.norm(g.pooled_cov - g.sigma0, ord="fro") / np.linalg.norm(g.sigma0, ord="fro")
        assert rel < 1e-10
        assert abs(g.actual_h_sigma - 0.8) < 1e-3
        assert min(np.linalg.eigvalsh(c).min() for c in g.covs) > 0


def test_whitened_simplex_mahalanobis_geometry_is_invariant():
    vals = []
    for d, q in [(32,0.125),(64,0.25),(128,0.5)]:
        g = build_phase0h_geometry(
            generator_family="mean_preserving_shared_eigenvector_eigenvalue_heterogeneity",
            C=5, d=d, q_ratio=q, target_h=0.0, seed=5001, mean_scale=1.85,
        )
        inv = np.linalg.inv(g.sigma0)
        diff = g.means[0] - g.means[1]
        vals.append(float(diff @ inv @ diff))
    assert max(vals) - min(vals) < 1e-10


def test_nested_support_prefixes_are_exact():
    g = build_phase0h_geometry(
        generator_family="mean_preserving_shared_eigenvector_eigenvalue_heterogeneity",
        C=5,d=32,q_ratio=0.25,target_h=0.2,seed=5002,mean_scale=1.85,
    )
    pools = sample_class_pools(g, n_per_class_from_rho(2.0,32), seed=123)
    x1,y1 = prefixes_to_xy(pools, n_per_class_from_rho(0.5,32))
    x2,y2 = prefixes_to_xy(pools, n_per_class_from_rho(1.0,32))
    n1 = n_per_class_from_rho(0.5,32)
    n2 = n_per_class_from_rho(1.0,32)
    for c in range(5):
        a = x1[y1==c]
        b = x2[y2==c]
        assert np.array_equal(a, b[:n1])
        assert len(b)==n2


def test_tau0_and_tau1_path_match_explicit_fixed_ridge_qda():
    g = build_phase0h_geometry(
        generator_family="mean_preserving_shared_eigenvector_eigenvalue_heterogeneity",
        C=5,d=32,q_ratio=0.25,target_h=0.4,seed=5003,mean_scale=1.85,
    )
    pools = sample_class_pools(g, 40, seed=321)
    Xtr,ytr = prefixes_to_xy(pools, 20)
    Xte,yte = prefixes_to_xy(sample_class_pools(g, 100, seed=654),100)
    stats=unbiased_covariance_stats(Xtr,ytr,5)
    acc,_=evaluate_phase0h_path(stats.means,stats.priors,stats.pooled_cov,stats.class_covs,Xte,yte,[0.0,1.0],0.10)
    a0=explicit_fixed_ridge_accuracy(stats,Xte,yte,0.0,0.10)
    a1=explicit_fixed_ridge_accuracy(stats,Xte,yte,1.0,0.10)
    assert abs(acc[0]-a0)<1e-12
    assert abs(acc[1]-a1)<1e-12


def test_deterministic_population_replay():
    p=_protocol()
    a=run_phase0h_population_replicate(p,"smoke",32,0.25,0.4,5000,rho_override=[0.5,1.0],tau_override=[0.0,0.5,1.0],test_n_override=500)
    b=run_phase0h_population_replicate(p,"smoke",32,0.25,0.4,5000,rho_override=[0.5,1.0],tau_override=[0.0,0.5,1.0],test_n_override=500)
    # Ignore wall-time fields only.
    for ra,rb in zip(a[0],b[0]):
        ra={k:v for k,v in ra.items() if 'wall_time' not in k}; rb={k:v for k,v in rb.items() if 'wall_time' not in k}
        assert ra==rb
    assert a[1]==b[1]
    assert a[2]==b[2]


def test_hinge_features_are_fixed_at_rho1():
    cells=pd.DataFrame({
        'actual_H_sigma_mean':[0.4,0.4,0.4],
        'actual_rho_mean':[0.5,1.0,2.0],
        'q_actual_mean':[0.25,0.25,0.25],
        'd':[64,64,64],
    })
    x=feature_frame(cells)
    assert x.loc[0,'u_minus']<0 and x.loc[0,'u_plus']==0
    assert x.loc[1,'u_minus']==0 and x.loc[1,'u_plus']==0
    assert x.loc[2,'u_minus']==0 and x.loc[2,'u_plus']>0


def test_protocol_implied_discovery_counts():
    p=_protocol(); d=p['discovery']
    structural=len(d['dimensions'])*len(d['q_effective_rank_over_d'])*len(d['H_sigma'])*len(d['rho_nminus1_over_d'])
    runs=structural*len(d['seeds'])
    curves=runs*len(p['fixed']['tau'])
    assert structural==315
    assert runs==9450
    assert curves==103950
