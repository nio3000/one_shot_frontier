from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from frontier.phase0s_generators import panel_geometry, sample_xy
from frontier.phase0s_predictor import freeze_predictor_from_phase0h, predict_spec
from frontier.phase0s import run_phase0s_replicate
from frontier.phase0s_gates import global_decision
from frontier.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_student_t_covariance_preserving_transform():
    geom = panel_geometry("panel_c", C=5, d=4, q_ratio=0.5, target_h=0.2, seed=123, mean_scale=1.85)
    X, y = sample_xy(geom, n_per_class=40000, seed=9, distribution="student_t", nu=5.0)
    for c in range(5):
        xc = X[y == c]
        cov = np.cov(xc, rowvar=False, ddof=1)
        rel = np.linalg.norm(cov - geom.covs[c], ord="fro") / np.linalg.norm(geom.covs[c], ord="fro")
        assert rel < 0.08


def test_noncommuting_geometry_preserves_pooled_covariance():
    geom = panel_geometry("panel_b", C=5, d=16, q_ratio=0.375, target_h=0.6, seed=44, mean_scale=1.85)
    rel = np.linalg.norm(geom.pooled_cov - geom.sigma0, ord="fro") / np.linalg.norm(geom.sigma0, ord="fro")
    assert rel < 1e-10
    assert abs(geom.actual_h_sigma - 0.6) < 1e-3


def test_phase0s_replicate_has_finite_curves():
    p = load_yaml(ROOT / "configs" / "phase0s_protocol.yaml")
    run, curves, oracle, mech = run_phase0s_replicate(
        p, "smoke", 16, 0.25, 0.2, 1.4, 11,
        tau_override=[0.0, 0.5, 1.0], test_n_override=500,
    )
    assert len(curves) == 3 and len(oracle) == 3
    assert all(np.isfinite(r["accuracy"]) for r in curves)
    assert run["tau0_class_cov_identity_max_abs"] == 0.0
    assert mech["population_pooled_cov_relative_fro_error"] < 1e-10


def test_predictor_freeze_prefers_phase0h_full_model(tmp_path: Path):
    cells = pd.DataFrame({
        "actual_H_sigma_mean": [0.0, 0.2, 0.4, 0.8],
        "effective_support_mean": [1.0, 2.0, 4.0, 8.0],
        "tau_0_5": [0.0, 0.1, 0.4, 0.8],
    })
    cells_path = tmp_path / "cells.csv"
    cells.to_csv(cells_path, index=False)
    spec = {"features": ["H", "log2_Seff", "H_x_log2_Seff"], "intercept": 0.1, "coefficients": {"H": 0.2, "log2_Seff": 0.3, "H_x_log2_Seff": 0.4}}
    fit = {"models": {"M_old": {"all_discovery_fit": spec}}}
    fit_path = tmp_path / "predictor_fit.json"
    fit_path.write_text(json.dumps(fit), encoding="utf-8")
    rs = tmp_path / "run_summary.json"
    rs.write_text(json.dumps({"git_head": "abc", "protocol_sha256": "def"}), encoding="utf-8")
    out = tmp_path / "freeze.json"
    frozen = freeze_predictor_from_phase0h(
        phase0h_predictor_fit=fit_path, phase0h_cells=cells_path, phase0h_run_summary=rs,
        output=out, fitting_git_head="ghi",
    )
    assert frozen["models"]["M_old"] == spec
    pred = predict_spec(cells, spec)
    assert np.all((pred >= 0) & (pred <= 1))
    assert frozen["source_phase0h_git_head"] == "abc"


def test_global_decision_contract():
    P = {"pass": True}
    F = {"pass": False}
    assert global_decision(P, P, P) == "SMOOTH_ESTIMABILITY_SURFACE_TRANSFER_SUPPORTED"
    assert global_decision(P, P, F) == "GAUSSIAN_GEOMETRY_TRANSFER_SUPPORTED_NON_GAUSSIAN_FAIL"
    assert global_decision(P, F, P) == "SCALE_TRANSFER_SUPPORTED_GEOMETRY_NOT_GENERAL"
    assert global_decision(F, P, P) == "SMOOTH_SURFACE_NOT_TRANSFERABLE"
