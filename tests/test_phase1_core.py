from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase1_eval import crossfit_interior_vs_endpoints, stratified_paired_bootstrap_bacc_delta
from frontier.phase1_featurebank import load_feature_bank
from frontier.phase1_models import covariance_family, predict_tau_path
from frontier.phase1_objects import central_sufficient_stats, federated_sufficient_stats, max_recovery_error
from frontier.phase1_partition import dirichlet_label_partition
from frontier.phase1_projection import orthoproject_matrix, project_features


def _toy(seed=1):
    rng = np.random.default_rng(seed)
    C, d = 4, 12
    X, y = [], []
    for c in range(C):
        X.append(rng.normal(loc=c * 0.5, scale=1.0 + c * 0.05, size=(80, d)))
        y.append(np.full(80, c))
    return np.vstack(X), np.concatenate(y)


def test_projection_is_deterministic_and_orthogonal():
    P1 = orthoproject_matrix(20, 8, "enc", 20260908)
    P2 = orthoproject_matrix(20, 8, "enc", 20260908)
    assert np.allclose(P1, P2)
    assert np.allclose(P1.T @ P1, np.eye(8), atol=1e-10)


def test_partition_is_deterministic_and_covers_every_record():
    X, y = _toy()
    a = dirichlet_label_partition(y, 20, 0.1, 20260908, "toy")
    b = dirichlet_label_partition(y, 20, 0.1, 20260908, "toy")
    assert np.array_equal(a, b)
    assert len(a) == len(y)
    assert a.min() >= 0 and a.max() < 20


def test_federated_object_recovers_central_stats():
    X, y = _toy()
    client = dirichlet_label_partition(y, 20, 0.1, 20260908, "toy")
    a = central_sufficient_stats(X, y)
    b, payload = federated_sufficient_stats(X, y, client, 20)
    assert max_recovery_error(a, b) < 1e-9
    assert payload["class_specific_second_moment_bytes_float32"] > payload["prototype_bytes_float32"]


def test_tau_endpoints_have_expected_covariance_identity():
    X, y = _toy()
    s = central_sufficient_stats(X, y)
    c0 = covariance_family(s, 0.0, 0.1)
    c1 = covariance_family(s, 1.0, 0.1)
    assert np.max(np.abs(c0 - c0[0][None, :, :])) < 1e-12
    ridge = 0.1 * s.mean_variance_pool * np.eye(X.shape[1])
    assert np.allclose(c1, s.class_covs + ridge[None, :, :])


def test_tau_path_finite_even_when_projected():
    X, y = _toy()
    Xp = project_features(X, 8, "enc", 20260908)
    s = central_sufficient_stats(Xp, y)
    pred = predict_tau_path(s, Xp[:40], [0.0, 0.5, 1.0], 0.1)
    assert set(pred) == {0.0, 0.5, 1.0}
    assert all(np.isfinite(p).all() for p in pred.values())


def test_crossfit_selection_and_bootstrap_are_deterministic():
    _, y = _toy()
    y = y[:160]
    pred = {}
    for t in [0.0, 0.1, 0.5, 0.9, 1.0]:
        p = y.copy()
        # deterministic errors differ by tau
        nerr = int(30 * abs(t - 0.5))
        if nerr:
            p[:nerr] = (p[:nerr] + 1) % 4
        pred[t] = p
    r1 = crossfit_interior_vs_endpoints(y, pred, [0.1, 0.5, 0.9], [0.0, 1.0], 20260908)
    r2 = crossfit_interior_vs_endpoints(y, pred, [0.1, 0.5, 0.9], [0.0, 1.0], 20260908)
    assert r1["selected"] == r2["selected"]
    ci1 = stratified_paired_bootstrap_bacc_delta(y, r1["pred_interior"], r1["pred_endpoint"], 200, 20260908)
    ci2 = stratified_paired_bootstrap_bacc_delta(y, r1["pred_interior"], r1["pred_endpoint"], 200, 20260908)
    assert ci1 == ci2


def test_feature_bank_roundtrip(tmp_path):
    X, y = _toy()
    p = tmp_path / "toy.npz"
    meta = {"bank_id": "toy", "dataset_id": "toy", "modality": "engineering", "encoder_id": "toyenc"}
    np.savez(p, X_train=X.astype(np.float32), y_train=y, X_test=X[:40].astype(np.float32), y_test=y[:40], metadata_json=json.dumps(meta))
    b = load_feature_bank(p)
    assert b.bank_id == "toy"
    assert b.X_train.shape == X.shape
