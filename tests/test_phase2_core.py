from __future__ import annotations
import numpy as np

from frontier.phase1_objects import central_sufficient_stats, max_recovery_error
from frontier.phase1_partition import dirichlet_label_partition
from frontier.phase1_models import predict_tau_path
from frontier.phase2_selector import build_folded_one_shot_object, select_tau_folded_summary_cv
from frontier.phase2_eval import evaluate_adaptive_condition
from frontier.phase2_baseline_adapter import validate_adapter_entry


def toy(seed=3):
    rng=np.random.default_rng(seed); C=3; d=8
    X=[]; y=[]
    for c in range(C):
        mu=np.zeros(d); mu[c]=1.5
        X.append(rng.normal(size=(80,d))+mu); y.append(np.full(80,c))
    return np.vstack(X),np.concatenate(y)


def test_folded_object_recovers_full_stats():
    X,y=toy(); clients=dirichlet_label_partition(y,5,0.3,7,"toy")
    obj=build_folded_one_shot_object(X,y,clients,5,3,2,9,"toy")
    cen=central_sufficient_stats(X,y,3)
    assert max_recovery_error(cen,obj.full_stats) < 1e-10
    assert obj.payload["adaptive_upload_ratio_vs_class_specific_object"] <= 2.0


def test_selector_is_deterministic_and_grid_member():
    X,y=toy(); clients=dirichlet_label_partition(y,5,0.3,7,"toy")
    obj=build_folded_one_shot_object(X,y,clients,5,3,2,9,"toy")
    taus=[0.0,0.5,1.0]
    a,sa=select_tau_folded_summary_cv(obj,taus,0.1); b,sb=select_tau_folded_summary_cv(obj,taus,0.1)
    assert a==b and sa==sb and a in taus


def test_eval_returns_upper_bounds_and_ci():
    X,y=toy(); clients=dirichlet_label_partition(y,5,0.3,7,"toy")
    obj=build_folded_one_shot_object(X,y,clients,5,3,2,9,"toy")
    path=predict_tau_path(obj.full_stats,X,[0.0,0.5,1.0],0.1)
    out=evaluate_adaptive_condition(y,path,0.5,[0.0,1.0],[0.5],100,1,0.005)
    assert "adaptive_gain_ci95_low" in out
    assert out["oracle_best_bacc_test"] >= out["best_endpoint_bacc_test_oracle"] - 1e-12


def test_unqualified_external_adapter_blocked():
    q=validate_adapter_entry({"method_id":"X","execution_track":"model_level_external","official_repo":"x","qualified":False})
    assert not q.qualified and len(q.reasons)>0
