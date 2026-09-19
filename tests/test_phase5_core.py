import json
from pathlib import Path

import numpy as np
import pandas as pd

from frontier.phase1_objects import central_sufficient_stats
from frontier.phase5_domain_tools import deterministic_group_roles, no_group_leakage, selected_pair_count
from frontier.phase5_summary_geometry import second_order_distance, m3_upload_ratio
from frontier.phase5_pairing import build_pair_selection
from frontier.phase5_gates import evaluate_phase5_gates
from frontier.phase5_wilds_adapters import DATASET_SPECS


def _toy():
    rng=np.random.default_rng(20260910)
    C,d=4,12
    X=[]; y=[]; dom=[]; grp=[]
    g=0
    for domain in range(3):
        for group in range(12):
            for c in range(C):
                n=6
                mu=np.zeros(d); mu[c]=1.0; mu[4]+=0.03*domain
                X.append(rng.normal(size=(n,d))+mu)
                y.extend([c]*n); dom.extend([domain]*n); grp.extend([g]*n)
            g += 1
    return np.vstack(X),np.asarray(y),np.asarray(dom),np.asarray(grp)


def _protocol():
    return {
        "fixed":{
            "group_split_seed":20260910,
            "group_fit_fraction":0.5,
            "class_subset_seed":20260910,
            "pair_near_fraction":0.2,
            "pair_min_selected":1,
            "pair_max_selected":10,
            "gate_bootstrap_replicates":100,
            "permutation_replicates":100,
            "gate_seed":20260910,
        },
        "tier1":{"datasets":{"camelyon17":{
            "min_shared_classes":4,
            "min_fit_per_class":5,
            "min_eval_per_class":5,
            "max_shared_classes":4,
        }}},
        "gates":{
            "E2_min_qualifying_conditions":1,
            "E3_robust_fraction_min":0.1,
            "E3_cluster_ci_low_min":0.0,
            "E4_spearman_min":-1.0,
            "E4_p_max":1.0,
            "E4_min_positive_datasets":0,
            "E5_max_upload_ratio":1.01,
        },
        "decisions":{"A":"A","B":"B","C":"C","D":"D"},
    }


def test_public_dataset_specs_locked():
    assert DATASET_SPECS["camelyon17"]["version"]=="1.0"
    assert DATASET_SPECS["rxrx1"]["version"]=="1.0"
    assert DATASET_SPECS["iwildcam"]["version"]=="2.0"


def test_group_hash_has_no_leakage():
    X,y,d,g=_toy()
    r=deterministic_group_roles("toy",d,g,20260910,0.5)
    assert set(np.unique(r)).issubset({0,1})
    assert no_group_leakage(d,g,r)


def test_second_order_distance_zero_identity():
    rng=np.random.default_rng(3)
    X=np.vstack([rng.normal(loc=c,size=(60,8)) for c in range(3)])
    y=np.repeat(np.arange(3),60)
    assert second_order_distance(X,y,X.copy(),y.copy(),[0,1,2]) < 1e-12


def test_pair_selection_is_deterministic_and_t2_only():
    X,y,d,g=_toy()
    p=_protocol()
    a=build_pair_selection("camelyon17","toy__enc","enc",12,X,y,d,g,p)
    b=build_pair_selection("camelyon17","toy__enc","enc",12,X,y,d,g,p)
    assert a==b
    assert a["n_selected_pairs"]==selected_pair_count(a["n_eligible_pairs"],0.2,1,10)
    assert all("t2_distance" in x and "m3_distance" not in x for x in a["selected_pairs"])


def test_m3_overhead_under_frozen_bound():
    assert m3_upload_ratio(64,16) <= 1.01
    assert m3_upload_ratio(128,16) <= 1.01


def test_gate_logic_retains_negative_rows():
    rows=[]
    for i in range(6):
        rows.append({
            "condition_id":f"c{i}",
            "dataset_id":["camelyon17","rxrx1","iwildcam"][i%3],
            "encoder_id":["resnet18_imagenet1k_v1","vit_b_16_imagenet1k_v1"][i%2],
            "projection_dim":[64,128][i%2],
            "robust_decision_critical":True,
            "m3_distance":float(i+1),
            "deterministic_pair_minimax_regret":0.01+0.001*i,
            "m3_upload_ratio_vs_t2":1.005,
        })
    # Add an explicit negative pair; it must remain in prevalence denominator.
    rows.append({
        "condition_id":"c0","dataset_id":"camelyon17",
        "encoder_id":"resnet18_imagenet1k_v1","projection_dim":64,
        "robust_decision_critical":False,"m3_distance":0.5,
        "deterministic_pair_minimax_regret":0.0,"m3_upload_ratio_vs_t2":1.005,
    })
    df=pd.DataFrame(rows)
    p=_protocol()
    out=evaluate_phase5_gates(df,p,{"manifest_ok":True,"pair_manifest_ok":True,"no_group_leakage":True})
    assert out["E3_prevalence"]["n_selected_pairs"]==7
    assert out["E3_prevalence"]["n_robust_pairs"]==6
