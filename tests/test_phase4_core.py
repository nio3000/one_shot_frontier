import numpy as np

from frontier.phase4_interventions import class_means, reflection_mask, reflect_by_class, second_order_equivalence
from frontier.phase4_escape_summaries import (
    rademacher_directions,
    diagonal_third_summary,
    rademacher_third_summary,
    rademacher_fourth_summary,
    federated_raw_power_sum,
    central_raw_power_sum,
    normalized_raw_aggregation_error,
)
from frontier.phase4_eval import deterministic_minimax_regret, randomized_minimax_regret


def _toy(seed=11):
    rng = np.random.default_rng(seed)
    X=[]; y=[]
    for c in range(3):
        z=rng.lognormal(mean=0.1*c, sigma=0.5+0.05*c, size=(160,8))
        z[:,:2]+=1.2*c
        X.append(z); y.append(np.full(len(z),c))
    return np.vstack(X), np.concatenate(y)


def test_reflection_preserves_second_order_and_flips_third_all_classes():
    X,y=_toy()
    C=3
    mu=class_means(X,y,C)
    mask=np.ones(C,dtype=bool)
    Xr=reflect_by_class(X,y,mu,mask)
    eq=second_order_equivalence(X,Xr,y,C)
    assert eq["counts_equal"]
    assert eq["normalized_communication_diff"] < 1e-12
    assert eq["normalized_model_parameter_diff"] < 1e-12
    m3=diagonal_third_summary(X,y,C)
    m3r=diagonal_third_summary(Xr,y,C)
    assert np.allclose(m3r,-m3,atol=1e-9,rtol=1e-9)


def test_even_fourth_negative_control_is_invariant():
    X,y=_toy()
    C=3
    mu=class_means(X,y,C)
    Xr=reflect_by_class(X,y,mu,np.ones(C,dtype=bool))
    a=rademacher_fourth_summary(X,y,C,16,20260910)
    b=rademacher_fourth_summary(Xr,y,C,16,20260910)
    assert np.allclose(a,b,atol=1e-9,rtol=1e-9)


def test_hash_half_pattern_is_deterministic_and_nontrivial():
    a=reflection_mask(10,"hash_half_reflect",20260910,"bank")
    b=reflection_mask(10,"hash_half_reflect",20260910,"bank")
    assert np.array_equal(a,b)
    assert a.any() and (~a).any()


def test_federated_third_sketch_aggregation_matches_central():
    X,y=_toy()
    C=3; d=X.shape[1]
    R=rademacher_directions(d,16,20260910)
    client=np.arange(len(y))%7
    cent=central_raw_power_sum(X,y,C,R,3)
    fed,pairs=federated_raw_power_sum(X,y,client,7,C,R,3)
    assert pairs>0
    assert normalized_raw_aggregation_error(cent,fed) < 1e-12


def test_minimax_regret_matches_phase3_reference_shape():
    a=np.array([0.67,0.67,0.665,0.665,0.83,0.83,0.83,0.83,0.83,0.83,0.83])
    b=np.array([0.83,0.83,0.585,0.585,0.67,0.67,0.67,0.67,0.67,0.67,0.67])
    assert abs(deterministic_minimax_regret(a,b)-0.16) < 1e-12
    assert abs(randomized_minimax_regret(a,b)-0.08) < 1e-12


def test_phase4_gate_engine_can_reach_A_on_synthetic_rows():
    import pandas as pd
    from frontier.phase4_gates import evaluate_phase4_gates
    rows=[]
    datasets=[("a","nonmedical"),("b","medical"),("c","nonmedical")]
    encs=["r","v"]
    dims=[64,128]
    i=0
    for ds,mod in datasets:
        for enc in encs:
            for dim in dims:
                i+=1
                for intervention in ["all_reflect","hash_half_reflect"]:
                    rows.append({
                        "condition_id":f"c{i}","dataset_id":ds,"modality":mod,"encoder_id":enc,"projection_dim":dim,
                        "counts_equal":True,"t2_comm_norm_diff":0.0,"t2_model_norm_diff":0.0,"max_head_parameter_norm_diff":0.0,
                        "deterministic_pair_minimax_regret":0.01,"curve_linf_shift":0.02,"optimal_sets_disjoint":True,
                        "diag_m3_distance":0.2,"sketch_m3_distance":0.1,"sketch_m4_distance":0.0,
                        "diag_m3_aggregation_norm_error":0.0,"sketch_m3_aggregation_norm_error":0.0,
                        "diag_m3_upload_ratio_vs_t2":1.03,"sketch_m3_upload_ratio_vs_t2":1.005,
                    })
    protocol={
        "gates":{
            "D1":{"max_t2_norm_diff":1e-10},
            "D2":{"max_head_parameter_norm_diff":1e-10},
            "D3":{"min_pair_gap":0.005,"min_qualifying_conditions":6,"min_datasets":3,"require_medical":True,"require_nonmedical":True,"require_both_encoders":True,"require_both_dims":True},
            "D4":{"min_third_summary_distance":1e-6,"min_diag_separation_fraction":0.95,"min_sketch_separation_fraction":0.90,"max_fourth_control_distance":1e-8},
            "D5":{"max_aggregation_norm_error":1e-10,"max_diag_upload_ratio_vs_t2":1.05,"max_sketch_upload_ratio_vs_t2":1.01},
        },
        "decisions":{"A":"A","B":"B","C":"C"},
    }
    g=evaluate_phase4_gates(pd.DataFrame(rows),protocol)
    assert g["decision"]=="A"
    assert g["escape_candidate_promotion"]=="rademacher_m3_k16"
