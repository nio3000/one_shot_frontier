import json

import numpy as np
import pandas as pd

from frontier.n1_support_matched import (
    _sequential_binary_partition_correct_counts,
    balanced_accuracy_binary,
    canonicalize_institutions,
    context_support_table,
    derive_age_band,
    evaluate_n1_gate,
    natural_context_table,
    prepare_authoritative_rows,
    simulate_arm_c_exact_support,
    t_het_from_context_risks,
)


def _mini_protocol():
    return {
        "authoritative_input": {
            "required_columns": ["record_id","institution","age","age_band","y_true","prob_seed42","prob_seed52","prob_seed62","prob_mean","y_pred"],
            "record_id_unique": True,
        },
        "context_construction": {
            "institutions": ["I0"],
            "age_bands": ["<40","40-59","60-74","75+"],
            "eligibility": {"min_N":1,"min_positive":1,"min_negative":1},
        },
        "prediction_rule": {
            "seed_columns": ["prob_seed42","prob_seed52","prob_seed62"],
            "threshold": 0.3475392659505208,
            "probability_tolerance": 1e-12,
        },
    }


def _row(rid, age, y, probs):
    band = derive_age_band(pd.Series([age]))[0]
    pm = float(np.mean(probs))
    yp = int(pm >= 0.3475392659505208)
    return [rid,"I0",age,band,y,*probs,pm,yp]


def test_balanced_accuracy_binary_exact():
    y=np.array([1,1,0,0]); p=np.array([1,0,0,0])
    assert balanced_accuracy_binary(y,p)==0.75


def test_frozen_age_boundaries_exact():
    got = derive_age_band(pd.Series([39.999,40,59.999,60,74.999,75,90]))
    assert got.tolist()==["<40","40-59","40-59","60-74","60-74","75+","75+"]


def test_prepare_authoritative_rows_recomputes_mean_and_threshold():
    rows=[
        _row("a",35,1,[0.2,0.4,0.6]),
        _row("b",35,0,[0.1,0.1,0.1]),
        _row("c",45,1,[0.8,0.8,0.8]),
        _row("d",45,0,[0.2,0.2,0.2]),
        _row("e",65,1,[0.9,0.9,0.9]),
        _row("f",65,0,[0.1,0.1,0.1]),
        _row("g",80,1,[0.7,0.7,0.7]),
        _row("h",80,0,[0.1,0.1,0.1]),
    ]
    df=pd.DataFrame(rows,columns=_mini_protocol()["authoritative_input"]["required_columns"])
    out,audit=prepare_authoritative_rows(df,_mini_protocol())
    assert audit["age_band_recomputed_exact"]
    assert audit["y_pred_recomputed_exact"]
    assert set(out.context_id)=={"I0|<40","I0|40-59","I0|60-74","I0|75+"}


def test_majority_vote_substitution_is_rejected():
    p=_mini_protocol()
    # mean is below threshold, but 2/3 probabilities are individually > 0.2; force wrong supplied y_pred=1.
    df=pd.DataFrame([_row("a",35,1,[0.2,0.2,0.2])],columns=p["authoritative_input"]["required_columns"])
    df.loc[0,"y_pred"]=1
    try:
        prepare_authoritative_rows(df,p)
    except ValueError as e:
        assert "frozen probability-mean threshold" in str(e)
    else:
        raise AssertionError("wrong y_pred was not rejected")


def test_binary_partition_preserves_exact_class_support():
    rng=np.random.default_rng(7)
    sizes=np.array([3,4,5])
    x=_sequential_binary_partition_correct_counts(total_correct=7,total_incorrect=5,slot_sizes=sizes,replicates=1000,rng=rng)
    assert x.shape==(1000,3)
    assert np.all(x>=0) and np.all(x<=sizes[None,:])
    assert np.all(x.sum(axis=1)==7)


def _canonical_13():
    rng=np.random.default_rng(2)
    rows=[]; sid=0
    counts=[5,4,4]
    c=0
    for j,nctx in enumerate(counts):
        for k in range(nctx):
            np_=4+k%2; nn_=5+(k+j)%2
            for y,n in [(1,np_),(0,nn_)]:
                for i in range(n):
                    pred = y if rng.random()<0.8-0.1*(k%3) else 1-y
                    rows.append((f"s{sid}",f"I{j}",f"B{k}",f"C{c}",y,pred)); sid+=1
            c+=1
    return pd.DataFrame(rows,columns=["sample_id","institution_id","age_band","context_id","y_true","y_pred"])


def test_arm_c_deterministic_and_finite():
    rows=_canonical_13()
    nat=natural_context_table(rows)
    a=simulate_arm_c_exact_support(rows,nat,replicates=200,seed=20260921,ddof=0)
    b=simulate_arm_c_exact_support(rows,nat,replicates=200,seed=20260921,ddof=0)
    assert np.array_equal(a,b)
    assert np.isfinite(a).all() and len(a)==200
    assert np.isfinite(t_het_from_context_risks(nat,ddof=0))


def test_gate_rule_is_exact_conjunction():
    null=np.array([0.1]*19+[0.2])
    g=evaluate_n1_gate(natural_t_het=0.15,arm_c_null=null,alpha=0.05)
    assert g["median_condition_pass"] is True
    assert g["p_condition_pass"] is False
    assert g["decision"]=="N1_SUPPORT_DOMINATED"
    null2=np.linspace(0.0,0.09,100)
    g2=evaluate_n1_gate(natural_t_het=1.0,arm_c_null=null2,alpha=0.05)
    assert g2["decision"]=="N1_PASS_STRUCTURE_BEYOND_SUPPORT"


def test_authoritative_institution_representation_normalization_exact():
    raw = pd.Series(["ptb_xl", "chapman_shaoxing", "ningbo", "cpsc_family"])
    expected = ["PTB-XL", "Chapman-Shaoxing", "Ningbo", "CPSC-family"]
    mapped, audit = canonicalize_institutions(raw, expected)
    assert mapped.tolist() == expected
    assert audit == {
        "ptb_xl": "PTB-XL",
        "chapman_shaoxing": "Chapman-Shaoxing",
        "ningbo": "Ningbo",
        "cpsc_family": "CPSC-family",
    }


def test_institution_normalization_rejects_unknown_or_ambiguous_aliases():
    expected = ["PTB-XL", "Chapman-Shaoxing"]
    try:
        canonicalize_institutions(pd.Series(["ptb_xl", "other_site"]), expected)
    except ValueError as e:
        assert "unknown_actual" in str(e)
    else:
        raise AssertionError("unknown institution was not rejected")

    try:
        canonicalize_institutions(pd.Series(["ptb_xl", "PTB XL", "chapman_shaoxing"]), expected)
    except ValueError as e:
        assert "refusing merge" in str(e)
    else:
        raise AssertionError("ambiguous raw aliases were silently merged")
