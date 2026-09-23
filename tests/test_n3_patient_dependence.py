from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.n3_patient_dependence import (
    bind_ptb_patients,
    cluster_bootstrap_status,
    context_support,
    derive_age_band,
    evaluate_primary_gate,
    exact_support_matched_null,
    identity_fingerprint,
    monte_carlo_upper_p,
    natural_contexts,
    patient_selection_hash,
    prepare_ptbxl_metadata,
    select_one_ecg_per_patient,
    status_polarity_reversal,
    t_het,
    wilson_interval,
    wilson_status,
)


def protocol():
    return yaml.safe_load((ROOT / "configs" / "nature_upgrade_n3_protocol.yaml").read_text(encoding="utf-8"))


def test_age_band_boundaries_exact():
    a = pd.Series([39.999, 40, 59.999, 60, 74.999, 75])
    assert derive_age_band(a).tolist() == ["<40", "40-59", "40-59", "60-74", "60-74", "75+"]


def test_wilson_interval_contains_empirical_rate():
    lo, hi = wilson_interval(80, 100, 1.959963984540054)
    assert lo < 0.8 < hi


def test_hash_selection_is_global_and_order_invariant():
    d = pd.DataFrame([
        {"patient_id":"1","record_id":"a","ecg_id":"1","age_band":"40-59"},
        {"patient_id":"1","record_id":"b","ecg_id":"2","age_band":"75+"},
        {"patient_id":"2","record_id":"c","ecg_id":"3","age_band":"60-74"},
    ])
    a = select_one_ecg_per_patient(d, 20260922, "hash")
    b = select_one_ecg_per_patient(d.iloc[::-1].copy(), 20260922, "hash")
    assert set(a.record_id) == set(b.record_id)
    assert len(a) == 2
    h1 = patient_selection_hash("1", "a", 20260922)
    h2 = patient_selection_hash("1", "b", 20260922)
    expect = "a" if h1 < h2 else "b"
    assert a.loc[a.patient_id == "1", "record_id"].iloc[0] == expect


def test_ptb_binding_accepts_filename_and_numeric_representations():
    p = protocol()
    meta = pd.DataFrame({
        "ecg_id":[1,2,3], "patient_id":[10,10,11],
        "filename_lr":["records100/00000/00001_lr","records100/00000/00002_lr","records100/00000/00003_lr"],
        "filename_hr":["records500/00000/00001_hr","records500/00000/00002_hr","records500/00000/00003_hr"],
    })
    meta2, _ = prepare_ptbxl_metadata(meta, p)
    rows = pd.DataFrame({
        "record_id":["records500/00000/00001_hr.hea", "00002_lr", "3"],
        "institution_id":["PTB-XL"]*3, "age":[50,50,60], "age_band":["40-59","40-59","60-74"],
        "context_id":["PTB-XL|40-59","PTB-XL|40-59","PTB-XL|60-74"],
        "y_true":[0,1,0], "y_pred":[0,1,0], "prob_mean":[.1,.8,.1],
    })
    ptb, audit = bind_ptb_patients(rows, meta2)
    assert audit["linkage_fraction"] == 1.0
    assert ptb.patient_id.tolist() == ["10", "10", "11"]


def test_identity_fingerprint_counts_repeats_and_cross_age():
    d = pd.DataFrame({
        "patient_id":["a","a","b","c","c","c"],
        "age_band":["40-59","75+","60-74","40-59","40-59","40-59"],
    })
    fp = identity_fingerprint(d)
    assert fp["bound_ptbxl_rows"] == 6
    assert fp["unique_patients"] == 3
    assert fp["repeated_patients"] == 2
    assert fp["repeated_patient_records"] == 5
    assert fp["max_ecgs_per_patient"] == 3
    assert fp["cross_age_patients"] == 1


def test_wilson_status_and_polarity_reversal():
    d = pd.DataFrame({"y_true":[1]*100+[0]*100, "y_pred":[1]*90+[0]*10+[0]*95+[1]*5})
    s = wilson_status(d, source_bacc=0.95, margin=0.075, z=1.959963984540054)
    assert s["N"] == 200
    assert s["status"] in {"DEFINITE_FAILURE","DEFINITE_NONFAILURE","AMBIGUOUS"}
    assert status_polarity_reversal("DEFINITE_FAILURE", "DEFINITE_NONFAILURE")
    assert not status_polarity_reversal("AMBIGUOUS", "DEFINITE_NONFAILURE")


def test_cluster_bootstrap_is_deterministic_for_fixed_rng_seed():
    rows = []
    for p in range(30):
        for j in range(1 + (p % 3 == 0)):
            y = (p + j) % 2
            pred = y if p % 5 else 1-y
            rows.append({"patient_id":str(p), "y_true":y, "y_pred":pred})
    d = pd.DataFrame(rows)
    a = cluster_bootstrap_status(d, source_bacc=.92, margin=.075, replicates=300, rng=np.random.default_rng(123), batch_size=64)
    b = cluster_bootstrap_status(d, source_bacc=.92, margin=.075, replicates=300, rng=np.random.default_rng(123), batch_size=64)
    assert a == b
    assert a["replicates"] == 300


def _small_protocol():
    p = protocol()
    p["context"]["institutions"] = ["I0", "I1"]
    p["context"]["age_bands"] = ["40-59", "60-74"]
    p["context"]["support"] = {"min_N":1,"min_positive":1,"min_negative":1}
    return p


def test_support_natural_and_thet():
    p = _small_protocol()
    rows=[]
    for inst in ["I0","I1"]:
        for band in ["40-59","60-74"]:
            for i in range(20):
                y=i%2
                pred=y if not (band=="60-74" and i<4) else 1-y
                rows.append({"institution_id":inst,"age_band":band,"context_id":f"{inst}|{band}","y_true":y,"y_pred":pred})
    d=pd.DataFrame(rows)
    s=context_support(d,p)
    n=natural_contexts(d,s)
    val=t_het(n,ddof=0)
    assert len(n)==4
    assert val>=0


def test_exact_support_matched_null_is_finite_and_deterministic():
    p = _small_protocol()
    rows=[]
    for inst in ["I0","I1"]:
        for band in ["40-59","60-74"]:
            for i in range(30):
                y=i%2
                pred=y if (i%7) else 1-y
                rows.append({"institution_id":inst,"age_band":band,"context_id":f"{inst}|{band}","y_true":y,"y_pred":pred})
    d=pd.DataFrame(rows)
    s=context_support(d,p)
    n=natural_contexts(d,s)
    a=exact_support_matched_null(d,n,replicates=200,seed=20260922,ddof=0)
    b=exact_support_matched_null(d,n,replicates=200,seed=20260922,ddof=0)
    assert np.array_equal(a,b)
    assert len(a)==200 and np.isfinite(a).all()
    assert 0 < monte_carlo_upper_p(a, float(np.median(a))) <= 1


def test_primary_gate_all_seven_conditions():
    p=protocol()
    g=evaluate_primary_gate(
        linkage_fraction=1.0, fingerprint_match=True, same_eligible_contexts=True,
        cluster_concordance=3, dedup_concordance=3,
        cluster_polarity_reversals=0, dedup_polarity_reversals=0,
        replay_p=0.01, natural_t=0.002, null_median=0.0001, protocol=p)
    assert g["pass"]
    assert g["decision"]=="N3_PASS_PATIENT_DEPENDENCE_ROBUST"
    g2=evaluate_primary_gate(
        linkage_fraction=1.0, fingerprint_match=True, same_eligible_contexts=True,
        cluster_concordance=2, dedup_concordance=3,
        cluster_polarity_reversals=0, dedup_polarity_reversals=0,
        replay_p=0.01, natural_t=0.002, null_median=0.0001, protocol=p)
    assert not g2["pass"]
    assert g2["decision"]=="N3_PATIENT_DEPENDENCE_SENSITIVITY_DETECTED"
