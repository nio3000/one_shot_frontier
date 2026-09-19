from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_corr(x, y) -> float:
    a=pd.Series(np.asarray(x,dtype=float)).rank(method="average").to_numpy()
    b=pd.Series(np.asarray(y,dtype=float)).rank(method="average").to_numpy()
    if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return 0.0
    return float(np.corrcoef(a,b)[0,1])


def _cluster_bootstrap_fraction(df: pd.DataFrame, reps: int, seed: int) -> tuple[float,float,float]:
    clusters=sorted(df["condition_id"].unique())
    rng=np.random.default_rng(int(seed))
    vals=[]
    for _ in range(int(reps)):
        picked=rng.choice(clusters,size=len(clusters),replace=True)
        parts=[df[df["condition_id"]==c] for c in picked]
        z=pd.concat(parts,ignore_index=True)
        vals.append(float(z["robust_decision_critical"].mean()))
    lo,hi=np.quantile(vals,[0.025,0.975])
    return float(np.mean(df["robust_decision_critical"])),float(lo),float(hi)


def _blocked_perm_p(df: pd.DataFrame, reps: int, seed: int) -> tuple[float,float,dict]:
    obs=_rank_corr(df["m3_distance"],df["deterministic_pair_minimax_regret"])
    rng=np.random.default_rng(int(seed))
    ge=0
    vals=np.asarray(df["deterministic_pair_minimax_regret"],dtype=float)
    for _ in range(int(reps)):
        perm=vals.copy()
        for ds in df["dataset_id"].unique():
            idx=np.where(df["dataset_id"].to_numpy()==ds)[0]
            perm[idx]=rng.permutation(perm[idx])
        if _rank_corr(df["m3_distance"],perm) >= obs - 1e-15:
            ge += 1
    p=(ge+1)/(int(reps)+1)
    by={}
    for ds,g in df.groupby("dataset_id"):
        by[str(ds)]=_rank_corr(g["m3_distance"],g["deterministic_pair_minimax_regret"])
    return float(obs),float(p),by


def evaluate_phase5_gates(df: pd.DataFrame, protocol: dict, integrity: dict) -> dict:
    g=protocol["gates"]
    allfinite=bool(np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()) if len(df) else False
    E1=bool(integrity.get("manifest_ok") and integrity.get("pair_manifest_ok") and integrity.get("no_group_leakage") and allfinite)

    robust=df[df["robust_decision_critical"]]
    qconds=sorted(robust["condition_id"].unique().tolist())
    q=df[df["condition_id"].isin(qconds)]
    E2=bool(
        len(qconds) >= int(g["E2_min_qualifying_conditions"])
        and set(q["dataset_id"]) == {"camelyon17","rxrx1","iwildcam"}
        and q["encoder_id"].nunique() >= 2
        and q["projection_dim"].nunique() >= 2
    )

    frac,ci_lo,ci_hi=_cluster_bootstrap_fraction(df,int(protocol["fixed"]["gate_bootstrap_replicates"]),int(protocol["fixed"]["gate_seed"]))
    E3=bool(frac >= float(g["E3_robust_fraction_min"]) and ci_lo >= float(g["E3_cluster_ci_low_min"]))

    rho,p,by=_blocked_perm_p(df,int(protocol["fixed"]["permutation_replicates"]),int(protocol["fixed"]["gate_seed"]))
    pos=sum(v>0 for v in by.values())
    E4=bool(rho >= float(g["E4_spearman_min"]) and p <= float(g["E4_p_max"]) and pos >= int(g["E4_min_positive_datasets"]))

    max_ratio=float(df["m3_upload_ratio_vs_t2"].max()) if len(df) else float("inf")
    E5=bool(max_ratio <= float(g["E5_max_upload_ratio"]))

    if E1 and E2 and E3 and E4 and E5:
        decision=protocol["decisions"]["A"]
    elif E1 and E2 and E3 and E5:
        decision=protocol["decisions"]["B"]
    elif E1 and E5 and len(robust)>0:
        decision=protocol["decisions"]["C"]
    else:
        decision=protocol["decisions"]["D"]

    return {
        "E1_provenance_integrity":{"pass":E1},
        "E2_natural_gap_breadth":{
            "pass":E2,
            "n_qualifying_conditions":len(qconds),
            "datasets":sorted(set(q["dataset_id"].astype(str))),
            "encoders":sorted(set(q["encoder_id"].astype(str))),
            "projection_dims":sorted(set(q["projection_dim"].astype(int))),
        },
        "E3_prevalence":{
            "pass":E3,
            "robust_pair_fraction":frac,
            "cluster_bootstrap_ci95":[ci_lo,ci_hi],
            "n_selected_pairs":int(len(df)),
            "n_robust_pairs":int(len(robust)),
        },
        "E4_higher_order_corroboration":{
            "pass":E4,
            "pooled_spearman":rho,
            "blocked_permutation_p":p,
            "within_dataset_spearman":by,
            "n_datasets_positive_direction":pos,
        },
        "E5_communication":{
            "pass":E5,
            "max_m3_upload_ratio_vs_t2":max_ratio,
        },
        "decision":decision,
        "m3_escape_phase_authorized":bool(decision==protocol["decisions"]["A"]),
    }
