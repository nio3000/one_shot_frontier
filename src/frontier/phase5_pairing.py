from __future__ import annotations

import numpy as np

from .phase5_domain_tools import (
    class_counts,
    deterministic_class_subset,
    deterministic_group_roles,
    no_group_leakage,
    selected_pair_count,
)
from .phase5_summary_geometry import second_order_distance


def build_pair_selection(
    dataset_id: str,
    bank_id: str,
    encoder_id: str,
    projection_dim: int,
    X: np.ndarray,
    y: np.ndarray,
    domain_id: np.ndarray,
    group_id: np.ndarray,
    protocol: dict,
) -> dict:
    roles=deterministic_group_roles(
        dataset_id,domain_id,group_id,
        int(protocol["fixed"]["group_split_seed"]),
        float(protocol["fixed"]["group_fit_fraction"]),
    )
    ds=protocol["tier1"]["datasets"][dataset_id]
    domains=sorted(map(int,np.unique(domain_id)))
    eligible=[]
    for i,a in enumerate(domains):
        for b in domains[i+1:]:
            pair_id=f"{dataset_id}__{int(a):04d}__{int(b):04d}"
            af=(domain_id==a)&(roles==0); ae=(domain_id==a)&(roles==1)
            bf=(domain_id==b)&(roles==0); be=(domain_id==b)&(roles==1)
            caf,cae,cbf,cbe=class_counts(y,af),class_counts(y,ae),class_counts(y,bf),class_counts(y,be)
            shared=[]
            for c in sorted(set(caf)&set(cae)&set(cbf)&set(cbe)):
                if (
                    caf[c] >= int(ds["min_fit_per_class"])
                    and cae[c] >= int(ds["min_eval_per_class"])
                    and cbf[c] >= int(ds["min_fit_per_class"])
                    and cbe[c] >= int(ds["min_eval_per_class"])
                ):
                    shared.append(int(c))
            if len(shared) < int(ds["min_shared_classes"]):
                continue
            shared=deterministic_class_subset(
                dataset_id,pair_id,shared,int(ds["max_shared_classes"]),int(protocol["fixed"]["class_subset_seed"])
            )
            Xa=X[af]; ya=y[af]; Xb=X[bf]; yb=y[bf]
            dist=second_order_distance(Xa,ya,Xb,yb,shared)
            eligible.append({
                "pair_id":pair_id,
                "domain_a":int(a),
                "domain_b":int(b),
                "shared_classes":[int(c) for c in shared],
                "n_shared_classes":int(len(shared)),
                "t2_distance":float(dist),
                "fit_n_a":int(np.sum(af & np.isin(y,shared))),
                "eval_n_a":int(np.sum(ae & np.isin(y,shared))),
                "fit_n_b":int(np.sum(bf & np.isin(y,shared))),
                "eval_n_b":int(np.sum(be & np.isin(y,shared))),
            })
    eligible=sorted(eligible,key=lambda r:(r["t2_distance"],r["pair_id"]))
    k=selected_pair_count(
        len(eligible),float(protocol["fixed"]["pair_near_fraction"]),
        int(protocol["fixed"]["pair_min_selected"]),int(protocol["fixed"]["pair_max_selected"])
    )
    selected=[]
    for rank,row in enumerate(eligible[:k],start=1):
        z=dict(row)
        z["selection_rank"]=rank
        z["eligible_pair_count"]=len(eligible)
        z["selected_pair_count"]=k
        z["t2_empirical_percentile"]=float(rank/max(len(eligible),1))
        selected.append(z)
    return {
        "condition_id":f"{bank_id}__d{int(projection_dim)}",
        "dataset_id":dataset_id,
        "bank_id":bank_id,
        "encoder_id":encoder_id,
        "projection_dim":int(projection_dim),
        "n_domains":len(domains),
        "n_eligible_pairs":len(eligible),
        "n_selected_pairs":k,
        "no_group_leakage":bool(no_group_leakage(domain_id,group_id,roles)),
        "selected_pairs":selected,
    }
