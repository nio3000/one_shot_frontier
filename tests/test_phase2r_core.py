import numpy as np

from frontier.phase1_objects import central_sufficient_stats
from frontier.phase2_selector import build_folded_one_shot_object
from frontier.phase2r_selector import select_tau_objective, objective_curve


def _toy(seed=7):
    rng=np.random.default_rng(seed)
    C,d=3,6
    X=[]; y=[]; clients=[]
    for c in range(C):
        mu=np.zeros(d); mu[c]=1.5
        cov=np.eye(d)*(0.8+0.15*c)
        z=rng.multivariate_normal(mu,cov,size=120)
        X.append(z); y.append(np.full(120,c)); clients.append(np.arange(120)%6)
    return np.vstack(X),np.concatenate(y),np.concatenate(clients),C


def test_phase2r_objectives_finite_and_on_grid():
    X,y,client,C=_toy()
    obj=build_folded_one_shot_object(X,y,client,6,C,2,20260909,"toy")
    taus=[0.0,0.5,1.0]
    for oid in ["nll_control","mean_pairwise_expected_margin","hard_competitor_expected_margin"]:
        t,s=select_tau_objective(obj,taus,0.10,oid)
        assert t in taus
        assert len(s)==3
        assert np.isfinite(list(s.values())).all()


def test_phase2r_deterministic():
    X,y,client,C=_toy()
    obj1=build_folded_one_shot_object(X,y,client,6,C,2,20260909,"toy")
    obj2=build_folded_one_shot_object(X,y,client,6,C,2,20260909,"toy")
    for oid in ["mean_pairwise_expected_margin","hard_competitor_expected_margin"]:
        assert objective_curve(obj1,[0,0.5,1],0.1,oid)==objective_curve(obj2,[0,0.5,1],0.1,oid)
