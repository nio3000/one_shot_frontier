from __future__ import annotations

import numpy as np

from .phase1_objects import central_sufficient_stats
from .phase4_eval import pair_decision_metrics
from .phase1_models import covariance_family


def _heads(stats, tau: float, alpha0: float):
    covs=covariance_family(stats,float(tau),float(alpha0))
    out=[]
    for c in range(len(stats.counts)):
        eig,Q=np.linalg.eigh(covs[c])
        if eig.min() <= 0 or not np.isfinite(eig).all():
            raise ValueError("non-SPD covariance")
        mu=stats.means[c]
        out.append((eig,Q,mu,float(np.log(eig).sum()),float(np.log(max(stats.priors[c],1e-15)))))
    return out


def _pred(X: np.ndarray, heads) -> np.ndarray:
    X=np.asarray(X,dtype=np.float64)
    scores=np.empty((len(X),len(heads)),dtype=np.float64)
    for c,(eig,Q,mu,logdet,logprior) in enumerate(heads):
        z=(X-mu)@Q
        mahal=np.sum((z*z)/eig,axis=1)
        scores[:,c]=-0.5*(mahal+logdet)+logprior
    return scores.argmax(1).astype(np.int64)


def _bacc_curve_from_pred(y: np.ndarray, preds: np.ndarray) -> np.ndarray:
    y=np.asarray(y,dtype=np.int64)
    vals=[]
    for t in range(preds.shape[0]):
        per=[]
        for c in np.unique(y):
            m=y==c
            per.append(float(np.mean(preds[t,m]==c)))
        vals.append(float(np.mean(per)))
    return np.asarray(vals,dtype=np.float64)


def domain_risk_curve(Xfit, yfit, Xeval, yeval, taus, alpha0):
    C=int(max(np.max(yfit),np.max(yeval))+1)
    stats=central_sufficient_stats(Xfit,yfit,C)
    preds=[]
    for tau in taus:
        preds.append(_pred(Xeval,_heads(stats,float(tau),float(alpha0))))
    preds=np.stack(preds,axis=0)
    return _bacc_curve_from_pred(yeval,preds), preds


def bootstrap_gamma_ci(y_a, pred_a, y_b, pred_b, taus, n_boot: int, seed: int) -> tuple[float,float]:
    rng=np.random.default_rng(int(seed))
    y_a=np.asarray(y_a,dtype=np.int64); y_b=np.asarray(y_b,dtype=np.int64)
    classes_a=[np.where(y_a==c)[0] for c in np.unique(y_a)]
    classes_b=[np.where(y_b==c)[0] for c in np.unique(y_b)]
    vals=np.empty(int(n_boot),dtype=np.float64)
    for b in range(int(n_boot)):
        ca=np.zeros(pred_a.shape[0],dtype=np.float64)
        cb=np.zeros(pred_b.shape[0],dtype=np.float64)
        for idx in classes_a:
            s=rng.choice(idx,size=len(idx),replace=True)
            ca += np.mean(pred_a[:,s] == y_a[s][None,:],axis=1)
        ca /= len(classes_a)
        for idx in classes_b:
            s=rng.choice(idx,size=len(idx),replace=True)
            cb += np.mean(pred_b[:,s] == y_b[s][None,:],axis=1)
        cb /= len(classes_b)
        vals[b]=pair_decision_metrics(ca,cb,taus)["deterministic_pair_minimax_regret"]
    lo,hi=np.quantile(vals,[0.025,0.975])
    return float(lo),float(hi)
