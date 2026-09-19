from __future__ import annotations

import numpy as np

from .phase1_objects import central_sufficient_stats
from .phase4_escape_summaries import rademacher_third_summary, normalized_summary_distance


def remap_classes(X: np.ndarray, y: np.ndarray, classes: list[int]) -> tuple[np.ndarray,np.ndarray,dict[int,int]]:
    classes = list(map(int, classes))
    mp={c:i for i,c in enumerate(classes)}
    mask=np.isin(y, classes)
    yy=np.asarray([mp[int(v)] for v in np.asarray(y)[mask]], dtype=np.int64)
    return np.asarray(X)[mask], yy, mp


def second_order_distance(Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray, classes: list[int]) -> float:
    Xa2, ya2, _ = remap_classes(Xa, ya, classes)
    Xb2, yb2, _ = remap_classes(Xb, yb, classes)
    C=len(classes)
    a=central_sufficient_stats(Xa2, ya2, C)
    b=central_sufficient_stats(Xb2, yb2, C)
    d=Xa2.shape[1]
    terms=[]
    for c in range(C):
        s2=0.5*(np.trace(a.class_covs[c])/d + np.trace(b.class_covs[c])/d) + 1e-12
        prior=(np.log(max(a.priors[c],1e-15))-np.log(max(b.priors[c],1e-15)))**2
        mean=float(np.sum((a.means[c]-b.means[c])**2)/(d*s2))
        cov=float(np.sum((a.class_covs[c]-b.class_covs[c])**2)/(d*s2*s2))
        terms.append((prior+mean+cov)/3.0)
    return float(np.sqrt(np.mean(terms)))


def m3_distance(Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray, classes: list[int], k: int, seed: int) -> float:
    Xa2, ya2, _ = remap_classes(Xa, ya, classes)
    Xb2, yb2, _ = remap_classes(Xb, yb, classes)
    C=len(classes)
    ma=rademacher_third_summary(Xa2, ya2, C, int(k), int(seed))
    mb=rademacher_third_summary(Xb2, yb2, C, int(k), int(seed))
    return normalized_summary_distance(ma, mb)


def m3_upload_ratio(d: int, k: int) -> float:
    sym=int(d)*(int(d)+1)//2
    t2_scalars=1+int(d)+sym
    return float((t2_scalars+int(k))/t2_scalars)
