from __future__ import annotations

import hashlib
import numpy as np

from .phase1_objects import central_sufficient_stats


def _stable_bit(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int(h[0] & 1)


def class_means(X: np.ndarray, y: np.ndarray, C: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    C = int(C if C is not None else y.max() + 1)
    out = np.zeros((C, X.shape[1]), dtype=np.float64)
    for c in range(C):
        xc = X[y == c]
        if len(xc) == 0:
            raise ValueError(f"class {c} absent")
        out[c] = xc.mean(axis=0, dtype=np.float64)
    return out


def reflection_mask(C: int, pattern: str, seed: int, bank_id: str) -> np.ndarray:
    if pattern == "all_reflect":
        return np.ones(int(C), dtype=bool)
    if pattern != "hash_half_reflect":
        raise ValueError(f"unknown reflection pattern: {pattern}")
    mask = np.array([
        bool(_stable_bit("phase4_reflect", int(seed), str(bank_id), int(c)))
        for c in range(int(C))
    ], dtype=bool)
    if C > 1 and (mask.all() or (~mask).all()):
        mask[0] = ~mask[0]
    return mask


def reflect_by_class(
    X: np.ndarray,
    y: np.ndarray,
    means: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    means = np.asarray(means, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    out = X.copy()
    for c in range(len(mask)):
        idx = np.flatnonzero(y == c)
        if mask[c] and len(idx):
            out[idx] = 2.0 * means[c] - X[idx]
    return out


def _normalized_max_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b)) / denom)


def second_order_equivalence(
    X_a: np.ndarray,
    X_b: np.ndarray,
    y: np.ndarray,
    C: int | None = None,
) -> dict[str, float | bool]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    C = int(C if C is not None else y.max() + 1)
    a = central_sufficient_stats(X_a, y, C)
    b = central_sufficient_stats(X_b, y, C)
    counts_equal = bool(np.array_equal(a.counts, b.counts))
    comm = max(
        _normalized_max_diff(a.sums, b.sums),
        _normalized_max_diff(a.second, b.second),
    )
    model = max(
        _normalized_max_diff(a.means, b.means),
        _normalized_max_diff(a.class_covs, b.class_covs),
        _normalized_max_diff(a.pooled_cov, b.pooled_cov),
        _normalized_max_diff(a.priors, b.priors),
    )
    return {
        "counts_equal": counts_equal,
        "normalized_communication_diff": float(comm),
        "normalized_model_parameter_diff": float(model),
    }
