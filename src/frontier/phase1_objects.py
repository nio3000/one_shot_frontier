from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SufficientStats:
    counts: np.ndarray
    sums: np.ndarray
    second: np.ndarray
    means: np.ndarray
    class_covs: np.ndarray
    pooled_cov: np.ndarray
    priors: np.ndarray
    mean_variance_pool: float


def _stats_from_count_sum_second(counts: np.ndarray, sums: np.ndarray, second: np.ndarray) -> SufficientStats:
    counts = np.asarray(counts, dtype=np.int64)
    sums = np.asarray(sums, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    C, d = sums.shape
    if np.any(counts <= 0):
        raise ValueError("All classes must be present globally")
    means = sums / counts[:, None]
    covs = np.zeros((C, d, d), dtype=np.float64)
    pooled_scatter = np.zeros((d, d), dtype=np.float64)
    pooled_df = 0
    for c in range(C):
        scatter = second[c] - np.outer(sums[c], sums[c]) / counts[c]
        scatter = 0.5 * (scatter + scatter.T)
        if counts[c] > 1:
            covs[c] = scatter / (counts[c] - 1)
            pooled_scatter += scatter
            pooled_df += counts[c] - 1
        else:
            covs[c].fill(0.0)
    if pooled_df <= 0:
        raise ValueError("Insufficient pooled degrees of freedom")
    pooled = 0.5 * ((pooled_scatter / pooled_df) + (pooled_scatter / pooled_df).T)
    priors = counts / counts.sum()
    mv = float(np.trace(pooled) / d)
    return SufficientStats(counts, sums, second, means, covs, pooled, priors, mv)


def central_sufficient_stats(X: np.ndarray, y: np.ndarray, C: int | None = None) -> SufficientStats:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    C = int(C if C is not None else y.max() + 1)
    d = X.shape[1]
    counts = np.zeros(C, dtype=np.int64)
    sums = np.zeros((C, d), dtype=np.float64)
    second = np.zeros((C, d, d), dtype=np.float64)
    for c in range(C):
        xc = X[y == c]
        counts[c] = len(xc)
        if len(xc):
            sums[c] = xc.sum(axis=0, dtype=np.float64)
            second[c] = xc.T @ xc
    return _stats_from_count_sum_second(counts, sums, second)


def federated_sufficient_stats(X: np.ndarray, y: np.ndarray, client_ids: np.ndarray, n_clients: int, C: int | None = None) -> tuple[SufficientStats, dict[str, int]]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    client_ids = np.asarray(client_ids, dtype=np.int64)
    C = int(C if C is not None else y.max() + 1)
    d = X.shape[1]
    counts = np.zeros(C, dtype=np.int64)
    sums = np.zeros((C, d), dtype=np.float64)
    second = np.zeros((C, d, d), dtype=np.float64)
    present_pairs = 0
    for k in range(n_clients):
        mask_k = client_ids == k
        for c in range(C):
            xc = X[mask_k & (y == c)]
            if not len(xc):
                continue
            present_pairs += 1
            counts[c] += len(xc)
            sums[c] += xc.sum(axis=0, dtype=np.float64)
            second[c] += xc.T @ xc
    stats = _stats_from_count_sum_second(counts, sums, second)
    sym = d * (d + 1) // 2
    count_bytes = 8
    float_bytes = 4
    bytes_proto = present_pairs * (count_bytes + d * float_bytes)
    bytes_o5 = present_pairs * (count_bytes + (d + sym) * float_bytes)
    # O4 sends class counts/sums for present client-class pairs and one pooled second moment per non-empty client.
    nonempty_clients = int(len(np.unique(client_ids)))
    bytes_o4 = bytes_proto + nonempty_clients * sym * float_bytes
    return stats, {
        "present_client_class_pairs": int(present_pairs),
        "prototype_bytes_float32": int(bytes_proto),
        "pooled_covariance_bytes_float32": int(bytes_o4),
        "class_specific_second_moment_bytes_float32": int(bytes_o5),
    }


def max_recovery_error(a: SufficientStats, b: SufficientStats) -> float:
    vals = [
        np.max(np.abs(a.counts - b.counts)),
        np.max(np.abs(a.sums - b.sums)),
        np.max(np.abs(a.second - b.second)),
        np.max(np.abs(a.means - b.means)),
        np.max(np.abs(a.class_covs - b.class_covs)),
        np.max(np.abs(a.pooled_cov - b.pooled_cov)),
        np.max(np.abs(a.priors - b.priors)),
    ]
    return float(max(vals))
