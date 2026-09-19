from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

from .phase1_objects import SufficientStats, _stats_from_count_sum_second
from .phase1_models import covariance_family


def _stable_seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**32 - 1)


@dataclass(frozen=True)
class FoldedObject:
    fold_stats: tuple[SufficientStats, ...]
    full_stats: SufficientStats
    fold_ids: np.ndarray
    payload: dict[str, int | float]


def deterministic_client_class_folds(
    y: np.ndarray,
    client_ids: np.ndarray,
    n_folds: int,
    seed: int,
    bank_id: str,
) -> np.ndarray:
    """Assign each client-class record to a deterministic local fold.

    The assignment is made before communication.  Each client sends fold-specific
    sufficient statistics in one message; the server never needs raw training vectors.
    """
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    client_ids = np.asarray(client_ids, dtype=np.int64).reshape(-1)
    if len(y) != len(client_ids):
        raise ValueError("y/client_ids length mismatch")
    if n_folds < 2:
        raise ValueError("n_folds must be >=2")
    folds = np.empty(len(y), dtype=np.int64)
    for k in np.unique(client_ids):
        for c in np.unique(y[client_ids == k]):
            idx = np.flatnonzero((client_ids == k) & (y == c))
            rng = np.random.default_rng(_stable_seed("phase2_fold", seed, bank_id, int(k), int(c)))
            idx = idx[rng.permutation(len(idx))]
            folds[idx] = np.arange(len(idx), dtype=np.int64) % int(n_folds)
    if np.any((folds < 0) | (folds >= n_folds)):
        raise RuntimeError("Incomplete fold assignment")
    return folds


def _aggregate_by_fold(
    X: np.ndarray,
    y: np.ndarray,
    client_ids: np.ndarray,
    fold_ids: np.ndarray,
    n_clients: int,
    n_folds: int,
    C: int,
) -> FoldedObject:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    client_ids = np.asarray(client_ids, dtype=np.int64)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    d = X.shape[1]
    counts_f = np.zeros((n_folds, C), dtype=np.int64)
    sums_f = np.zeros((n_folds, C, d), dtype=np.float64)
    second_f = np.zeros((n_folds, C, d, d), dtype=np.float64)
    present = 0
    for k in range(int(n_clients)):
        mk = client_ids == k
        for f in range(int(n_folds)):
            mf = mk & (fold_ids == f)
            if not np.any(mf):
                continue
            for c in range(C):
                xc = X[mf & (y == c)]
                if not len(xc):
                    continue
                present += 1
                counts_f[f, c] += len(xc)
                sums_f[f, c] += xc.sum(axis=0, dtype=np.float64)
                second_f[f, c] += xc.T @ xc
    fold_stats = tuple(_stats_from_count_sum_second(counts_f[f], sums_f[f], second_f[f]) for f in range(n_folds))
    counts = counts_f.sum(axis=0)
    sums = sums_f.sum(axis=0)
    second = second_f.sum(axis=0)
    full = _stats_from_count_sum_second(counts, sums, second)
    sym = d * (d + 1) // 2
    folded_bytes = present * (8 + (d + sym) * 4)
    # Full class-specific object if folds were not distinguished.
    present_pairs = 0
    for k in range(int(n_clients)):
        mk = client_ids == k
        for c in range(C):
            if np.any(mk & (y == c)):
                present_pairs += 1
    full_o5_bytes = present_pairs * (8 + (d + sym) * 4)
    return FoldedObject(
        fold_stats=fold_stats,
        full_stats=full,
        fold_ids=fold_ids,
        payload={
            "present_client_class_fold_pairs": int(present),
            "present_client_class_pairs": int(present_pairs),
            "adaptive_folded_object_bytes_float32": int(folded_bytes),
            "class_specific_object_bytes_float32": int(full_o5_bytes),
            "adaptive_upload_ratio_vs_class_specific_object": float(folded_bytes / max(full_o5_bytes, 1)),
        },
    )


def build_folded_one_shot_object(
    X: np.ndarray,
    y: np.ndarray,
    client_ids: np.ndarray,
    n_clients: int,
    C: int,
    n_folds: int,
    seed: int,
    bank_id: str,
) -> FoldedObject:
    folds = deterministic_client_class_folds(y, client_ids, n_folds, seed, bank_id)
    return _aggregate_by_fold(X, y, client_ids, folds, n_clients, n_folds, C)


def _heldout_balanced_conditional_nll(train: SufficientStats, val: SufficientStats, tau: float, alpha0: float) -> float:
    """Balanced per-class held-out Gaussian conditional NLL from moments only."""
    covs = covariance_family(train, float(tau), float(alpha0))
    vals = []
    for c in range(len(train.counts)):
        n = int(val.counts[c])
        if n <= 0:
            continue
        mu = train.means[c]
        eig, Q = np.linalg.eigh(covs[c])
        if eig.min() <= 0 or not np.isfinite(eig).all():
            return float("inf")
        inv = (Q / eig) @ Q.T
        # Sum_i (x_i-mu)(x_i-mu)^T from held-out sufficient statistics.
        centered_second = (
            val.second[c]
            - np.outer(val.sums[c], mu)
            - np.outer(mu, val.sums[c])
            + n * np.outer(mu, mu)
        )
        centered_second = 0.5 * (centered_second + centered_second.T)
        mahal_sum = float(np.sum(inv * centered_second.T))
        logdet = float(np.log(eig).sum())
        prior = max(float(train.priors[c]), 1e-15)
        nll = 0.5 * (n * logdet + mahal_sum) - n * np.log(prior)
        vals.append(nll / n)
    if not vals:
        return float("inf")
    return float(np.mean(vals))


def select_tau_folded_summary_cv(obj: FoldedObject, taus: list[float], alpha0: float) -> tuple[float, dict[float, float]]:
    if len(obj.fold_stats) != 2:
        raise ValueError("Phase 2 primary selector currently requires exactly two folds")
    scores: dict[float, float] = {}
    a, b = obj.fold_stats
    for tau in map(float, taus):
        score = 0.5 * (
            _heldout_balanced_conditional_nll(a, b, tau, alpha0)
            + _heldout_balanced_conditional_nll(b, a, tau, alpha0)
        )
        scores[tau] = float(score)
    best = min(scores.values())
    selected = min(t for t in sorted(scores) if abs(scores[t] - best) <= 1e-12)
    return float(selected), scores
