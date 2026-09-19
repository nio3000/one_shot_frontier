from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Metrics:
    balanced_accuracy: float
    accuracy: float
    macro_f1: float


def metrics(y: np.ndarray, pred: np.ndarray) -> Metrics:
    return Metrics(
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        accuracy=float(accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
    )


def tau_0_5(y: np.ndarray, pred_by_tau: dict[float, np.ndarray], tolerance: float = 0.005) -> tuple[float, float, dict[float, float]]:
    scores = {float(t): float(balanced_accuracy_score(y, p)) for t, p in pred_by_tau.items()}
    best = max(scores.values())
    qualifying = [t for t in sorted(scores) if scores[t] >= best - tolerance]
    return float(min(qualifying)), float(best), scores


def _select_tau(y: np.ndarray, idx: np.ndarray, pred_by_tau: dict[float, np.ndarray], candidates: list[float]) -> float:
    vals = []
    for t in sorted(map(float, candidates)):
        vals.append((float(balanced_accuracy_score(y[idx], pred_by_tau[t][idx])), t))
    best = max(v[0] for v in vals)
    return float(min(t for score, t in vals if abs(score - best) <= 1e-15))


def crossfit_interior_vs_endpoints(
    y: np.ndarray,
    pred_by_tau: dict[float, np.ndarray],
    interior_candidates: list[float],
    endpoint_candidates: list[float],
    seed: int,
) -> dict[str, object]:
    idx = np.arange(len(y))
    a, b = train_test_split(idx, test_size=0.5, random_state=int(seed), stratify=y)
    pred_int = np.empty_like(y)
    pred_end = np.empty_like(y)
    selected = []
    for select_idx, eval_idx, name in [(a, b, "A_to_B"), (b, a, "B_to_A")]:
        ti = _select_tau(y, select_idx, pred_by_tau, interior_candidates)
        te = _select_tau(y, select_idx, pred_by_tau, endpoint_candidates)
        pred_int[eval_idx] = pred_by_tau[ti][eval_idx]
        pred_end[eval_idx] = pred_by_tau[te][eval_idx]
        selected.append({"fold": name, "interior_tau": ti, "endpoint_tau": te})
    mi = metrics(y, pred_int)
    me = metrics(y, pred_end)
    return {
        "pred_interior": pred_int,
        "pred_endpoint": pred_end,
        "selected": selected,
        "interior_metrics": mi,
        "endpoint_metrics": me,
        "delta_balanced_accuracy": float(mi.balanced_accuracy - me.balanced_accuracy),
    }


def stratified_paired_bootstrap_bacc_delta(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Exact stratified record bootstrap for BACC delta using multinomial category counts.

    Within each class, paired correctness difference is in {-1,0,+1}. Resampling records
    with replacement is equivalent to a multinomial draw over those three categories.
    """
    rng = np.random.default_rng(int(seed))
    classes = np.unique(y)
    boot = np.zeros(int(n_boot), dtype=np.float64)
    for c in classes:
        idx = np.flatnonzero(y == c)
        diff = (pred_a[idx] == y[idx]).astype(np.int8) - (pred_b[idx] == y[idx]).astype(np.int8)
        n = len(diff)
        counts = np.array([(diff == -1).sum(), (diff == 0).sum(), (diff == 1).sum()], dtype=np.float64)
        probs = counts / n
        draws = rng.multinomial(n, probs, size=int(n_boot))
        boot += (draws[:, 2] - draws[:, 0]) / n
    boot /= len(classes)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)
