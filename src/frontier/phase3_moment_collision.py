
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np


BASE_SUPPORT = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
NULL_VECTOR = np.array([-1.0, 3.0, -3.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class ClassSpec:
    mu: float
    scale: float


@dataclass(frozen=True)
class WitnessConfig:
    class0: ClassSpec = ClassSpec(mu=-0.75, scale=0.30)
    class1: ClassSpec = ClassSpec(mu=0.75, scale=0.70)
    epsilon: float = 0.08
    alpha0: float = 0.10


def support(spec: ClassSpec) -> np.ndarray:
    return spec.mu + spec.scale * BASE_SUPPORT


def weights(epsilon: float, sign: int) -> np.ndarray:
    w = np.full(4, 0.25, dtype=np.float64) + float(sign) * float(epsilon) * NULL_VECTOR
    if np.min(w) < -1e-12:
        raise ValueError("epsilon creates a negative probability")
    if abs(float(w.sum()) - 1.0) > 1e-12:
        raise AssertionError("weights do not sum to one")
    return w


def task_distributions(cfg: WitnessConfig):
    """Return task A/B class-conditional 1-D discrete distributions.

    Task A: class0 uses +epsilon, class1 uses -epsilon.
    Task B: class0 uses -epsilon, class1 uses +epsilon.
    """
    s0 = support(cfg.class0)
    s1 = support(cfg.class1)
    return {
        "A": [(s0, weights(cfg.epsilon, +1)), (s1, weights(cfg.epsilon, -1))],
        "B": [(s0, weights(cfg.epsilon, -1)), (s1, weights(cfg.epsilon, +1))],
    }


def population_moments(points: np.ndarray, probs: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.sum(points * probs))
    second = float(np.sum((points ** 2) * probs))
    var = float(second - mean * mean)
    return mean, second, var


def summary_signature(task) -> tuple:
    out = []
    for points, probs in task:
        mean, second, _ = population_moments(points, probs)
        out.append((1.0, mean, second))
    return tuple(out)


def exact_empirical_counts(epsilon: float, sign: int, n_per_class: int = 100) -> tuple[int, ...]:
    """Exact counts for eps values with hundredth granularity and n=100."""
    eps = Fraction(str(float(epsilon)))
    base = Fraction(1, 4)
    v = [-1, 3, -3, 1]
    vals = [(base + sign * eps * vi) * n_per_class for vi in v]
    if not all(vv.denominator == 1 for vv in vals):
        raise ValueError("n_per_class does not yield exact integer witness counts")
    counts = tuple(int(vv) for vv in vals)
    if min(counts) < 0 or sum(counts) != n_per_class:
        raise ValueError("invalid exact empirical counts")
    return counts


def exact_empirical_stats(spec: ClassSpec, epsilon: float, sign: int, n_per_class: int = 100):
    # The primary witness uses decimal parameters with exact finite representations.
    mu = Fraction(str(spec.mu))
    scale = Fraction(str(spec.scale))
    base_pts = [Fraction(-3), Fraction(-1), Fraction(1), Fraction(3)]
    pts = [mu + scale * z for z in base_pts]
    counts = exact_empirical_counts(epsilon, sign, n_per_class)
    n = sum(counts)
    s = sum(p * c for p, c in zip(pts, counts))
    ss = sum(p * p * c for p, c in zip(pts, counts))
    return {"n": n, "sum": s, "second_sum": ss, "points": pts, "counts": counts}


def exact_collision_check(cfg: WitnessConfig, n_per_class: int = 100) -> dict:
    a0 = exact_empirical_stats(cfg.class0, cfg.epsilon, +1, n_per_class)
    b0 = exact_empirical_stats(cfg.class0, cfg.epsilon, -1, n_per_class)
    a1 = exact_empirical_stats(cfg.class1, cfg.epsilon, -1, n_per_class)
    b1 = exact_empirical_stats(cfg.class1, cfg.epsilon, +1, n_per_class)
    return {
        "class0_exact": (a0["n"], a0["sum"], a0["second_sum"]) == (b0["n"], b0["sum"], b0["second_sum"]),
        "class1_exact": (a1["n"], a1["sum"], a1["second_sum"]) == (b1["n"], b1["sum"], b1["second_sum"]),
        "class0_A": a0,
        "class0_B": b0,
        "class1_A": a1,
        "class1_B": b1,
    }


def third_central_moment(points: np.ndarray, probs: np.ndarray) -> float:
    m = float(np.sum(points * probs))
    return float(np.sum(((points - m) ** 3) * probs))


def _shared_and_class_vars(task) -> tuple[list[float], list[float]]:
    mus, vars_ = [], []
    for points, probs in task:
        mean, _, var = population_moments(points, probs)
        mus.append(mean)
        vars_.append(var)
    return mus, vars_


def gaussian_tau_predict(x: float, mus: list[float], vars_: list[float], tau: float, alpha0: float) -> int:
    pooled = float(np.mean(vars_))
    scale = pooled
    scores = []
    for mu, var in zip(mus, vars_):
        vv = (1.0 - tau) * pooled + tau * var + alpha0 * scale
        if vv <= 0:
            raise ValueError("non-positive variance")
        score = -0.5 * (((x - mu) ** 2) / vv + math.log(vv)) + math.log(0.5)
        scores.append(score)
    return int(np.argmax(scores))


def balanced_accuracy(task, tau: float, alpha0: float) -> float:
    mus, vars_ = _shared_and_class_vars(task)
    per_class = []
    for c, (points, probs) in enumerate(task):
        correct = np.array([gaussian_tau_predict(float(x), mus, vars_, float(tau), alpha0) == c for x in points])
        per_class.append(float(np.sum(probs * correct.astype(np.float64))))
    return float(np.mean(per_class))


def risk_curves(cfg: WitnessConfig, taus: list[float]) -> dict[str, np.ndarray]:
    tasks = task_distributions(cfg)
    return {
        name: np.array([balanced_accuracy(task, float(t), cfg.alpha0) for t in taus], dtype=np.float64)
        for name, task in tasks.items()
    }


def deterministic_minimax_regret(curves: dict[str, np.ndarray]) -> float:
    a, b = curves["A"], curves["B"]
    ra = float(a.max()) - a
    rb = float(b.max()) - b
    return float(np.min(np.maximum(ra, rb)))


def randomized_minimax_regret(curves: dict[str, np.ndarray]) -> dict:
    """Exact two-task finite-action minimax over randomized selectors.

    A two-constraint LP has an optimum supported on at most two actions.
    """
    a, b = curves["A"], curves["B"]
    ra = float(a.max()) - a
    rb = float(b.max()) - b
    m = len(ra)
    best = (float("inf"), None)
    for i in range(m):
        val = max(float(ra[i]), float(rb[i]))
        if val < best[0]:
            best = (val, {"support": [i], "prob": [1.0]})
    for i in range(m):
        for j in range(i + 1, m):
            den = (ra[i] - ra[j]) - (rb[i] - rb[j])
            if abs(float(den)) <= 1e-15:
                continue
            p = float((rb[j] - ra[j]) / den)
            if -1e-12 <= p <= 1.0 + 1e-12:
                p = min(1.0, max(0.0, p))
                ea = p * float(ra[i]) + (1 - p) * float(ra[j])
                eb = p * float(rb[i]) + (1 - p) * float(rb[j])
                val = max(ea, eb)
                if val < best[0]:
                    best = (val, {"support": [i, j], "prob": [p, 1.0 - p], "regret_A": ea, "regret_B": eb})
    return {"value": float(best[0]), **best[1]}


def epsilon_sensitivity(epsilons: list[float], taus: list[float], alpha0: float = 0.10) -> list[dict]:
    rows = []
    for eps in epsilons:
        cfg = WitnessConfig(epsilon=float(eps), alpha0=float(alpha0))
        curves = risk_curves(cfg, taus)
        rows.append({
            "epsilon": float(eps),
            "deterministic_minimax_regret": deterministic_minimax_regret(curves),
            "randomized_minimax_regret": randomized_minimax_regret(curves)["value"],
            "best_A": float(curves["A"].max()),
            "best_B": float(curves["B"].max()),
        })
    return rows


def third_order_escape(cfg: WitnessConfig) -> dict:
    tasks = task_distributions(cfg)
    out = {}
    for name, task in tasks.items():
        m3 = [third_central_moment(points, probs) for points, probs in task]
        out[name] = m3
    task_identifiable = (
        out["A"][0] > 0 and out["A"][1] < 0 and
        out["B"][0] < 0 and out["B"][1] > 0
    )
    return {"third_central_moments": out, "task_identifiable_by_sign_pattern": bool(task_identifiable)}


def arbitrary_dimension_summary(cfg: WitnessConfig, d: int) -> dict:
    if d < 1:
        raise ValueError("d must be >=1")
    tasks = task_distributions(cfg)
    summaries = {}
    for name, task in tasks.items():
        cls = []
        for c, (pts, probs) in enumerate(task):
            mu1, _, var1 = population_moments(pts, probs)
            mu = np.zeros(d)
            mu[0] = mu1
            cov = np.eye(d)
            cov[0, 0] = var1
            cls.append({"mean": mu, "cov": cov})
        summaries[name] = cls
    max_abs = 0.0
    for c in range(2):
        max_abs = max(max_abs, float(np.max(np.abs(summaries["A"][c]["mean"] - summaries["B"][c]["mean"]))))
        max_abs = max(max_abs, float(np.max(np.abs(summaries["A"][c]["cov"] - summaries["B"][c]["cov"]))))
    return {"d": int(d), "max_summary_difference": max_abs, "full_rank_nuisance_extension": True}
