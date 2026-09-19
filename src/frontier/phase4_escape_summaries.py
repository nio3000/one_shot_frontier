from __future__ import annotations

import hashlib
import numpy as np


def _stable_seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**32 - 1)


def rademacher_directions(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed("phase4_rademacher", int(d), int(k), int(seed)))
    R = rng.choice(np.array([-1.0, 1.0]), size=(int(d), int(k)))
    R /= np.sqrt(float(d))
    return np.asarray(R, dtype=np.float64)


def _raw_power_sums(X: np.ndarray, y: np.ndarray, C: int, R: np.ndarray, max_power: int) -> list[np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    R = np.asarray(R, dtype=np.float64)
    out = [np.zeros((int(C), R.shape[1]), dtype=np.float64) for _ in range(max_power + 1)]
    for c in range(int(C)):
        Z = X[y == c] @ R
        n = len(Z)
        out[0][c].fill(float(n))
        if n == 0:
            continue
        for p in range(1, max_power + 1):
            out[p][c] = np.sum(Z ** p, axis=0, dtype=np.float64)
    return out


def _central_standardized_from_raw(raw: list[np.ndarray], power: int, eps: float = 1e-12) -> np.ndarray:
    n = np.maximum(raw[0], 1.0)
    m1 = raw[1] / n
    m2 = raw[2] / n
    var = np.maximum(m2 - m1 * m1, eps)
    if power == 3:
        m3 = raw[3] / n
        central = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        return central / np.power(var, 1.5)
    if power == 4:
        m3 = raw[3] / n
        m4 = raw[4] / n
        central = m4 - 4.0 * m1 * m3 + 6.0 * (m1 ** 2) * m2 - 3.0 * (m1 ** 4)
        return central / (var ** 2)
    raise ValueError("power must be 3 or 4")


def central_standardized_moment(X: np.ndarray, y: np.ndarray, C: int, R: np.ndarray, power: int) -> np.ndarray:
    raw = _raw_power_sums(X, y, C, R, power)
    return _central_standardized_from_raw(raw, power)


def diagonal_third_summary(X: np.ndarray, y: np.ndarray, C: int) -> np.ndarray:
    d = np.asarray(X).shape[1]
    return central_standardized_moment(X, y, C, np.eye(d, dtype=np.float64), 3)


def rademacher_third_summary(X: np.ndarray, y: np.ndarray, C: int, k: int, seed: int) -> np.ndarray:
    R = rademacher_directions(np.asarray(X).shape[1], int(k), int(seed))
    return central_standardized_moment(X, y, C, R, 3)


def rademacher_fourth_summary(X: np.ndarray, y: np.ndarray, C: int, k: int, seed: int) -> np.ndarray:
    R = rademacher_directions(np.asarray(X).shape[1], int(k), int(seed))
    return central_standardized_moment(X, y, C, R, 4)


def federated_raw_power_sum(
    X: np.ndarray,
    y: np.ndarray,
    client_ids: np.ndarray,
    n_clients: int,
    C: int,
    R: np.ndarray,
    power: int,
) -> tuple[np.ndarray, int]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    client_ids = np.asarray(client_ids, dtype=np.int64).reshape(-1)
    R = np.asarray(R, dtype=np.float64)
    agg = np.zeros((int(C), R.shape[1]), dtype=np.float64)
    present_pairs = 0
    for k_client in range(int(n_clients)):
        mk = client_ids == k_client
        for c in range(int(C)):
            xc = X[mk & (y == c)]
            if not len(xc):
                continue
            present_pairs += 1
            z = xc @ R
            agg[c] += np.sum(z ** int(power), axis=0, dtype=np.float64)
    return agg, int(present_pairs)


def central_raw_power_sum(X: np.ndarray, y: np.ndarray, C: int, R: np.ndarray, power: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    R = np.asarray(R, dtype=np.float64)
    out = np.zeros((int(C), R.shape[1]), dtype=np.float64)
    for c in range(int(C)):
        z = X[y == c] @ R
        if len(z):
            out[c] = np.sum(z ** int(power), axis=0, dtype=np.float64)
    return out


def normalized_summary_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.sqrt(a.size), 1.0))


def normalized_raw_aggregation_error(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b)) / denom)
