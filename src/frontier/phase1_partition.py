from __future__ import annotations

import hashlib
import numpy as np


def _stable_seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**32 - 1)


def dirichlet_label_partition(y: np.ndarray, n_clients: int, alpha: float, seed: int, bank_id: str = "") -> np.ndarray:
    """Return client id per record. Every record is assigned exactly once.

    The partition is only a one-shot object-construction audit; full participation makes
    the aggregated sufficient statistics partition invariant.
    """
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(_stable_seed("phase1_partition", seed, bank_id))
    client = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = idx[rng.permutation(len(idx))]
        probs = rng.dirichlet(np.full(n_clients, float(alpha)))
        counts = rng.multinomial(len(idx), probs)
        off = 0
        for k, n in enumerate(counts):
            if n:
                client[idx[off:off+n]] = k
            off += n
        if off != len(idx):
            raise RuntimeError("Partition count mismatch")
    if np.any(client < 0) or len(client) != len(y):
        raise RuntimeError("Incomplete client assignment")
    return client
