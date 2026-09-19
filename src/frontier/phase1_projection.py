from __future__ import annotations

import hashlib
import numpy as np


def stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32 - 1)


def orthoproject_matrix(input_dim: int, output_dim: int, encoder_id: str, seed: int) -> np.ndarray:
    if output_dim > input_dim:
        raise ValueError(f"output_dim={output_dim} exceeds input_dim={input_dim}")
    rng = np.random.default_rng(stable_seed("phase1_projection", encoder_id, input_dim, output_dim, seed))
    g = rng.standard_normal((input_dim, output_dim))
    q, r = np.linalg.qr(g, mode="reduced")
    # Deterministic sign convention removes QR sign ambiguity.
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    return np.asarray(q, dtype=np.float64)


def project_features(X: np.ndarray, output_dim: int, encoder_id: str, seed: int) -> np.ndarray:
    P = orthoproject_matrix(X.shape[1], output_dim, encoder_id, seed)
    return np.asarray(X, dtype=np.float64) @ P
