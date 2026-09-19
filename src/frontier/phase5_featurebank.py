from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Tier1FeatureBank:
    bank_id: str
    dataset_id: str
    encoder_id: str
    X: np.ndarray
    y: np.ndarray
    domain_id: np.ndarray
    group_id: np.ndarray
    official_split_id: np.ndarray
    metadata: dict[str, Any]


def load_tier1_feature_bank(path: str | Path, metadata: dict[str, Any] | None = None) -> Tier1FeatureBank:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        required = {"X", "y", "domain_id", "group_id", "official_split_id", "metadata_json"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.int64).reshape(-1)
        domain = np.asarray(z["domain_id"], dtype=np.int64).reshape(-1)
        group = np.asarray(z["group_id"], dtype=np.int64).reshape(-1)
        split = np.asarray(z["official_split_id"], dtype=np.int64).reshape(-1)
        raw = z["metadata_json"]
        if np.ndim(raw) == 0:
            raw = raw.item()
        embedded = json.loads(str(raw))
    meta = dict(embedded)
    if metadata:
        meta.update(metadata)
    n = len(y)
    if X.ndim != 2 or len(X) != n or len(domain) != n or len(group) != n or len(split) != n:
        raise ValueError("Feature bank row mismatch")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite features")
    return Tier1FeatureBank(
        bank_id=str(meta["bank_id"]),
        dataset_id=str(meta["dataset_id"]),
        encoder_id=str(meta["encoder_id"]),
        X=X, y=y, domain_id=domain, group_id=group, official_split_id=split, metadata=meta
    )
