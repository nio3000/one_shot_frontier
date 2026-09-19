from __future__ import annotations

import hashlib
import math

import numpy as np


def _u01(*parts: object) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)


def deterministic_group_roles(dataset_id: str, domain_id: np.ndarray, group_id: np.ndarray, seed: int, fit_fraction: float) -> np.ndarray:
    domain_id = np.asarray(domain_id, dtype=np.int64)
    group_id = np.asarray(group_id, dtype=np.int64)
    role = np.full(len(domain_id), -1, dtype=np.int8)
    mapping = {}
    for d, g in sorted(set(zip(domain_id.tolist(), group_id.tolist()))):
        mapping[(d,g)] = 0 if _u01("phase5_group_split", dataset_id, int(d), int(g), int(seed)) < float(fit_fraction) else 1
    for i, key in enumerate(zip(domain_id.tolist(), group_id.tolist())):
        role[i] = mapping[key]
    return role


def no_group_leakage(domain_id: np.ndarray, group_id: np.ndarray, role: np.ndarray) -> bool:
    seen = {}
    for d,g,r in zip(map(int,domain_id), map(int,group_id), map(int,role)):
        key=(d,g)
        if key in seen and seen[key] != r:
            return False
        seen[key]=r
    return True


def class_counts(y: np.ndarray, mask: np.ndarray) -> dict[int,int]:
    vals, cnt = np.unique(np.asarray(y, dtype=np.int64)[mask], return_counts=True)
    return {int(v): int(n) for v,n in zip(vals,cnt)}


def deterministic_class_subset(dataset_id: str, pair_id: str, classes: list[int], max_classes: int, seed: int) -> list[int]:
    if len(classes) <= int(max_classes):
        return sorted(map(int, classes))
    scored=[]
    for c in classes:
        h=hashlib.sha256(f"phase5_class_subset|{dataset_id}|{pair_id}|{int(c)}|{int(seed)}".encode()).hexdigest()
        scored.append((h,int(c)))
    return sorted(c for _,c in sorted(scored)[:int(max_classes)])


def selected_pair_count(n_eligible: int, near_fraction: float, min_selected: int, max_selected: int) -> int:
    if n_eligible <= 0:
        return 0
    return min(int(max_selected), max(int(min_selected), int(math.ceil(float(near_fraction)*n_eligible))))
