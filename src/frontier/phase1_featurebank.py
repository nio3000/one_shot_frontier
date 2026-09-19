from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def collate_medmnist(batch):
    """Windows-spawn-safe MedMNIST collate function.

    Must remain module-level so torch DataLoader workers can pickle/import it
    under the Windows ``spawn`` multiprocessing start method.
    """
    import torch

    xs, ys = zip(*batch)
    xs = torch.stack(list(xs))
    labels = np.asarray(
        [np.asarray(y).reshape(-1)[0] for y in ys],
        dtype=np.int64,
    )
    ys = torch.as_tensor(labels, dtype=torch.long)
    return xs, ys


@dataclass(frozen=True)
class FeatureBank:
    bank_id: str
    dataset_id: str
    modality: str
    encoder_id: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any]


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_feature_bank(path: str | Path, metadata: dict[str, Any] | None = None) -> FeatureBank:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        required = {"X_train", "y_train", "X_test", "y_test"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"Feature bank {path} missing arrays: {sorted(missing)}")
        Xtr = np.asarray(z["X_train"], dtype=np.float32)
        ytr = np.asarray(z["y_train"], dtype=np.int64).reshape(-1)
        Xte = np.asarray(z["X_test"], dtype=np.float32)
        yte = np.asarray(z["y_test"], dtype=np.int64).reshape(-1)
        if "metadata_json" in z.files:
            raw = z["metadata_json"]
            if np.ndim(raw) == 0:
                raw = raw.item()
            embedded = json.loads(str(raw))
        else:
            embedded = {}
    meta = dict(embedded)
    if metadata:
        meta.update(metadata)
    if Xtr.ndim != 2 or Xte.ndim != 2:
        raise ValueError("Feature arrays must be 2-D")
    if Xtr.shape[1] != Xte.shape[1]:
        raise ValueError("Train/test feature dimensions differ")
    if len(Xtr) != len(ytr) or len(Xte) != len(yte):
        raise ValueError("Feature/label row count mismatch")
    if not np.isfinite(Xtr).all() or not np.isfinite(Xte).all():
        raise ValueError("Non-finite feature values")
    classes = np.unique(np.concatenate([ytr, yte]))
    if len(classes) < 2:
        raise ValueError("At least two classes are required")
    expected = np.arange(classes.max() + 1)
    if not np.array_equal(classes, expected):
        raise ValueError(f"Labels must be contiguous 0..C-1, got {classes.tolist()}")
    bank_id = str(meta.get("bank_id", path.stem))
    return FeatureBank(
        bank_id=bank_id,
        dataset_id=str(meta.get("dataset_id", "unknown")),
        modality=str(meta.get("modality", "unknown")),
        encoder_id=str(meta.get("encoder_id", "unknown")),
        X_train=Xtr,
        y_train=ytr,
        X_test=Xte,
        y_test=yte,
        metadata=meta,
    )


def validate_manifest(manifest: dict[str, Any], protocol: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    if manifest.get("freeze_type") != "PHASE1_FEATURE_BANK_MANIFEST":
        raise ValueError("Invalid Phase 1 feature-bank manifest type")
    if manifest.get("status") != "FROZEN":
        raise ValueError("Phase 1 feature-bank manifest is not FROZEN")
    required = {x["bank_id"]: x for x in protocol["feature_banks"]["required"]}
    entries = {x["bank_id"]: x for x in manifest.get("banks", [])}
    if set(entries) != set(required):
        raise ValueError(f"Manifest bank IDs differ from protocol. required={sorted(required)} actual={sorted(entries)}")
    validated: list[dict[str, Any]] = []
    for bank_id in sorted(required):
        e = dict(entries[bank_id])
        p = Path(e["path"])
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        actual = file_sha256(p)
        if actual != e.get("sha256"):
            raise ValueError(f"Hash mismatch for {bank_id}: {actual} != {e.get('sha256')}")
        for k in ("dataset_id", "modality", "encoder_id"):
            if str(e.get(k)) != str(required[bank_id][k]):
                raise ValueError(f"Manifest mismatch for {bank_id} field {k}")
        e["resolved_path"] = str(p)
        validated.append(e)
    return validated
