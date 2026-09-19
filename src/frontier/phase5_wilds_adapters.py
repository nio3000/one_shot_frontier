from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_SPECS = {
    "camelyon17": {
        "version": "1.0",
        "folder": "camelyon17_v1.0",
        "label_field": "tumor",
        "domain_field": "center",
        "group_field": "slide",
    },
    "rxrx1": {
        "version": "1.0",
        "folder": "rxrx1_v1.0",
        "label_field": "sirna_id",
        "domain_field": "cell_type",
        "group_field": "experiment",
    },
    "iwildcam": {
        "version": "2.0",
        "folder": "iwildcam_v2.0",
        "label_field": "y",
        "domain_field": "location_remapped",
        "group_field": "sequence_remapped",
    },
}


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _factorize_stable(values) -> tuple[np.ndarray, list[str]]:
    vals = np.asarray([str(x) for x in values], dtype=object)
    levels = sorted(set(vals.tolist()))
    mp = {v: i for i, v in enumerate(levels)}
    return np.asarray([mp[v] for v in vals], dtype=np.int64), levels


def _official_split_camelyon(df: pd.DataFrame) -> np.ndarray:
    # WILDS v1.0 explicitly sets center 1=OOD val, center 2=OOD test;
    # remaining records retain source/id split coding from metadata.csv.
    if "split" in df.columns:
        raw = df["split"].astype(str)
    else:
        raw = pd.Series(["unknown"] * len(df))
    vals = []
    for center, s in zip(df["center"].astype(int), raw):
        if center == 1:
            vals.append("val")
        elif center == 2:
            vals.append("test")
        else:
            vals.append(str(s))
    out, _ = _factorize_stable(vals)
    return out


def _official_split_rxrx1(df: pd.DataFrame) -> np.ndarray:
    vals = []
    for ds, site in zip(df["dataset"].astype(str), df["site"].astype(int)):
        if ds == "train" and site == 2:
            vals.append("id_test")
        else:
            vals.append(ds)
    out, _ = _factorize_stable(vals)
    return out


def _official_split_iwildcam(df: pd.DataFrame) -> np.ndarray:
    out, _ = _factorize_stable(df["split"].astype(str))
    return out


def read_normalized_metadata(dataset_id: str, root: str | Path) -> tuple[pd.DataFrame, dict]:
    if dataset_id not in DATASET_SPECS:
        raise ValueError(dataset_id)
    spec = DATASET_SPECS[dataset_id]
    droot = Path(root) / spec["folder"]
    meta_path = droot / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    df = pd.read_csv(meta_path)

    if dataset_id == "camelyon17":
        required = {"patient","node","x_coord","y_coord","tumor","center","slide"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Camelyon metadata missing {sorted(missing)}")
        rel = [
            f"patches/patient_{str(patient).zfill(3) if str(patient).isdigit() else patient}_node_{node}/"
            f"patch_patient_{str(patient).zfill(3) if str(patient).isdigit() else patient}_node_{node}_x_{x}_y_{y}.png"
            for patient,node,x,y in df[["patient","node","x_coord","y_coord"]].itertuples(index=False,name=None)
        ]
        split = _official_split_camelyon(df)
        extra = {"public_schema_domain": "hospital(center)", "public_schema_group": "slide"}
    elif dataset_id == "rxrx1":
        required = {"experiment","plate","well","site","sirna_id","cell_type","dataset"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"RxRx1 metadata missing {sorted(missing)}")
        rel = [
            f"images/{exp}/Plate{plate}/{well}_s{site}.png"
            for exp,plate,well,site in df[["experiment","plate","well","site"]].itertuples(index=False,name=None)
        ]
        split = _official_split_rxrx1(df)
        extra = {
            "public_shift_domain": "experiment(batch)",
            "phase5_primary_context": "cell_type",
            "phase5_leakage_group": "experiment",
        }
    else:
        required = {"filename","y","location_remapped","sequence_remapped","split"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"iWildCam metadata missing {sorted(missing)}")
        rel = [f"train/{x}" for x in df["filename"].astype(str)]
        split = _official_split_iwildcam(df)
        extra = {"public_schema_domain": "location", "public_schema_group": "sequence"}

    domain_raw = df[spec["domain_field"]].astype(str)
    group_raw = df[spec["group_field"]].astype(str)
    domain_id, domain_levels = _factorize_stable(domain_raw)
    group_id, group_levels = _factorize_stable(group_raw)

    out = pd.DataFrame({
        "source_row_index": np.arange(len(df), dtype=np.int64),
        "sample_path": [str(droot / p) for p in rel],
        "y": df[spec["label_field"]].astype(int).to_numpy(),
        "domain_id": domain_id,
        "group_id": group_id,
        "official_split_id": split,
    })
    meta = {
        "dataset_id": dataset_id,
        "dataset_version": spec["version"],
        "dataset_root": str(droot),
        "metadata_path": str(meta_path),
        "metadata_sha256": file_sha256(meta_path),
        "domain_levels": domain_levels,
        "group_level_count": len(group_levels),
        "n_rows": int(len(out)),
        **extra,
    }
    return out, meta
