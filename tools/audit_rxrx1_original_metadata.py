
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

REQUIRED = [
    "cell_type","dataset","experiment","plate","well","site",
    "well_type","sirna","sirna_id"
]

EXPECTED = {
    "n_rows": 125510,
    "n_experiments": 51,
    "n_classes": 1139,
    "sirna_id_min": 0,
    "sirna_id_max": 1138,
    "sites": [1,2],
    "cell_types": ["HEPG2","HUVEC","RPE","U2OS"],
    "original_train_experiments": 33,
    "original_test_experiments": 18,
}

# These are the four public-test experiments of the original RxRx1 competition.
# They are recorded here only as a candidate WILDS OOD-validation mapping;
# the script does NOT promote them to authoritative WILDS metadata.
PUBLIC_TEST_CANDIDATE = ["HEPG2-08","HUVEC-17","RPE-08","U2OS-04"]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default="reports/phase5/validation/rxrx1_original_metadata_audit.json")
    a=ap.parse_args()
    p=Path(a.metadata)
    df=pd.read_csv(p)

    missing=[c for c in REQUIRED if c not in df.columns]
    exp_by_split={}
    if "dataset" in df.columns:
        for split,g in df.groupby("dataset"):
            exp_by_split[str(split)] = sorted(g["experiment"].astype(str).unique().tolist())

    observed = {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": missing,
        "n_experiments": int(df["experiment"].nunique()) if "experiment" in df else None,
        "n_classes": int(df["sirna_id"].nunique()) if "sirna_id" in df else None,
        "sirna_id_min": int(df["sirna_id"].min()) if "sirna_id" in df else None,
        "sirna_id_max": int(df["sirna_id"].max()) if "sirna_id" in df else None,
        "sites": sorted(map(int, df["site"].dropna().unique())) if "site" in df else None,
        "cell_types": sorted(map(str, df["cell_type"].dropna().unique())) if "cell_type" in df else None,
        "dataset_values": sorted(map(str, df["dataset"].dropna().unique())) if "dataset" in df else None,
        "experiments_by_original_split": exp_by_split,
        "rows_by_original_split": {str(k):int(v) for k,v in df["dataset"].value_counts().to_dict().items()} if "dataset" in df else None,
        "sites_per_experiment_min": int(df.groupby("experiment")["site"].nunique().min()) if {"experiment","site"} <= set(df.columns) else None,
        "sites_per_experiment_max": int(df.groupby("experiment")["site"].nunique().max()) if {"experiment","site"} <= set(df.columns) else None,
    }

    checks = {
        "required_columns": not missing,
        "n_rows_125510": observed["n_rows"] == EXPECTED["n_rows"],
        "n_experiments_51": observed["n_experiments"] == EXPECTED["n_experiments"],
        "sirna_classes_1139": observed["n_classes"] == EXPECTED["n_classes"],
        "sirna_id_range_0_1138": observed["sirna_id_min"] == 0 and observed["sirna_id_max"] == 1138,
        "sirna_id_contiguous": set(df["sirna_id"].astype(int).unique()) == set(range(1139)) if "sirna_id" in df else False,
        "sites_exact_1_2": observed["sites"] == [1,2],
        "cell_types_exact": observed["cell_types"] == EXPECTED["cell_types"],
        "original_split_values_train_test": observed["dataset_values"] == ["test","train"],
        "original_train_experiments_33": len(exp_by_split.get("train",[])) == 33,
        "original_test_experiments_18": len(exp_by_split.get("test",[])) == 18,
        "public_test_candidate_present": set(PUBLIC_TEST_CANDIDATE).issubset(set(exp_by_split.get("test",[]))),
    }

    result = {
        "metadata": str(p),
        "metadata_sha256": sha256(p),
        "expected_authority_facts": EXPECTED,
        "observed": observed,
        "checks": checks,
        "all_core_original_metadata_checks_pass": all(v for k,v in checks.items() if k!="public_test_candidate_present"),
        "candidate_wilds_val_experiments_not_yet_promoted": PUBLIC_TEST_CANDIDATE,
        "governance_note": "This audit qualifies the original RxRx1 metadata universe. It does not by itself establish exact WILDS train/val/test labels."
    }
    out=Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__=="__main__":
    main()
