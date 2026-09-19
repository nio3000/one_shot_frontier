from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import pandas as pd

def expected_rel(row) -> str:
    return f"images/{row.experiment}/Plate{int(row.plate)}/{row.well}_s{int(row.site)}.png"

def normalize_member(s: str) -> str:
    s = s.replace("\\", "/").lstrip("./")
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--members-seen", required=True)
    ap.add_argument("--out", default="reports/phase5/validation/rxrx1_segmented_recovery_plan.json")
    a = ap.parse_args()

    md = pd.read_csv(a.metadata)
    expected = set(md.apply(expected_rel, axis=1))
    expected_by_exp = defaultdict(set)
    for p in expected:
        expected_by_exp[p.split("/")[1]].add(p)

    seen = {
        normalize_member(x)
        for x in Path(a.members_seen).read_text(encoding="utf-8").splitlines()
        if "/images/" in x.replace("\\", "/") or x.replace("\\", "/").lstrip("./").startswith("images/")
    }
    seen_images = {x for x in seen if x.lower().endswith(".png")}
    seen_expected = expected & seen_images

    rows = []
    for exp in sorted(expected_by_exp):
        exp_expected = expected_by_exp[exp]
        exp_seen = exp_expected & seen_expected
        rows.append({
            "experiment": exp,
            "expected_images": len(exp_expected),
            "seen_before_truncation": len(exp_seen),
            "missing_from_partial": len(exp_expected - exp_seen),
            "header_complete_candidate": exp_expected <= seen_expected,
        })

    result = {
        "metadata_rows": len(md),
        "expected_images": len(expected),
        "seen_expected_images_before_truncation": len(seen_expected),
        "missing_images": len(expected - seen_expected),
        "n_experiments": len(expected_by_exp),
        "header_complete_candidate_experiments": [r["experiment"] for r in rows if r["header_complete_candidate"]],
        "n_header_complete_candidate_experiments": sum(r["header_complete_candidate"] for r in rows),
        "experiments": rows,
        "note": "A complete tar header set is only a candidate. Final completeness is established from extracted readable files, not member headers alone."
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="experiments"}, indent=2))

if __name__ == "__main__":
    main()
