from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from PIL import Image

def expected_rel(row) -> str:
    return f"images/{row.experiment}/Plate{int(row.plate)}/{row.well}_s{int(row.site)}.png"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--verify-images", action="store_true")
    ap.add_argument("--out", default="reports/phase5/validation/rxrx1_staging_audit.json")
    a = ap.parse_args()

    root = Path(a.staging_root)
    md = pd.read_csv(a.metadata)
    expected = set(md.apply(expected_rel, axis=1))
    actual = {
        p.relative_to(root).as_posix()
        for p in (root/"images").rglob("*.png")
        if p.is_file()
    } if (root/"images").exists() else set()

    unreadable = []
    if a.verify_images:
        for rel in sorted(actual & expected):
            try:
                with Image.open(root/rel) as im:
                    im.verify()
            except Exception as e:
                unreadable.append({"path": rel, "error": repr(e)})

    by_exp_expected = Counter(p.split("/")[1] for p in expected)
    by_exp_actual = Counter(p.split("/")[1] for p in actual & expected)

    exp_rows = []
    for exp in sorted(by_exp_expected):
        exp_rows.append({
            "experiment": exp,
            "expected": by_exp_expected[exp],
            "present": by_exp_actual[exp],
            "missing": by_exp_expected[exp] - by_exp_actual[exp],
            "complete": by_exp_expected[exp] == by_exp_actual[exp],
        })

    result = {
        "staging_root": str(root),
        "expected_images": len(expected),
        "present_expected_images": len(actual & expected),
        "missing_images": len(expected - actual),
        "extra_images": len(actual - expected),
        "unreadable_images": len(unreadable),
        "complete_experiments": [x["experiment"] for x in exp_rows if x["complete"]],
        "n_complete_experiments": sum(x["complete"] for x in exp_rows),
        "incomplete_experiments": [x["experiment"] for x in exp_rows if not x["complete"]],
        "n_incomplete_experiments": sum(not x["complete"] for x in exp_rows),
        "experiments": exp_rows,
        "unreadable": unreadable,
        "full_complete": len(expected-actual)==0 and len(actual-expected)==0 and len(unreadable)==0,
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("experiments","unreadable")}, indent=2))

if __name__ == "__main__":
    main()
