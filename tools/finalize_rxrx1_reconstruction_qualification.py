from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--staging-audit", required=True)
    ap.add_argument("--pixel-audit", required=True)
    ap.add_argument("--out-json", default="reports/phase5/validation/rxrx1_reconstruction_qualification_final.json")
    ap.add_argument("--out-md", default="reports/phase5/validation/RXRX1_RECONSTRUCTION_QUALIFICATION_FINAL.md")
    a=ap.parse_args()

    md=Path(a.metadata)
    st=json.loads(Path(a.staging_audit).read_text(encoding="utf-8"))
    px=json.loads(Path(a.pixel_audit).read_text(encoding="utf-8"))

    gates={
        "authority_metadata_rows_125510": st.get("expected_images")==125510,
        "staging_exactly_complete": bool(st.get("full_complete")),
        "pixel_equivalence_128_exact": px.get("n_samples")==128 and bool(px.get("pixel_exact_all")),
        "pixel_max_abs_diff_zero": px.get("max_abs_diff_overall")==0,
        "pixel_metadata_unique_all": bool(px.get("metadata_unique_all")),
        "pixel_coverage_18_or_more_experiments": int(px.get("n_unique_experiments",0))>=18,
        "pixel_both_sites": px.get("sites")==[1,2],
    }
    decision="RXRX1_WILDS_DETERMINISTIC_RECONSTRUCTION_QUALIFIED" if all(gates.values()) else "RXRX1_RECONSTRUCTION_NOT_YET_QUALIFIED"

    result={
        "decision":decision,
        "gates":gates,
        "metadata_path":str(md),
        "metadata_sha256":sha256(md),
        "staging_audit":str(a.staging_audit),
        "pixel_audit":str(a.pixel_audit),
        "provenance_note":"Images are reconstructed/completed from official RxRx1/WILDS authorities; this is a provenance-qualified reconstruction, not a claim that the original full WILDS archive was downloaded intact."
    }
    outj=Path(a.out_json); outj.parent.mkdir(parents=True,exist_ok=True)
    outj.write_text(json.dumps(result,indent=2),encoding="utf-8")

    lines=[
        "# RxRx1-WILDS deterministic reconstruction qualification",
        "",
        f"**Decision:** `{decision}`",
        "",
        "## Gates",
        "",
    ]+[f"- `{k}` = **{'PASS' if v else 'FAIL'}**" for k,v in gates.items()]+[
        "",
        "## Authority",
        "",
        f"- metadata: `{md}`",
        f"- metadata SHA256: `{result['metadata_sha256']}`",
        f"- staging audit: `{a.staging_audit}`",
        f"- pixel audit: `{a.pixel_audit}`",
        "",
        "## Provenance boundary",
        "",
        result["provenance_note"],
    ]
    Path(a.out_md).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
