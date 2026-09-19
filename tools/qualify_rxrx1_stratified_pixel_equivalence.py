
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

def parse_rel(rel):
    p=Path(rel)
    exp=p.parts[-3]
    plate=int(p.parts[-2].replace("Plate",""))
    well,site=p.stem.rsplit("_s",1)
    return exp,plate,well,int(site)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--recovery-report", required=True)
    ap.add_argument("--wilds-sample-root", required=True)
    ap.add_argument("--original-channel-root", required=True)
    ap.add_argument("--original-metadata", required=True)
    ap.add_argument("--out", default="reports/phase5/validation/rxrx1_stratified_pixel_equivalence.json")
    a=ap.parse_args()

    rep=json.loads(Path(a.recovery_report).read_text(encoding="utf-8"))
    md=pd.read_csv(a.original_metadata)
    rows=[]
    for s in rep["extracted"]:
        rel=s["relative_sample_path"]
        exp,plate,well,site=parse_rel(rel)
        hit=md[(md.experiment.astype(str)==exp)&(md.plate.astype(int)==plate)&(md.well.astype(str)==well)&(md.site.astype(int)==site)]
        if len(hit)!=1:
            rows.append({"sample":rel,"metadata_match_count":len(hit),"exact_equal":False})
            continue
        chans=[]
        for ch in (1,2,3):
            p=Path(a.original_channel_root)/Path(rel).parent/f"{Path(rel).stem}_w{ch}.png"
            arr=np.asarray(Image.open(p))
            h,w=arr.shape[:2]
            y0=(h-256)//2; x0=(w-256)//2
            chans.append(arr[y0:y0+256,x0:x0+256])
        recon=np.stack(chans,axis=-1).astype(np.uint8)
        actual=np.asarray(Image.open(Path(a.wilds_sample_root)/rel).convert("RGB"))
        eq=bool(recon.shape==actual.shape and np.array_equal(recon,actual))
        rows.append({
            "sample":rel,"experiment":exp,"plate":plate,"site":site,
            "metadata_match_count":1,
            "sirna_id":int(hit.iloc[0].sirna_id),
            "exact_equal":eq,
            "max_abs_diff":int(np.max(np.abs(recon.astype(np.int16)-actual.astype(np.int16)))) if recon.shape==actual.shape else None
        })

    res={
        "n_samples":len(rows),
        "n_unique_experiments":len({r.get("experiment") for r in rows if r.get("experiment")}),
        "n_unique_experiment_plate":len({(r.get("experiment"),r.get("plate")) for r in rows if r.get("experiment")}),
        "sites":sorted({r.get("site") for r in rows if r.get("site") is not None}),
        "metadata_unique_all":bool(rows) and all(r["metadata_match_count"]==1 for r in rows),
        "pixel_exact_all":bool(rows) and all(r["exact_equal"] for r in rows),
        "max_abs_diff_overall":max((r["max_abs_diff"] for r in rows if r.get("max_abs_diff") is not None),default=None),
        "rows":rows,
        "governance_note":"PASS qualifies the observed partial-archive coverage only; it does not prove unseen experiments."
    }
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(res,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in res.items() if k!="rows"},indent=2))

if __name__=="__main__":
    main()
