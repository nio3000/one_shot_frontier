from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

def key(df):
    for c in ["experiment","plate","well","site"]:
        if c not in df.columns:
            raise ValueError(f"missing {c}")
    return (df.experiment.astype(str)+"|"+df.plate.astype(str)+"|"+
            df.well.astype(str)+"|"+df.site.astype(str))

def reconstruct(original_root: Path, row):
    chans=[]
    for ch in (1,2,3):
        p = original_root/"images"/str(row.experiment)/f"Plate{row.plate}"/f"{row.well}_s{row.site}_w{ch}.png"
        im = np.asarray(Image.open(p))
        if im.ndim != 2:
            raise ValueError(f"expected grayscale channel: {p}")
        h,w = im.shape
        y0=(h-256)//2; x0=(w-256)//2
        chans.append(im[y0:y0+256, x0:x0+256])
    return np.stack(chans, axis=-1).astype(np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wilds-metadata", required=True)
    ap.add_argument("--original-metadata", required=True)
    ap.add_argument("--original-image-root")
    ap.add_argument("--wilds-sample-root")
    ap.add_argument("--out", default="reports/phase5/validation/rxrx1_reconstruction_feasibility.json")
    a=ap.parse_args()

    w=pd.read_csv(a.wilds_metadata)
    o=pd.read_csv(a.original_metadata)
    wk, ok = key(w), key(o)
    ws, os_ = set(wk), set(ok)
    report = {
        "wilds_rows": len(w),
        "original_rows": len(o),
        "site_key_exact_set_match": ws == os_,
        "wilds_only_count": len(ws-os_),
        "original_only_count": len(os_-ws),
    }
    common = sorted(ws & os_)
    if "sirna_id" in w.columns and "sirna_id" in o.columns and common:
        wm=w.assign(_k=wk).set_index("_k")["sirna_id"].loc[common].astype(int).to_numpy()
        om=o.assign(_k=ok).set_index("_k")["sirna_id"].loc[common].astype(int).to_numpy()
        report["sirna_id_exact_on_common"] = bool(np.array_equal(wm, om))

    pix=[]
    if a.original_image_root and a.wilds_sample_root:
        by_name={}
        for i,row in w.iterrows():
            name=f"{row.well}_s{row.site}.png"
            by_name.setdefault(name, []).append(i)
        for sp in sorted(Path(a.wilds_sample_root).glob("*.png")):
            cand=by_name.get(sp.name, [])
            for i in cand:
                row=w.iloc[i]
                p=Path(a.original_image_root)/"images"/str(row.experiment)/f"Plate{row.plate}"/f"{row.well}_s{row.site}_w1.png"
                if not p.exists():
                    continue
                rec=reconstruct(Path(a.original_image_root), row)
                act=np.asarray(Image.open(sp).convert("RGB"))
                pix.append({
                    "sample": sp.name,
                    "experiment": str(row.experiment),
                    "exact_equal": bool(rec.shape == act.shape and np.array_equal(rec, act)),
                    "max_abs_diff": int(np.max(np.abs(rec.astype(np.int16)-act.astype(np.int16)))) if rec.shape == act.shape else None,
                })
                break

    report["pixel_equivalence_samples"]=pix
    report["pixel_exact_all_tested"]=bool(pix) and all(x["exact_equal"] for x in pix)
    report["promotion_rule"]="No reconstructed RxRx1-WILDS may enter the frozen feature manifest without exact metadata identity plus exact sampled pixel equivalence."
    out=Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
