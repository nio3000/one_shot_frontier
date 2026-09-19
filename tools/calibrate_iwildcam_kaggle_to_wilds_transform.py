from __future__ import annotations
import argparse, io, json, math, statistics, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin

RESAMPLES={
    "NEAREST": Image.Resampling.NEAREST,
    "BOX": Image.Resampling.BOX,
    "BILINEAR": Image.Resampling.BILINEAR,
    "HAMMING": Image.Resampling.HAMMING,
    "BICUBIC": Image.Resampling.BICUBIC,
    "LANCZOS": Image.Resampling.LANCZOS,
}

def decode(b:bytes):
    with Image.open(io.BytesIO(b)) as im:
        im.load()
        return np.asarray(im.convert("RGB"),dtype=np.int16), im.size, im.mode

def sample_even(items,n):
    if n>=len(items): return items
    idx=np.linspace(0,len(items)-1,n,dtype=int)
    return [items[i] for i in idx]

def save_candidate(img, size, resample, target_im):
    x=img.convert("RGB").resize(size,resample=resample)
    buf=io.BytesIO()
    kwargs={}
    q=getattr(target_im,"quantization",None)
    if q:
        kwargs["qtables"]={int(k):list(v) for k,v in q.items()}
    try:
        s=JpegImagePlugin.get_sampling(target_im)
        if s in (0,1,2):
            kwargs["subsampling"]=s
    except Exception:
        pass
    x.save(buf,format="JPEG",**kwargs)
    return buf.getvalue()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--codalab-root", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="reports/phase5/validation/iwildcam_transform_calibration.json")
    a=ap.parse_args()

    md=pd.read_csv(a.metadata)
    train=Path(a.codalab_root)/"train"
    overlap=sorted(fn for fn in md.filename.astype(str) if (train/fn).is_file())
    chosen=sample_even(overlap,a.n)
    wanted=set(chosen)

    zmap={}
    with zipfile.ZipFile(a.zip) as z:
        for info in z.infolist():
            b=PurePosixPath(info.filename.replace("\\","/")).name
            if b in wanted:
                if b in zmap: raise RuntimeError(f"duplicate basename in zip: {b}")
                zmap[b]=info.filename
        miss=wanted-set(zmap)
        if miss: raise RuntimeError(f"missing in Kaggle ZIP: {len(miss)}")

        stats={k:{"exact":0,"mae":[],"max_abs":[]} for k in RESAMPLES}
        dims=[]
        for fn in chosen:
            src_bytes=z.read(zmap[fn])
            tgt_path=train/fn
            with Image.open(io.BytesIO(src_bytes)) as src, Image.open(tgt_path) as tgt:
                src.load(); tgt.load()
                sw,sh=src.size; tw,th=tgt.size
                dims.append({
                    "filename":fn,
                    "source":[sw,sh],
                    "target":[tw,th],
                    "floor_w":math.floor(sw*448/sh),
                    "round_w":round(sw*448/sh),
                    "ceil_w":math.ceil(sw*448/sh),
                })
                targ=np.asarray(tgt.convert("RGB"),dtype=np.int16)
                for name,resample in RESAMPLES.items():
                    cand_bytes=save_candidate(src,(tw,th),resample,tgt)
                    cand,_,_=decode(cand_bytes)
                    d=np.abs(cand-targ)
                    exact=bool(np.array_equal(cand,targ))
                    stats[name]["exact"]+=int(exact)
                    stats[name]["mae"].append(float(d.mean()))
                    stats[name]["max_abs"].append(int(d.max()))

    summary={}
    for name,s in stats.items():
        summary[name]={
            "exact":s["exact"],
            "n":len(chosen),
            "mean_mae":float(statistics.mean(s["mae"])),
            "median_mae":float(statistics.median(s["mae"])),
            "max_abs_overall":max(s["max_abs"]),
        }

    width_rule_counts={
        "floor":sum(d["target"][0]==d["floor_w"] for d in dims),
        "round":sum(d["target"][0]==d["round_w"] for d in dims),
        "ceil":sum(d["target"][0]==d["ceil_w"] for d in dims),
    }
    ranking=sorted(summary.items(),key=lambda kv:(-kv[1]["exact"],kv[1]["mean_mae"]))
    result={
        "n_calibration":len(chosen),
        "width_rule_counts":width_rule_counts,
        "resample_ranking":ranking,
        "best":ranking[0],
        "dimensions":dims,
        "note":"Candidate JPEGs reuse each target image's quantization tables and sampling only for calibration. Promotion requires a constant/generalizable JPEG signature plus full-overlap validation.",
    }
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
