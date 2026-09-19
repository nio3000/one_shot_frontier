from __future__ import annotations
import argparse, io, json, math, zipfile
from pathlib import Path, PurePosixPath
import numpy as np, pandas as pd
from PIL import Image, JpegImagePlugin

RESAMPLES={
    "NEAREST": Image.Resampling.NEAREST,
    "BOX": Image.Resampling.BOX,
    "BILINEAR": Image.Resampling.BILINEAR,
    "HAMMING": Image.Resampling.HAMMING,
    "BICUBIC": Image.Resampling.BICUBIC,
    "LANCZOS": Image.Resampling.LANCZOS,
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--zip",required=True)
    ap.add_argument("--metadata",required=True)
    ap.add_argument("--codalab-root",required=True)
    ap.add_argument("--resample",required=True,choices=sorted(RESAMPLES))
    ap.add_argument("--out",default="reports/phase5/validation/iwildcam_reconstruction_rule_validation.json")
    a=ap.parse_args()

    md=pd.read_csv(a.metadata); train=Path(a.codalab_root)/"train"
    overlap=sorted(fn for fn in md.filename.astype(str) if (train/fn).is_file())
    wanted=set(overlap); zmap={}
    with zipfile.ZipFile(a.zip) as z:
        for info in z.infolist():
            b=PurePosixPath(info.filename.replace("\\","/")).name
            if b in wanted:
                if b in zmap: raise RuntimeError(f"duplicate {b}")
                zmap[b]=info.filename
        miss=wanted-set(zmap)
        if miss: raise RuntimeError(f"missing overlap {len(miss)}")
        exact=0; total_mae=0.0; worst=0; bad=[]
        for i,fn in enumerate(overlap,1):
            with Image.open(io.BytesIO(z.read(zmap[fn]))) as src, Image.open(train/fn) as tgt:
                src.load(); tgt.load()
                x=src.convert("RGB").resize(tgt.size,resample=RESAMPLES[a.resample])
                buf=io.BytesIO(); kw={}
                q=getattr(tgt,"quantization",None)
                if q: kw["qtables"]={int(k):list(v) for k,v in q.items()}
                try:
                    s=JpegImagePlugin.get_sampling(tgt)
                    if s in (0,1,2): kw["subsampling"]=s
                except Exception: pass
                x.save(buf,format="JPEG",**kw)
                with Image.open(io.BytesIO(buf.getvalue())) as c:
                    c.load()
                    A=np.asarray(c.convert("RGB"),dtype=np.int16)
                    B=np.asarray(tgt.convert("RGB"),dtype=np.int16)
                d=np.abs(A-B)
                eq=np.array_equal(A,B)
                exact+=int(eq); total_mae+=float(d.mean()); worst=max(worst,int(d.max()))
                if not eq and len(bad)<20:
                    bad.append({"filename":fn,"mae":float(d.mean()),"max_abs":int(d.max())})
            if i%1000==0:
                print({"done":i,"total":len(overlap),"exact":exact,"mean_mae_so_far":total_mae/i})

    r={"n_compared":len(overlap),"exact":exact,"exact_fraction":exact/len(overlap),
       "mean_mae":total_mae/len(overlap),"max_abs_overall":worst,
       "mismatch_examples":bad,
       "decision":"PASS_EXACT" if exact==len(overlap) else "NOT_EXACT"}
    o=Path(a.out); o.parent.mkdir(parents=True,exist_ok=True); o.write_text(json.dumps(r,indent=2),encoding="utf-8")
    print(json.dumps(r,indent=2))

if __name__=="__main__":
    main()
