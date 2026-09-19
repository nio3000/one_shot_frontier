
import argparse,zipfile,json,io
from pathlib import Path,PurePosixPath
import pandas as pd
from PIL import Image,ImageChops
ap=argparse.ArgumentParser(); ap.add_argument("--zip",required=True); ap.add_argument("--metadata",required=True)
ap.add_argument("--codalab-root",required=True); ap.add_argument("--out",default="reports/phase5/validation/iwildcam_kaggle_codalab_equivalence.json")
a=ap.parse_args()
auth=set(pd.read_csv(a.metadata).filename.astype(str)); root=Path(a.codalab_root)/"train"
overlap=sorted(fn for fn in auth if (root/fn).is_file())
wanted=set(overlap); zmap={}
with zipfile.ZipFile(a.zip) as z:
    for i in z.infolist():
        b=PurePosixPath(i.filename.replace("\\","/")).name
        if b in wanted:
            if b in zmap: raise RuntimeError(f"duplicate basename {b}")
            zmap[b]=i.filename
    miss=wanted-set(zmap)
    if miss: raise RuntimeError(f"missing overlap files: {len(miss)}")
    byte=pixel=0; bad=[]
    for k,fn in enumerate(overlap,1):
        cb=(root/fn).read_bytes(); kb=z.read(zmap[fn])
        if cb==kb: byte+=1; pixel+=1
        else:
            with Image.open(io.BytesIO(cb)) as ia, Image.open(io.BytesIO(kb)) as ib:
                ia.load(); ib.load()
                if ia.mode!=ib.mode: ia=ia.convert("RGB"); ib=ib.convert("RGB")
                eq=ia.size==ib.size and ImageChops.difference(ia,ib).getbbox() is None
                if eq: pixel+=1
                else: bad.append(fn)
        if k%1000==0: print({"done":k,"total":len(overlap),"pixel_exact":pixel})
r={"n_compared":len(overlap),"byte_exact":byte,"pixel_exact":pixel,
   "pixel_exact_all":pixel==len(overlap),"mismatches":len(bad),"mismatch_examples":bad[:20],
   "decision":"IWILDCAM_KAGGLE_TO_WILDS_IMAGE_EQUIVALENCE_PASS" if pixel==len(overlap) else "IWILDCAM_KAGGLE_TO_WILDS_IMAGE_EQUIVALENCE_FAIL"}
o=Path(a.out); o.parent.mkdir(parents=True,exist_ok=True); o.write_text(json.dumps(r,indent=2),encoding="utf-8"); print(json.dumps(r,indent=2))
