
import argparse,zipfile,json,shutil,os
from pathlib import Path,PurePosixPath
import pandas as pd
from PIL import Image
def valid(p):
    try:
        with Image.open(p) as im: im.verify()
        return True
    except: return False
ap=argparse.ArgumentParser(); ap.add_argument("--zip",required=True); ap.add_argument("--metadata",required=True)
ap.add_argument("--out-root",required=True); ap.add_argument("--equivalence-report",required=True)
ap.add_argument("--reuse-existing-root"); a=ap.parse_args()
eq=json.loads(Path(a.equivalence_report).read_text(encoding="utf-8"))
if eq.get("decision")!="IWILDCAM_KAGGLE_TO_WILDS_IMAGE_EQUIVALENCE_PASS": raise RuntimeError("equivalence PASS required")
auth=sorted(set(pd.read_csv(a.metadata).filename.astype(str))); wanted=set(auth); out=Path(a.out_root)/"train"; out.mkdir(parents=True,exist_ok=True)
if a.reuse_existing_root:
    src=Path(a.reuse_existing_root)/"train"
    for fn in auth:
        s=src/fn; d=out/fn
        if s.is_file() and not d.exists(): shutil.copy2(s,d)
with zipfile.ZipFile(a.zip) as z:
    zmap={}
    for i in z.infolist():
        b=PurePosixPath(i.filename.replace("\\","/")).name
        if b in wanted:
            if b in zmap: raise RuntimeError(f"duplicate basename {b}")
            zmap[b]=i.filename
    if len(zmap)!=len(wanted): raise RuntimeError(f"zip only matches {len(zmap)}/{len(wanted)} authority files")
    extracted=already=0
    for k,fn in enumerate(auth,1):
        d=out/fn
        if d.exists() and valid(d): already+=1
        else:
            t=d.with_suffix(d.suffix+".part")
            with z.open(zmap[fn]) as s, t.open("wb") as f: shutil.copyfileobj(s,f,1<<20)
            if not valid(t): raise RuntimeError(f"invalid image {fn}")
            os.replace(t,d); extracted+=1
        if k%5000==0: print({"done":k,"total":len(auth),"already":already,"extracted":extracted})
print(json.dumps({"authority_files":len(auth),"already_valid":already,"extracted_from_kaggle":extracted,"complete":True},indent=2))
