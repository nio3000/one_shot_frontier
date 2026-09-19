from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, random, shutil, time, urllib.request
from pathlib import Path
import pandas as pd
from PIL import Image

BUNDLE="0x6313da2b204647e79a14b468131fcd64"
BASE="https://worksheets.codalab.org/rest/bundles/{bundle}/contents/blob/train/{filename}?support_redirect=1"

def valid_image(p:Path)->bool:
    try:
        if not p.exists() or p.stat().st_size<=0: return False
        with Image.open(p) as im: im.verify()
        return True
    except Exception:
        return False

def fetch_one(fn,out,bundle,retries,timeout):
    target=out/"train"/fn
    target.parent.mkdir(parents=True,exist_ok=True)
    if valid_image(target):
        return {"filename":fn,"status":"already_valid","bytes":target.stat().st_size}
    tmp=target.with_suffix(target.suffix+".part")
    url=BASE.format(bundle=bundle,filename=fn)
    last=None
    for i in range(retries+1):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"phase5-iwildcam/1.0"})
            with urllib.request.urlopen(req,timeout=timeout) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r,f,1<<20)
            if not valid_image(tmp): raise RuntimeError("downloaded file invalid")
            os.replace(tmp,target)
            return {"filename":fn,"status":"downloaded","bytes":target.stat().st_size}
        except Exception as e:
            last=repr(e)
            if tmp.exists():
                try: tmp.unlink()
                except OSError: pass
            if i<retries: time.sleep(min(30,2**i+random.random()))
    return {"filename":fn,"status":"failed","error":last}

def select_probe(df,n):
    w=df[["filename","split","location_remapped"]].sort_values(
        ["split","location_remapped","filename"],kind="mergesort")
    groups=[g.reset_index(drop=True) for _,g in w.groupby(["split","location_remapped"],sort=True)]
    out=[]; idx=0
    while len(out)<n and groups:
        new=[]
        for g in groups:
            if idx<len(g) and len(out)<n: out.append(str(g.iloc[idx].filename))
            if idx+1<len(g): new.append(g)
        groups=new; idx+=1
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metadata",required=True)
    ap.add_argument("--out-root",required=True)
    ap.add_argument("--bundle",default=BUNDLE)
    ap.add_argument("--workers",type=int,default=12)
    ap.add_argument("--retries",type=int,default=8)
    ap.add_argument("--timeout",type=int,default=120)
    ap.add_argument("--probe",type=int,default=0)
    ap.add_argument("--log-every",type=int,default=500)
    a=ap.parse_args()

    md=pd.read_csv(a.metadata)
    if md.filename.duplicated().any(): raise RuntimeError("duplicate authority filenames")
    fns=select_probe(md,a.probe) if a.probe>0 else sorted(md.filename.astype(str).tolist())
    out=Path(a.out_root); out.mkdir(parents=True,exist_ok=True)
    logp=out/"_filewise_recovery_log.jsonl"
    results=[]
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex, logp.open("a",encoding="utf-8") as log:
        futs=[ex.submit(fetch_one,fn,out,a.bundle,a.retries,a.timeout) for fn in fns]
        for i,f in enumerate(cf.as_completed(futs),1):
            r=f.result(); results.append(r)
            log.write(json.dumps(r,ensure_ascii=False)+"\n"); log.flush()
            if i%a.log_every==0 or i==len(fns):
                counts={}
                for x in results: counts[x["status"]]=counts.get(x["status"],0)+1
                print(json.dumps({"done":i,"total":len(fns),"status":counts}))
    failed=[x for x in results if x["status"]=="failed"]
    s={"requested":len(fns),"downloaded":sum(x["status"]=="downloaded" for x in results),
       "already_valid":sum(x["status"]=="already_valid" for x in results),
       "failed":len(failed),"complete":len(failed)==0,"failed_examples":failed[:20]}
    (out/"_filewise_recovery_summary.json").write_text(json.dumps(s,indent=2),encoding="utf-8")
    print(json.dumps(s,indent=2))
if __name__=="__main__": main()
