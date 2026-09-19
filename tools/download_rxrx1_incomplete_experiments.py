from __future__ import annotations
import argparse,json,shutil,subprocess,tarfile
from pathlib import Path,PurePosixPath

DEFAULT_BUNDLE="0x6b7a05a3056a434498f0bb1252eb8440"
BASE="https://worksheets.codalab.org/rest/bundles/{bundle}/contents/blob/images/{experiment}"

def _normalized_target_rel(member_name:str, experiment:str):
    p=PurePosixPath(member_name.replace("\\","/").lstrip("./"))
    if ".." in p.parts or not p.parts:
        raise RuntimeError(f"unsafe archive member: {member_name}")
    parts=list(p.parts)
    if parts[0]=="images":
        if len(parts)<4 or parts[1]!=experiment: return None
        rel=Path(*parts)
    elif parts[0]==experiment:
        rel=Path("images",*parts)
    elif parts[0].startswith("Plate"):
        rel=Path("images",experiment,*parts)
    else:
        return None
    return rel if rel.suffix.lower()==".png" else None

def extract_experiment_tar(tar_path:Path, staging_root:Path, experiment:str):
    n=0
    with tarfile.open(tar_path,"r:*") as tf:
        for m in tf:
            if not m.isfile(): continue
            rel=_normalized_target_rel(m.name,experiment)
            if rel is None: continue
            target=staging_root/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            f=tf.extractfile(m)
            if f is not None:
                target.write_bytes(f.read()); n+=1
    return n

def curl_download(url,out):
    curl=shutil.which("curl.exe") or shutil.which("curl")
    if not curl: raise RuntimeError("curl not found")
    subprocess.run([curl,"-L","--fail","--show-error","--retry","20","--retry-all-errors",
                    "--connect-timeout","30","--output",str(out),url],check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--audit",required=True)
    ap.add_argument("--staging-root",required=True)
    ap.add_argument("--bundle",default=DEFAULT_BUNDLE)
    ap.add_argument("--max-experiments",type=int,default=1)
    ap.add_argument("--only")
    ap.add_argument("--reuse-existing-archives",action="store_true")
    a=ap.parse_args()
    audit=json.loads(Path(a.audit).read_text(encoding="utf-8"))
    exps=[x["experiment"] for x in audit["experiments"] if not x["complete"]]
    if a.only: exps=[x for x in exps if x==a.only]
    if a.max_experiments>0: exps=exps[:a.max_experiments]
    print(json.dumps({"selected":exps},indent=2))
    staging=Path(a.staging_root); temp=staging/"_segment_downloads"; temp.mkdir(parents=True,exist_ok=True)
    log=[]
    for exp in exps:
        arc=temp/f"{exp}.tar.gz"
        reused=a.reuse_existing_archives and arc.exists() and tarfile.is_tarfile(arc)
        if not reused:
            url=BASE.format(bundle=a.bundle,experiment=exp)+"?support_redirect=1"
            curl_download(url,arc)
            if not tarfile.is_tarfile(arc):
                curl_download(BASE.format(bundle=a.bundle,experiment=exp),arc)
        if not tarfile.is_tarfile(arc): raise RuntimeError(f"not tar: {arc}")
        n=extract_experiment_tar(arc,staging,exp)
        log.append({"experiment":exp,"reused":reused,"archive_bytes":arc.stat().st_size,"png_written":n})
    (temp/"segmented_download_log_v2.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
    print(json.dumps(log,indent=2))
if __name__=="__main__": main()
