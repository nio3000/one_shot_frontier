
from __future__ import annotations
import argparse, gzip, json, re, tarfile
from pathlib import Path, PurePosixPath

PAT = re.compile(r"(?:^|/)images/(?P<experiment>[^/]+)/Plate(?P<plate>\d+)/(?P<well>[A-Z]\d+)_s(?P<site>[12])\.png$")

def parse_member(s):
    m=PAT.search(s.replace("\\","/").lstrip("./"))
    if not m: return None
    d=m.groupdict()
    d["plate"]=int(d["plate"]); d["site"]=int(d["site"])
    d["member"]=s
    d["key"]=(d["experiment"],d["plate"],d["site"],d["well"])
    return d

def relpath(d):
    return Path(d["experiment"])/f"Plate{d['plate']}"/f"{d['well']}_s{d['site']}.png"

def score(c, chosen, seen_exp, seen_ep, seen_eps):
    exp=c["experiment"]; ep=(exp,c["plate"]); eps=(exp,c["plate"],c["site"])
    return (
        1 if exp not in seen_exp else 0,
        1 if ep not in seen_ep else 0,
        1 if eps not in seen_eps else 0,
        -sum(x["experiment"]==exp for x in chosen),
        -sum((x["experiment"],x["plate"])==ep for x in chosen),
        -c["plate"],
        -c["site"],
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--members-seen", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=128)
    a=ap.parse_args()

    members=Path(a.members_seen).read_text(encoding="utf-8").splitlines()
    candidates=[]
    seen=set()
    for s in members:
        d=parse_member(s)
        if d and d["key"] not in seen:
            candidates.append(d); seen.add(d["key"])

    chosen=[]
    seen_exp=set(); seen_ep=set(); seen_eps=set()
    pool=candidates[:]
    while pool and len(chosen)<a.max_samples:
        pool.sort(key=lambda c:(score(c,chosen,seen_exp,seen_ep,seen_eps), c["member"]), reverse=True)
        c=pool.pop(0)
        chosen.append(c)
        seen_exp.add(c["experiment"])
        seen_ep.add((c["experiment"],c["plate"]))
        seen_eps.add((c["experiment"],c["plate"],c["site"]))

    wanted={c["member"].replace("\\","/").lstrip("./"):c for c in chosen}
    out=Path(a.out_dir)
    sample_root=out/"sample_images"
    sample_root.mkdir(parents=True, exist_ok=True)
    extracted=[]
    truncated=False; err=None

    try:
        with tarfile.open(Path(a.archive), "r|gz") as tf:
            for m in tf:
                norm=m.name.replace("\\","/").lstrip("./")
                if norm not in wanted or not m.isfile():
                    continue
                c=wanted[norm]
                target=sample_root/relpath(c)
                target.parent.mkdir(parents=True, exist_ok=True)
                f=tf.extractfile(m)
                if f is not None:
                    target.write_bytes(f.read())
                    extracted.append({
                        "kind":"sample_image",
                        "member":m.name,
                        "relative_sample_path":str(target.relative_to(sample_root)).replace("\\","/"),
                        "path":str(target),
                        "bytes":target.stat().st_size,
                        "experiment":c["experiment"],"plate":c["plate"],"site":c["site"],"well":c["well"],
                    })
    except (EOFError,gzip.BadGzipFile,tarfile.ReadError,OSError) as e:
        truncated=True; err=repr(e)

    res={
        "strategy":"deterministic greedy coverage: experiment -> experiment×plate -> experiment×plate×site; no outcome use",
        "candidate_images_before_truncation":len(candidates),
        "requested":len(chosen),
        "extracted_count":len(extracted),
        "unique_experiments":sorted({x["experiment"] for x in extracted}),
        "n_unique_experiments":len({x["experiment"] for x in extracted}),
        "n_unique_experiment_plate":len({(x["experiment"],x["plate"]) for x in extracted}),
        "sites":sorted({x["site"] for x in extracted}),
        "truncated_archive":truncated,
        "stream_error":err,
        "extracted":extracted,
    }
    (out/"stratified_recovery_report.json").write_text(json.dumps(res,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in res.items() if k!="extracted"},indent=2))

if __name__=="__main__":
    main()
