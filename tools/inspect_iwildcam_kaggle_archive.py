
import argparse,zipfile,json,collections
from pathlib import Path,PurePosixPath
import pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument("--zip",required=True); ap.add_argument("--metadata",required=True)
ap.add_argument("--out",default="reports/phase5/validation/iwildcam_kaggle_archive_inspection.json")
a=ap.parse_args()
auth=set(pd.read_csv(a.metadata).filename.astype(str))
paths={}; dups=collections.defaultdict(list); top=collections.Counter(); jpg=0; entries=0
with zipfile.ZipFile(a.zip) as z:
    for i in z.infolist():
        entries+=1; n=i.filename.replace("\\","/"); p=PurePosixPath(n)
        if p.parts: top[p.parts[0]]+=1
        if p.suffix.lower() in {".jpg",".jpeg"}:
            jpg+=1; b=p.name
            if b in auth:
                if b in paths: dups[b].append(n)
                else: paths[b]=n
r={"zip_GB":Path(a.zip).stat().st_size/1e9,"entries":entries,"jpg_entries":jpg,
   "top_prefixes":top.most_common(20),"authority_filenames":len(auth),
   "authority_unique_matches":len(paths),"authority_missing":len(auth-set(paths)),
   "authority_duplicate_basenames":len(dups),
   "exact_universe_candidate":len(paths)==len(auth) and not dups}
o=Path(a.out); o.parent.mkdir(parents=True,exist_ok=True); o.write_text(json.dumps(r,indent=2),encoding="utf-8")
print(json.dumps(r,indent=2))
