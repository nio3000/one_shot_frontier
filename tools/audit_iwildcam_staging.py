from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
from PIL import Image

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--metadata",required=True)
    ap.add_argument("--categories",required=True)
    ap.add_argument("--verify-images",action="store_true")
    ap.add_argument("--out",default="reports/phase5/validation/iwildcam_staging_audit.json")
    a=ap.parse_args()
    root=Path(a.root); md=pd.read_csv(a.metadata)
    expected=set(md.filename.astype(str))
    train=root/"train"
    actual={p.name for p in train.iterdir() if p.is_file() and not p.name.endswith(".part")} if train.exists() else set()
    unread=[]
    if a.verify_images:
        for fn in sorted(expected&actual):
            try:
                with Image.open(train/fn) as im: im.verify()
            except Exception as e: unread.append({"filename":fn,"error":repr(e)})
    ys=set(map(int,md.y.unique())); locs=set(map(int,md.location_remapped.unique())); seqs=set(map(int,md.sequence_remapped.unique()))
    checks={
      "rows_203029":len(md)==203029,
      "classes_182":ys==set(range(182)),
      "split_values_exact":set(md["split"].unique())=={"train","val","test","id_val","id_test"},
      "locations_323_contiguous":locs==set(range(323)),
      "sequences_contiguous":seqs==set(range(max(seqs)+1)),
      "filenames_unique":md.filename.nunique()==len(md),
      "all_files_present":len(expected-actual)==0,
      "no_extras":len(actual-expected)==0,
      "all_readable":len(unread)==0,
    }
    r={"metadata_sha256":sha256(a.metadata),"categories_sha256":sha256(a.categories),
       "rows":len(md),"classes":len(ys),"splits":md["split"].value_counts().to_dict(),
       "locations":md.location_remapped.nunique(),"sequences":md.sequence_remapped.nunique(),
       "expected_files":len(expected),"present_expected_files":len(expected&actual),
       "missing_files":len(expected-actual),"extra_files":len(actual-expected),
       "unreadable_images":len(unread),"checks":checks,"full_complete":all(checks.values())}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2),encoding="utf-8")
    print(json.dumps(r,indent=2))
if __name__=="__main__": main()
