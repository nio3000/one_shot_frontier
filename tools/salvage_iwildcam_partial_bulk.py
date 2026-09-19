from __future__ import annotations
import argparse, gzip, json, tarfile
from pathlib import Path, PurePosixPath
import pandas as pd
from PIL import Image

def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--staging-root", required=True)
    ap.add_argument("--out", default="reports/phase5/validation/iwildcam_partial_bulk_salvage.json")
    a=ap.parse_args()

    archive=Path(a.archive)
    md=pd.read_csv(a.metadata)
    authority=set(md["filename"].astype(str))
    train=Path(a.staging_root)/"train"
    train.mkdir(parents=True,exist_ok=True)

    seen=0
    authority_members=0
    already_valid=0
    extracted_new=0
    invalid_members=0
    stream_truncated=False
    stream_error=None
    first_members=[]
    last_members=[]

    try:
        with tarfile.open(archive, "r|gz") as tf:
            for m in tf:
                seen += 1
                name=m.name.replace("\\","/")
                if len(first_members)<5:
                    first_members.append(name)
                last_members=(last_members+[name])[-5:]
                if not m.isfile():
                    continue
                bn=PurePosixPath(name).name
                if bn not in authority:
                    continue
                authority_members += 1
                dst=train/bn
                if dst.exists() and valid_image(dst):
                    already_valid += 1
                    continue
                f=tf.extractfile(m)
                if f is None:
                    invalid_members += 1
                    continue
                tmp=dst.with_suffix(dst.suffix+".part")
                data=f.read()
                tmp.write_bytes(data)
                if not valid_image(tmp):
                    invalid_members += 1
                    try: tmp.unlink()
                    except OSError: pass
                    continue
                tmp.replace(dst)
                extracted_new += 1
    except (EOFError, gzip.BadGzipFile, tarfile.ReadError, OSError) as e:
        stream_truncated=True
        stream_error=repr(e)

    result={
        "archive":str(archive),
        "archive_bytes":archive.stat().st_size,
        "members_seen_before_eof":seen,
        "authority_members_seen":authority_members,
        "already_valid_skipped":already_valid,
        "new_official_images_salvaged":extracted_new,
        "invalid_members":invalid_members,
        "stream_truncated":stream_truncated,
        "stream_error":stream_error,
        "first_members":first_members,
        "last_members":last_members,
    }
    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
