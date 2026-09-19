from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path
import pandas as pd
from PIL import Image, JpegImagePlugin

def qsig(im: Image.Image) -> str:
    q = getattr(im, "quantization", None)
    if not q:
        return "NONE"
    payload = json.dumps({int(k): list(v) for k,v in sorted(q.items())}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--codalab-root", required=True)
    ap.add_argument("--out", default="reports/phase5/validation/iwildcam_wilds_jpeg_signature.json")
    a=ap.parse_args()

    md=pd.read_csv(a.metadata)
    train=Path(a.codalab_root)/"train"
    files=sorted(fn for fn in md.filename.astype(str) if (train/fn).is_file())
    if not files:
        raise RuntimeError("no official WILDS images found")

    heights=collections.Counter()
    modes=collections.Counter()
    sampling=collections.Counter()
    qtables=collections.Counter()
    progressive=collections.Counter()
    exif_orientation=collections.Counter()
    widths=[]
    bad=[]

    for i,fn in enumerate(files,1):
        p=train/fn
        try:
            with Image.open(p) as im:
                widths.append(im.width)
                heights[im.height]+=1
                modes[im.mode]+=1
                try:
                    sampling[str(JpegImagePlugin.get_sampling(im))]+=1
                except Exception:
                    sampling["ERR"]+=1
                qtables[qsig(im)]+=1
                progressive[str(bool(im.info.get("progressive") or im.info.get("progression")))]+=1
                ex=im.getexif()
                exif_orientation[str(ex.get(274,"NONE"))]+=1
        except Exception as e:
            bad.append({"filename":fn,"error":repr(e)})
        if i%5000==0:
            print({"done":i,"total":len(files)})

    result={
        "n_official_images":len(files),
        "height_counts":heights.most_common(),
        "width_min":min(widths),
        "width_max":max(widths),
        "mode_counts":modes.most_common(),
        "sampling_counts":sampling.most_common(),
        "qtable_signature_counts":qtables.most_common(),
        "progressive_counts":progressive.most_common(),
        "exif_orientation_counts":exif_orientation.most_common(),
        "unreadable":len(bad),
        "unreadable_examples":bad[:20],
    }
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
