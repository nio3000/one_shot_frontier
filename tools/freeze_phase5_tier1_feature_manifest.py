from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPECTED=[
    "camelyon17__resnet18_imagenet1k_v1",
    "camelyon17__vit_b_16_imagenet1k_v1",
    "rxrx1__resnet18_imagenet1k_v1",
    "rxrx1__vit_b_16_imagenet1k_v1",
    "iwildcam__resnet18_imagenet1k_v1",
    "iwildcam__vit_b_16_imagenet1k_v1",
]

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feature-dir",default=str(ROOT/"data"/"phase5_tier1"/"features"))
    ap.add_argument("--out",default=str(ROOT/"configs"/"phase5_tier1_feature_manifest.json"))
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists(): raise RuntimeError(f"Refusing overwrite: {out}")
    auth=ROOT/"configs"/"phase5_tier1_unblind_authorization.json"
    if not auth.exists(): raise RuntimeError("Authorization missing")
    entries=[]
    for bid in EXPECTED:
        p=Path(args.feature_dir)/f"{bid}.npz"
        if not p.exists(): raise FileNotFoundError(p)
        import numpy as np
        with np.load(p,allow_pickle=False) as z:
            raw=z["metadata_json"]; raw=raw.item() if np.ndim(raw)==0 else raw
            meta=json.loads(str(raw))
            n=len(z["y"]); d=z["X"].shape[1]
        if meta.get("bank_id")!=bid: raise RuntimeError(f"bank_id mismatch {bid}")
        entries.append({
            "bank_id":bid,
            "dataset_id":meta["dataset_id"],
            "dataset_version":meta["dataset_version"],
            "encoder_id":meta["encoder_id"],
            "path":str(p.relative_to(ROOT)).replace("\\","/"),
            "sha256":sha(p),
            "metadata_sha256":meta["metadata_sha256"],
            "encoder_state_sha256":meta["encoder_state_sha256"],
            "feature_dim":int(d),
            "n_rows":int(n),
            "n_classes":int(meta["n_classes"]),
            "n_domains":int(meta["n_domains"]),
        })
    payload={
        "freeze_type":"PHASE5_TIER1_FEATURE_MANIFEST",
        "status":"FROZEN_AFTER_AUTHORIZED_UNBLIND",
        "authorization_sha256":sha(auth),
        "banks":entries,
    }
    canon=json.dumps(payload,sort_keys=True,separators=(",",":")).encode()
    payload["payload_sha256"]=hashlib.sha256(canon).hexdigest()
    out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()
