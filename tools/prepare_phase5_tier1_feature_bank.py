from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from frontier.phase5_wilds_adapters import DATASET_SPECS, read_normalized_metadata
from frontier.phase5_wilds_adapters import file_sha256


def _guard():
    p=ROOT/"configs"/"phase5_tier1_unblind_authorization.json"
    if not p.exists():
        raise RuntimeError("Feature extraction blocked: committed unblind authorization artifact required")
    a=json.loads(p.read_text(encoding="utf-8"))
    if a.get("status")!="AUTHORIZED_AFTER_PHASE5_FREEZE":
        raise RuntimeError("Invalid unblind authorization")


def _state_dict_sha256(model) -> str:
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        arr=tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.shape).encode()); h.update(str(arr.dtype).encode()); h.update(arr.tobytes())
    return h.hexdigest()


def _build_encoder(encoder_id,device):
    import torch
    import torchvision
    from torch import nn
    from torchvision.models import resnet18,ResNet18_Weights,vit_b_16,ViT_B_16_Weights
    if encoder_id=="resnet18_imagenet1k_v1":
        weights=ResNet18_Weights.IMAGENET1K_V1
        model=resnet18(weights=weights); model.fc=nn.Identity()
    elif encoder_id=="vit_b_16_imagenet1k_v1":
        weights=ViT_B_16_Weights.IMAGENET1K_V1
        model=vit_b_16(weights=weights); model.heads=nn.Identity()
    else:
        raise ValueError(encoder_id)
    return model.to(device).eval(),weights,torchvision.__version__,torch.__version__


class ImageRecordDataset:
    def __init__(self, paths, transform):
        self.paths=list(map(str,paths)); self.transform=transform
    def __len__(self): return len(self.paths)
    def __getitem__(self,idx):
        from PIL import Image
        img=Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=list(DATASET_SPECS),required=True)
    ap.add_argument("--encoder",choices=["resnet18_imagenet1k_v1","vit_b_16_imagenet1k_v1"],required=True)
    ap.add_argument("--root",default=str(ROOT/"data"/"phase5_tier1_raw"))
    ap.add_argument("--out-dir",default=str(ROOT/"data"/"phase5_tier1"/"features"))
    ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--device",default=None)
    args=ap.parse_args()
    _guard()

    import torch
    from torch.utils.data import DataLoader

    records,dsmeta=read_normalized_metadata(args.dataset,args.root)
    missing=[p for p in records["sample_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image files missing; first={missing[0]}")

    device=args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model,weights,tv_ver,torch_ver=_build_encoder(args.encoder,device)
    dataset=ImageRecordDataset(records["sample_path"],weights.transforms())
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.startswith("cuda"))

    feats=[]
    with torch.inference_mode():
        for x in loader:
            z=model(x.to(device,non_blocking=True))
            if isinstance(z,(tuple,list)): z=z[0]
            feats.append(z.detach().cpu().numpy().astype(np.float32,copy=False))
    X=np.concatenate(feats,axis=0)

    bank_id=f"{args.dataset}__{args.encoder}"
    auth_path=ROOT/"configs"/"phase5_tier1_unblind_authorization.json"
    meta={
        **dsmeta,
        "bank_id":bank_id,
        "encoder_id":args.encoder,
        "encoder_state_sha256":_state_dict_sha256(model),
        "weights_id":str(weights),
        "preprocessing_id":f"torchvision_weights_transforms::{weights}",
        "torch_version":torch_ver,
        "torchvision_version":tv_ver,
        "feature_dim":int(X.shape[1]),
        "n_rows":int(len(X)),
        "n_classes":int(records["y"].nunique()),
        "n_domains":int(records["domain_id"].nunique()),
        "custom_representation_route":True,
        "wilds_leaderboard_compliance_claim":False,
        "authorization_sha256":file_sha256(auth_path),
    }
    outdir=Path(args.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/f"{bank_id}.npz"
    np.savez(
        out,
        X=X,
        y=records["y"].to_numpy(dtype=np.int64),
        domain_id=records["domain_id"].to_numpy(dtype=np.int64),
        group_id=records["group_id"].to_numpy(dtype=np.int64),
        official_split_id=records["official_split_id"].to_numpy(dtype=np.int64),
        metadata_json=json.dumps(meta,ensure_ascii=False),
    )
    (outdir/f"{bank_id}.meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"bank":str(out),"sha256":file_sha256(out),"metadata":meta},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
