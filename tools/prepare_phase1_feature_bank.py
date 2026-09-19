from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from frontier.phase1_featurebank import collate_medmnist


def _state_dict_sha256(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _extract(loader, model, device: str):
    import torch
    feats, labels = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            z = model(x)
            if isinstance(z, (tuple, list)):
                z = z[0]
            feats.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
            labels.append(np.asarray(y).reshape(-1).astype(np.int64))
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def _build_encoder(encoder_id: str, device: str):
    import torch
    import torchvision
    from torch import nn
    from torchvision.models import (
        resnet18, ResNet18_Weights,
        vit_b_16, ViT_B_16_Weights,
    )
    if encoder_id == "resnet18_imagenet1k_v1":
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
    elif encoder_id == "vit_b_16_imagenet1k_v1":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        model.heads = nn.Identity()
    else:
        raise ValueError(encoder_id)
    model = model.to(device).eval()
    return model, weights, torchvision.__version__, torch.__version__


def _dataset(dataset_id: str, root: Path, transform, allow_download: bool, split_seed: int):
    import torch
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split
    from torchvision import datasets

    if dataset_id == "cifar100":
        tr = datasets.CIFAR100(root=str(root), train=True, transform=transform, download=allow_download)
        te = datasets.CIFAR100(root=str(root), train=False, transform=transform, download=allow_download)
        meta = {"split_id": "official_train__official_test"}
        return tr, te, meta
    if dataset_id == "eurosat":
        full = datasets.EuroSAT(root=str(root), transform=transform, download=allow_download)
        labels = np.array([s[1] for s in full.samples], dtype=np.int64)
        idx = np.arange(len(full))
        tr_idx, te_idx = train_test_split(idx, test_size=0.20, random_state=int(split_seed), stratify=labels)
        meta = {
            "split_id": f"stratified_80_20_seed{split_seed}",
            "train_indices_sha256": hashlib.sha256(np.sort(tr_idx).astype(np.int64).tobytes()).hexdigest(),
            "test_indices_sha256": hashlib.sha256(np.sort(te_idx).astype(np.int64).tobytes()).hexdigest(),
        }
        return Subset(full, tr_idx.tolist()), Subset(full, te_idx.tolist()), meta
    if dataset_id in {"pathmnist", "dermamnist"}:
        import medmnist
        info = medmnist.INFO[dataset_id]
        cls = getattr(medmnist, info["python_class"])
        tr = cls(split="train", root=str(root), transform=transform, download=allow_download)
        te = cls(split="test", root=str(root), transform=transform, download=allow_download)
        meta = {"split_id": "official_train__official_test", "medmnist_version": getattr(medmnist, "__version__", "unknown")}
        return tr, te, meta
    raise ValueError(dataset_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar100", "eurosat", "pathmnist", "dermamnist"], required=True)
    ap.add_argument("--encoder", choices=["resnet18_imagenet1k_v1", "vit_b_16_imagenet1k_v1"], required=True)
    ap.add_argument("--root", default="data/phase1_raw")
    ap.add_argument("--out-dir", default="data/phase1/features")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--split-seed", type=int, default=20260908)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, weights, tv_version, torch_version = _build_encoder(args.encoder, device)
    transform = weights.transforms()
    tr, te, split_meta = _dataset(args.dataset, Path(args.root), transform, args.allow_download, args.split_seed)

    use_custom = args.dataset in {"pathmnist", "dermamnist"}
    loader_kw = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.startswith("cuda"))
    tr_loader = DataLoader(tr, collate_fn=collate_medmnist if use_custom else None, **loader_kw)
    te_loader = DataLoader(te, collate_fn=collate_medmnist if use_custom else None, **loader_kw)

    Xtr, ytr = _extract(tr_loader, model, device)
    Xte, yte = _extract(te_loader, model, device)
    bank_id = f"{args.dataset}__{args.encoder}"
    modality = "medical" if args.dataset in {"pathmnist", "dermamnist"} else "nonmedical"
    meta = {
        "bank_id": bank_id,
        "dataset_id": args.dataset,
        "modality": modality,
        "encoder_id": args.encoder,
        "encoder_state_sha256": _state_dict_sha256(model),
        "weights_id": str(weights),
        "preprocessing_id": f"torchvision_weights_transforms::{weights}",
        "torch_version": torch_version,
        "torchvision_version": tv_version,
        "feature_dim": int(Xtr.shape[1]),
        "train_n": int(len(ytr)),
        "test_n": int(len(yte)),
        "n_classes": int(max(ytr.max(), yte.max()) + 1),
        "label_min": int(min(ytr.min(), yte.min())),
        "label_max": int(max(ytr.max(), yte.max())),
        **split_meta,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{bank_id}.npz"
    np.savez(out, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, metadata_json=json.dumps(meta, ensure_ascii=False))
    (out_dir / f"{bank_id}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"bank": str(out), "metadata": meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
