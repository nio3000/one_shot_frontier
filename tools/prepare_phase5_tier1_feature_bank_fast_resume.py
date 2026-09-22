from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase5_wilds_adapters import DATASET_SPECS, read_normalized_metadata
from frontier.phase5_wilds_adapters import file_sha256

ENCODER_DIMS = {
    "resnet18_imagenet1k_v1": 512,
    "vit_b_16_imagenet1k_v1": 768,
}


def _guard() -> None:
    p = ROOT / "configs" / "phase5_tier1_unblind_authorization.json"
    if not p.exists():
        raise RuntimeError("Feature extraction blocked: committed unblind authorization artifact required")
    a = json.loads(p.read_text(encoding="utf-8"))
    if a.get("status") != "AUTHORIZED_AFTER_PHASE5_FREEZE":
        raise RuntimeError("Invalid unblind authorization")


def _state_dict_sha256(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _build_encoder(encoder_id: str, device: str):
    import torch
    import torchvision
    from torch import nn
    from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16

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

    return model.to(device).eval(), weights, torchvision.__version__, torch.__version__


class FilesystemImageDataset:
    def __init__(self, paths, transform):
        self.paths = list(map(str, paths))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        with Image.open(self.paths[idx]) as img:
            return self.transform(img.convert("RGB"))


class ZipImageDataset:
    def __init__(self, zip_path: str | Path, members, transform):
        self.zip_path = str(zip_path)
        self.members = list(map(str, members))
        self.transform = transform
        self._zf = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zf"] = None
        return state

    def _zip(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path, "r")
        return self._zf

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        from PIL import Image
        data = self._zip().read(self.members[idx])
        with Image.open(io.BytesIO(data)) as img:
            return self.transform(img.convert("RGB"))


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _expected_bank_sha(bank_id: str) -> str | None:
    manifest = ROOT / "configs" / "phase5_tier1_feature_manifest.json"
    if not manifest.exists():
        return None
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    for bank in obj.get("banks", []):
        if bank.get("bank_id") == bank_id:
            return bank.get("sha256")
    return None


def _canonical_relpaths(records, dataset_id: str, canonical_root: Path) -> list[Path]:
    folder = DATASET_SPECS[dataset_id]["folder"]
    droot = canonical_root / folder
    out = []
    for p in records["sample_path"]:
        pp = Path(p)
        try:
            out.append(pp.relative_to(droot))
        except ValueError as e:
            raise RuntimeError(f"Sample path is not under canonical dataset root: {pp} vs {droot}") from e
    return out


def _make_loader(dataset, batch_size: int, workers: int, device: str, prefetch_factor: int):
    from torch.utils.data import DataLoader
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
    )
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def _validate_resume_state(state: dict, expected: dict) -> None:
    for k in ("bank_id", "metadata_sha256", "n_rows", "feature_dim", "dtype"):
        if state.get(k) != expected.get(k):
            raise RuntimeError(
                f"Resume state mismatch for {k}: state={state.get(k)!r}, expected={expected.get(k)!r}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact-output resumable Phase-5 feature extraction")
    ap.add_argument("--dataset", choices=list(DATASET_SPECS), required=True)
    ap.add_argument("--encoder", choices=list(ENCODER_DIMS), required=True)
    ap.add_argument("--root", default=str(ROOT / "data" / "phase5_tier1_raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "phase5_tier1" / "features"))
    ap.add_argument("--io-mode", choices=["filesystem", "zip"], default="filesystem")
    ap.add_argument("--io-root", default=None)
    ap.add_argument("--source-zip", default=None)
    ap.add_argument("--zip-prefix", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume-dir", default=None)
    ap.add_argument("--checkpoint-every-batches", type=int, default=10)
    ap.add_argument("--progress-every-batches", type=int, default=10)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--keep-resume", action="store_true")
    ap.add_argument("--no-sha-gate", action="store_true")
    args = ap.parse_args()

    _guard()
    import torch

    canonical_root = Path(args.root)
    records, dsmeta = read_normalized_metadata(args.dataset, canonical_root)
    bank_id = f"{args.dataset}__{args.encoder}"
    n_rows = int(len(records))
    feature_dim = int(ENCODER_DIMS[args.encoder])

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{bank_id}.npz"
    meta_out = outdir / f"{bank_id}.meta.json"

    resume_dir = Path(args.resume_dir) if args.resume_dir else outdir / ".resume" / bank_id
    state_path = resume_dir / "state.json"
    mmap_path = resume_dir / "X.float32.memmap"

    if args.restart and resume_dir.exists():
        shutil.rmtree(resume_dir)
    resume_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, weights, tv_ver, torch_ver = _build_encoder(args.encoder, device)

    expected_state = {
        "bank_id": bank_id,
        "metadata_sha256": dsmeta["metadata_sha256"],
        "n_rows": n_rows,
        "feature_dim": feature_dim,
        "dtype": "float32",
    }

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _validate_resume_state(state, expected_state)
        start_idx = int(state.get("completed_rows", 0))
        if not mmap_path.exists():
            raise RuntimeError(f"Resume state exists but memmap is missing: {mmap_path}")
        X = np.memmap(mmap_path, dtype=np.float32, mode="r+", shape=(n_rows, feature_dim), order="C")
        print(f"[resume] {start_idx}/{n_rows} ({100.0 * start_idx / n_rows:.2f}%)", flush=True)
    else:
        start_idx = 0
        X = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(n_rows, feature_dim), order="C")
        state = {**expected_state, "completed_rows": 0, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        X.flush()
        _atomic_json(state_path, state)

    if not (0 <= start_idx <= n_rows):
        raise RuntimeError(f"Invalid completed_rows={start_idx}")

    if start_idx < n_rows:
        relpaths = _canonical_relpaths(records, args.dataset, canonical_root)[start_idx:]
        transform = weights.transforms()

        if args.io_mode == "filesystem":
            io_root = Path(args.io_root) if args.io_root else canonical_root
            actual_root = io_root / DATASET_SPECS[args.dataset]["folder"]
            paths = [actual_root / rel for rel in relpaths]
            probes = [paths[0], paths[len(paths) // 2], paths[-1]] if paths else []
            missing = [p for p in probes if not p.exists()]
            if missing:
                raise FileNotFoundError(f"I/O root probe failed; missing: {missing[0]}")
            dataset = FilesystemImageDataset(paths, transform)
        else:
            if not args.source_zip:
                raise ValueError("--source-zip is required for --io-mode zip")
            zpath = Path(args.source_zip)
            if not zpath.exists():
                raise FileNotFoundError(zpath)
            prefix = args.zip_prefix or DATASET_SPECS[args.dataset]["folder"]
            prefix = prefix.strip("/\\").replace("\\", "/")
            members = [f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix() for rel in relpaths]
            with zipfile.ZipFile(zpath, "r") as zf:
                for member in (members[0], members[len(members) // 2], members[-1]):
                    try:
                        zf.getinfo(member)
                    except KeyError as e:
                        raise FileNotFoundError(f"ZIP member not found: {member}") from e
            dataset = ZipImageDataset(zpath, members, transform)

        loader = _make_loader(dataset, args.batch_size, args.workers, device, args.prefetch_factor)
        session_t0 = time.time()
        session_rows = 0
        completed = start_idx
        batch_no = 0

        try:
            with torch.inference_mode():
                for x in loader:
                    z = model(x.to(device, non_blocking=True))
                    if isinstance(z, (tuple, list)):
                        z = z[0]
                    arr = z.detach().cpu().numpy().astype(np.float32, copy=False)
                    b = int(arr.shape[0])
                    end = completed + b
                    if end > n_rows:
                        raise RuntimeError("Feature write exceeds n_rows")
                    X[completed:end, :] = arr
                    completed = end
                    session_rows += b
                    batch_no += 1

                    if batch_no % args.checkpoint_every_batches == 0:
                        X.flush()
                        state["completed_rows"] = completed
                        state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        _atomic_json(state_path, state)

                    if batch_no % args.progress_every_batches == 0:
                        elapsed = max(time.time() - session_t0, 1e-9)
                        rate = session_rows / elapsed
                        eta_s = (n_rows - completed) / max(rate, 1e-9)
                        print(
                            f"[progress] {completed}/{n_rows} ({100.0 * completed / n_rows:.2f}%) "
                            f"| {rate:.2f} img/s | ETA {eta_s / 60.0:.1f} min "
                            f"| mode={args.io_mode} workers={args.workers}",
                            flush=True,
                        )
        except KeyboardInterrupt:
            X.flush()
            state["completed_rows"] = completed
            state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            state["interrupted"] = True
            _atomic_json(state_path, state)
            print(f"\n[interrupt] checkpoint saved at {completed}/{n_rows} ({100.0 * completed / n_rows:.2f}%).", flush=True)
            print("[interrupt] Re-run the same command tomorrow to resume.", flush=True)
            raise SystemExit(130)

        X.flush()
        state["completed_rows"] = completed
        state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["interrupted"] = False
        _atomic_json(state_path, state)

    if int(state.get("completed_rows", 0)) != n_rows:
        raise RuntimeError(f"Extraction incomplete: {state.get('completed_rows')} != {n_rows}")

    del X
    X = np.memmap(mmap_path, dtype=np.float32, mode="r", shape=(n_rows, feature_dim), order="C")

    meta = {
        **dsmeta,
        "bank_id": bank_id,
        "encoder_id": args.encoder,
        "encoder_state_sha256": _state_dict_sha256(model),
        "weights_id": str(weights),
        "preprocessing_id": f"torchvision_weights_transforms::{weights}",
        "torch_version": torch_ver,
        "torchvision_version": tv_ver,
        "feature_dim": feature_dim,
        "n_rows": n_rows,
        "n_classes": int(records["y"].nunique()),
        "n_domains": int(records["domain_id"].nunique()),
        "custom_representation_route": True,
        "wilds_leaderboard_compliance_claim": False,
        "authorization_sha256": file_sha256(ROOT / "configs" / "phase5_tier1_unblind_authorization.json"),
    }

    print("[finalize] writing NPZ ...", flush=True)
    np.savez(
        out,
        X=X,
        y=records["y"].to_numpy(dtype=np.int64),
        domain_id=records["domain_id"].to_numpy(dtype=np.int64),
        group_id=records["group_id"].to_numpy(dtype=np.int64),
        official_split_id=records["official_split_id"].to_numpy(dtype=np.int64),
        metadata_json=json.dumps(meta, ensure_ascii=False),
    )
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    actual_sha = file_sha256(out)
    expected_sha = _expected_bank_sha(bank_id)
    exact = expected_sha is None or actual_sha == expected_sha
    print(json.dumps({
        "bank": str(out),
        "sha256": actual_sha,
        "expected_sha256": expected_sha,
        "exact_manifest_match": exact if expected_sha is not None else None,
        "metadata": meta,
        "resume_dir": str(resume_dir),
    }, ensure_ascii=False, indent=2))

    if expected_sha is not None and actual_sha != expected_sha and not args.no_sha_gate:
        raise SystemExit("FAIL: generated feature-bank SHA does not match the frozen manifest; checkpoint retained.")

    if exact and not args.keep_resume:
        del X
        shutil.rmtree(resume_dir, ignore_errors=True)
        print("[cleanup] exact SHA PASS; resume files removed.")


if __name__ == "__main__":
    main()
