from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frontier.phase1_featurebank import file_sha256, load_feature_bank


def _git_head(root: Path) -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "NO_GIT_HEAD"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(ROOT / "configs" / "phase1_protocol.yaml"))
    ap.add_argument("--features-dir", default=str(ROOT / "data" / "phase1" / "features"))
    ap.add_argument("--out", default=str(ROOT / "configs" / "phase1_feature_manifest.json"))
    args = ap.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    required = protocol["feature_banks"]["required"]
    features = Path(args.features_dir)
    banks = []
    for spec in required:
        bank_id = spec["bank_id"]
        p = features / f"{bank_id}.npz"
        meta_path = features / f"{bank_id}.meta.json"
        if not p.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing bank or metadata for {bank_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bank = load_feature_bank(p, meta)
        for k in ("dataset_id", "modality", "encoder_id"):
            if str(meta.get(k)) != str(spec[k]):
                raise RuntimeError(f"{bank_id} metadata field {k} does not match protocol")
        banks.append({
            **meta,
            "path": str(p.relative_to(ROOT)) if ROOT in p.resolve().parents else str(p.resolve()),
            "sha256": file_sha256(p),
            "meta_sha256": file_sha256(meta_path),
            "verified_feature_dim": int(bank.X_train.shape[1]),
            "verified_train_n": int(len(bank.y_train)),
            "verified_test_n": int(len(bank.y_test)),
            "verified_n_classes": int(max(bank.y_train.max(), bank.y_test.max()) + 1),
        })
    payload = {
        "freeze_type": "PHASE1_FEATURE_BANK_MANIFEST",
        "status": "FROZEN",
        "protocol_version_at_freeze": protocol.get("version"),
        "protocol_sha256_at_freeze": file_sha256(args.protocol),
        "git_head_at_freeze": _git_head(ROOT),
        "banks": banks,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    out = Path(args.out)
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite existing manifest: {out}")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
