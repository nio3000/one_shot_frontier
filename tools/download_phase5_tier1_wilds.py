from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "camelyon17": {"version": "1.0", "folder": "camelyon17_v1.0", "required_dir": "patches"},
    "rxrx1": {"version": "1.0", "folder": "rxrx1_v1.0", "required_dir": "images"},
    "iwildcam": {"version": "2.0", "folder": "iwildcam_v2.0", "required_dir": "train"},
}

def _authorization_guard() -> Path:
    p = ROOT / "configs" / "phase5_tier1_unblind_authorization.json"
    if not p.exists():
        raise RuntimeError("Tier-1 download blocked: unblind authorization is missing")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("status") != "AUTHORIZED_AFTER_PHASE5_FREEZE":
        raise RuntimeError("Tier-1 download blocked: authorization status invalid")
    return p

def inspect_dataset_state(root: Path, dataset_id: str) -> dict:
    spec = DATASETS[dataset_id]
    d = root / spec["folder"]
    release = d / f"RELEASE_v{spec['version']}.txt"
    metadata = d / "metadata.csv"
    archive = d / "archive.tar.gz"
    required_dir = d / spec["required_dir"]
    state = {
        "dataset_id": dataset_id,
        "version": spec["version"],
        "data_dir": str(d),
        "exists": d.exists(),
        "release_marker": release.exists(),
        "metadata_csv": metadata.exists(),
        "archive_exists": archive.exists(),
        "archive_bytes": archive.stat().st_size if archive.exists() else None,
        "required_image_dir": required_dir.exists(),
    }
    if metadata.exists() and required_dir.exists():
        state["status"] = "LOCALLY_USABLE_FOR_ADAPTER_PREFLIGHT"
    elif d.exists():
        state["status"] = "INCOMPLETE_LOCAL_DIRECTORY"
    else:
        state["status"] = "ABSENT"
    return state

def quarantine_incomplete(root: Path, dataset_id: str, quarantine_root: Path) -> Path:
    state = inspect_dataset_state(root, dataset_id)
    if state["status"] != "INCOMPLETE_LOCAL_DIRECTORY":
        raise RuntimeError(f"Refusing quarantine because state is {state['status']}")
    src = Path(state["data_dir"])
    quarantine_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = quarantine_root / f"{src.name}_INCOMPLETE_{stamp}"
    shutil.move(str(src), str(dst))
    return dst

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "phase5_tier1_raw"))
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--repair-incomplete", action="store_true")
    ap.add_argument("--quarantine-root", default=str(ROOT / "data" / "phase5_tier1_quarantine"))
    args = ap.parse_args()

    _authorization_guard()
    root = Path(args.root)
    quarantine_root = Path(args.quarantine_root)
    root.mkdir(parents=True, exist_ok=True)

    import wilds
    from wilds import get_dataset
    if getattr(wilds, "__version__", "unknown") != "2.0.0":
        raise RuntimeError(f"Expected wilds==2.0.0, got {getattr(wilds, '__version__', 'unknown')}")

    results = []
    for dataset_id in args.datasets:
        spec = DATASETS[dataset_id]
        before = inspect_dataset_state(root, dataset_id)
        print(json.dumps({"preflight": before}, indent=2))

        if before["status"] == "INCOMPLETE_LOCAL_DIRECTORY":
            if not args.repair_incomplete:
                raise RuntimeError(
                    f"{dataset_id}: incomplete local directory detected. "
                    "Re-run with --repair-incomplete to quarantine it non-destructively."
                )
            q = quarantine_incomplete(root, dataset_id, quarantine_root)
            print(f"QUARANTINED={q}")

        current = inspect_dataset_state(root, dataset_id)
        if current["status"] == "LOCALLY_USABLE_FOR_ADAPTER_PREFLIGHT":
            print(f"{dataset_id}: existing local dataset passes preflight; no redownload.")
            results.append(current)
            continue

        print(f"Downloading official WILDS dataset: {dataset_id} version={spec['version']}")
        try:
            get_dataset(
                dataset=dataset_id,
                version=spec["version"],
                root_dir=str(root),
                download=True,
            )
        except Exception as e:
            after = inspect_dataset_state(root, dataset_id)
            raise RuntimeError(
                f"{dataset_id}: WILDS download/initialization failed. "
                f"State={json.dumps(after, indent=2)}"
            ) from e

        after = inspect_dataset_state(root, dataset_id)
        if after["status"] != "LOCALLY_USABLE_FOR_ADAPTER_PREFLIGHT":
            raise RuntimeError(
                f"{dataset_id}: initialization returned without usable metadata/image directory. "
                f"State={json.dumps(after, indent=2)}"
            )
        print(json.dumps({"validated": after}, indent=2))
        results.append(after)

    print(json.dumps({"complete": True, "datasets": results}, indent=2))

if __name__ == "__main__":
    main()
