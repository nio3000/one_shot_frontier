from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "rxrx1": {
        "version": "1.0",
        "folder": "rxrx1_v1.0",
        "url": "https://worksheets.codalab.org/rest/bundles/0x6b7a05a3056a434498f0bb1252eb8440/contents/blob/",
        "compressed_size": 7_413_123_845,
        "required_dir": "images",
    },
    "iwildcam": {
        "version": "2.0",
        "folder": "iwildcam_v2.0",
        "url": "https://worksheets.codalab.org/rest/bundles/0x6313da2b204647e79a14b468131fcd64/contents/blob/",
        "compressed_size": 11_957_420_032,
        "required_dir": "train",
    },
}

def _authorization_guard() -> None:
    p = ROOT / "configs" / "phase5_tier1_unblind_authorization.json"
    if not p.exists():
        raise RuntimeError("Tier-1 download blocked: unblind authorization artifact missing")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("status") != "AUTHORIZED_AFTER_PHASE5_FREEZE":
        raise RuntimeError("Tier-1 download blocked: authorization status invalid")

def _curl() -> str:
    exe = shutil.which("curl.exe") or shutil.which("curl")
    if not exe:
        raise RuntimeError("curl/curl.exe not found. Windows 10/11 normally includes curl.exe.")
    return exe

def _download_resume(url: str, archive: Path, expected_size: int) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    before = archive.stat().st_size if archive.exists() else 0
    print(json.dumps({
        "archive": str(archive),
        "existing_bytes": before,
        "expected_bytes": expected_size,
        "resume_fraction": before / expected_size if expected_size else None,
    }, indent=2))

    cmd = [
        _curl(),
        "--location",
        "--fail",
        "--show-error",
        "--retry", "50",
        "--retry-all-errors",
        "--retry-delay", "5",
        "--connect-timeout", "30",
        "--speed-time", "120",
        "--speed-limit", "1024",
        "--continue-at", "-",
        "--output", str(archive),
        url,
    ]
    print("RUN:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        current = archive.stat().st_size if archive.exists() else 0
        raise RuntimeError(
            f"Resumable transfer stopped with curl exit={rc}. "
            f"Partial archive is preserved at {archive} "
            f"({current}/{expected_size} bytes). Re-run the same command to resume."
        )

    actual = archive.stat().st_size
    if actual != expected_size:
        raise RuntimeError(
            f"Archive size mismatch after transfer: actual={actual}, expected={expected_size}. "
            "Do not extract. Re-run to resume if smaller; quarantine if larger."
        )

def _extract_and_validate(dataset_id: str, root: Path, keep_archive: bool) -> None:
    from wilds.datasets.download_utils import extract_archive
    from wilds import get_dataset

    spec = DATASETS[dataset_id]
    d = root / spec["folder"]
    archive = d / "archive.tar.gz"

    metadata = d / "metadata.csv"
    required = d / spec["required_dir"]
    if metadata.exists() and required.exists():
        print(f"{dataset_id}: metadata/image directory already present; validating constructor.")
    else:
        print(f"{dataset_id}: extracting verified-size archive.")
        extract_archive(str(archive), str(d), remove_finished=False)

    ds = get_dataset(
        dataset=dataset_id,
        version=spec["version"],
        root_dir=str(root),
        download=False,
    )
    if not metadata.exists() or not required.exists():
        raise RuntimeError(f"{dataset_id}: constructor returned but required files are missing")

    print(json.dumps({
        "dataset_id": dataset_id,
        "version": spec["version"],
        "status": "WILDS_CONSTRUCTOR_VALIDATED",
        "n_samples": len(ds),
        "metadata_csv": str(metadata),
        "required_image_dir": str(required),
        "archive_bytes": archive.stat().st_size if archive.exists() else None,
    }, indent=2))

    if not keep_archive and archive.exists():
        archive.unlink()
        print(f"REMOVED_VERIFIED_ARCHIVE={archive}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "phase5_tier1_raw"))
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), required=True)
    ap.add_argument("--keep-archive", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    _authorization_guard()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for dataset_id in args.datasets:
        spec = DATASETS[dataset_id]
        d = root / spec["folder"]
        archive = d / "archive.tar.gz"

        if (d/"metadata.csv").exists() and (d/spec["required_dir"]).exists():
            if not args.download_only:
                _extract_and_validate(dataset_id, root, args.keep_archive)
            else:
                print(f"{dataset_id}: already extracted; download-only has nothing to do.")
            continue

        _download_resume(spec["url"], archive, int(spec["compressed_size"]))
        if not args.download_only:
            _extract_and_validate(dataset_id, root, args.keep_archive)

    print(json.dumps({"complete": True, "datasets": args.datasets}, indent=2))

if __name__ == "__main__":
    main()
