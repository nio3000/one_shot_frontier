#!/usr/bin/env python3
"""
Selective extraction of the frozen Phase-5 iWildCam-WILDS v2.0 authority universe
from the official WILDS/CodaLab root archive.

Scientific boundary:
- authority universe comes ONLY from metadata.csv filenames
- extra official train images are ignored
- no relabeling / no outcome-based filtering
- archive is processed sequentially in archive order
- extraction is resumable and atomic per file
- duplicate/missing authority members are hard failures
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

try:
    from PIL import Image
except Exception:
    Image = None


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_authority(metadata: Path) -> list[str]:
    with metadata.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "filename" not in rows[0]:
        raise RuntimeError("metadata.csv has no filename column")
    names = [str(r["filename"]) for r in rows]
    if any(not x for x in names):
        raise RuntimeError("empty filename in metadata.csv")
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate filename in metadata.csv")
    return names


def verify_image(path: Path) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for --verify-images")
    with Image.open(path) as im:
        im.verify()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--expected-archive-sha256", default=None)
    ap.add_argument("--verify-images", action="store_true")
    ap.add_argument("--log-every", type=int, default=5000)
    args = ap.parse_args()

    archive = Path(args.archive).resolve()
    metadata = Path(args.metadata).resolve()
    out_root = Path(args.out_root).resolve()
    train_out = out_root / "train"
    report = Path(args.report).resolve()

    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not metadata.is_file():
        raise FileNotFoundError(metadata)

    authority = load_authority(metadata)
    authority_set = set(authority)
    train_out.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    archive_sha = sha256_file(archive)
    if args.expected_archive_sha256:
        exp = args.expected_archive_sha256.lower()
        if archive_sha.lower() != exp:
            raise RuntimeError(
                f"archive SHA256 mismatch: got {archive_sha}, expected {exp}"
            )

    seen_authority: set[str] = set()
    extra_train = 0
    root_files: list[str] = []
    extracted = 0
    already_valid = 0
    verified = 0
    train_files = 0

    # Sequential archive-order pass: avoids catastrophic random seeking in gzip.
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue

            p = PurePosixPath(m.name)
            is_train = len(p.parts) >= 2 and p.parts[-2] == "train"

            if not is_train:
                root_files.append(m.name)
                continue

            train_files += 1
            fn = p.name

            if fn not in authority_set:
                extra_train += 1
                continue

            if fn in seen_authority:
                raise RuntimeError(f"duplicate authority member in archive: {fn}")
            seen_authority.add(fn)

            dest = train_out / fn
            if dest.is_file() and dest.stat().st_size > 0:
                if args.verify_images:
                    verify_image(dest)
                    verified += 1
                already_valid += 1
            else:
                src = tf.extractfile(m)
                if src is None:
                    raise RuntimeError(f"cannot read archive member: {m.name}")

                tmp = dest.with_suffix(dest.suffix + ".part")
                try:
                    with tmp.open("wb") as out:
                        shutil.copyfileobj(src, out, length=1024 * 1024)
                    os.replace(tmp, dest)
                finally:
                    if tmp.exists():
                        tmp.unlink(missing_ok=True)

                if args.verify_images:
                    verify_image(dest)
                    verified += 1
                extracted += 1

            done = len(seen_authority)
            if args.log_every > 0 and (done % args.log_every == 0):
                print(
                    f"[authority {done}/{len(authority)}] extracted={extracted} "
                    f"already_valid={already_valid} verified={verified} "
                    f"train_seen={train_files}",
                    flush=True,
                )

    missing_in_archive = sorted(authority_set - seen_authority)

    present = {p.name for p in train_out.glob("*.jpg") if p.is_file()}
    missing_final = sorted(authority_set - present)
    extra_final = sorted(present - authority_set)

    if args.verify_images:
        # Verify all final authority images, including any that may not have been reached
        # because a future implementation changes traversal behavior.
        for i, fn in enumerate(authority, 1):
            verify_image(train_out / fn)
            if i % args.log_every == 0 or i == len(authority):
                print(f"[final verify {i}/{len(authority)}]", flush=True)
        verified = len(authority)

    passed = (
        len(seen_authority) == len(authority)
        and not missing_in_archive
        and len(present) == len(authority)
        and not missing_final
        and not extra_final
    )

    payload = {
        "decision": (
            "IWILDCAM_WILDS_V2_OFFICIAL_ROOT_SELECTIVE_EXTRACTION_PASS"
            if passed
            else "IWILDCAM_WILDS_V2_OFFICIAL_ROOT_SELECTIVE_EXTRACTION_FAIL"
        ),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "metadata": str(metadata),
        "authority_count": len(authority),
        "archive_train_files": train_files,
        "archive_authority_members": len(seen_authority),
        "archive_extra_train_members": extra_train,
        "archive_root_files": root_files,
        "extracted_now": extracted,
        "already_valid": already_valid,
        "verified_images": verified if args.verify_images else None,
        "missing_in_archive": len(missing_in_archive),
        "final_present_jpg": len(present),
        "final_missing": len(missing_final),
        "final_extra": len(extra_final),
        "missing_in_archive_examples": missing_in_archive[:20],
        "missing_final_examples": missing_final[:20],
        "extra_final_examples": extra_final[:20],
        "out_root": str(out_root),
    }

    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
