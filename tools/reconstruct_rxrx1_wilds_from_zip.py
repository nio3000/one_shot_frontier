from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def expected_raw_member(row, ch: int) -> str:
    return (
        f"rxrx1/images/{row.experiment}/Plate{int(row.plate)}/"
        f"{row.well}_s{int(row.site)}_w{ch}.png"
    )


def expected_out_rel(row) -> Path:
    return (
        Path("images")
        / str(row.experiment)
        / f"Plate{int(row.plate)}"
        / f"{row.well}_s{int(row.site)}.png"
    )


def read_gray_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    try:
        data = zf.read(member)
    except KeyError as e:
        raise FileNotFoundError(f"Missing ZIP member: {member}") from e

    with Image.open(io.BytesIO(data)) as im:
        arr = np.asarray(im)

    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale source channel, got {arr.shape}: {member}")
    if arr.shape[0] < 256 or arr.shape[1] < 256:
        raise ValueError(f"Source smaller than 256x256: {arr.shape}: {member}")
    return arr


def reconstruct_rgb(zf: zipfile.ZipFile, row) -> np.ndarray:
    chans = []
    for ch in (1, 2, 3):
        member = expected_raw_member(row, ch)
        arr = read_gray_from_zip(zf, member)
        h, w = arr.shape
        y0 = (h - 256) // 2
        x0 = (w - 256) // 2
        crop = arr[y0:y0 + 256, x0:x0 + 256]
        if crop.shape != (256, 256):
            raise ValueError(f"Unexpected crop shape {crop.shape}: {member}")
        chans.append(crop)
    return np.stack(chans, axis=-1).astype(np.uint8, copy=False)


def existing_is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            return im.mode == "RGB" and im.size == (256, 256)
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--report", default="reports/phase5/validation/rxrx1_reconstruction_run.json")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows; use a small value for smoke")
    ap.add_argument("--overwrite-invalid", action="store_true")
    args = ap.parse_args()

    zpath = Path(args.zip)
    mpath = Path(args.metadata)
    out_root = Path(args.out_root)
    report_path = Path(args.report)

    if not zpath.is_file():
        raise FileNotFoundError(zpath)
    if not mpath.is_file():
        raise FileNotFoundError(mpath)

    df = pd.read_csv(mpath)
    required = {"experiment", "plate", "well", "site"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"Metadata missing columns: {missing_cols}")

    if args.limit > 0:
        df = df.iloc[: args.limit].copy()

    out_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    created = 0
    skipped = 0
    invalid_existing = 0
    failures = []

    with zipfile.ZipFile(zpath, "r") as zf:
        members = set(zf.namelist())

        # Fail closed before doing substantial work: every requested W1/W2/W3 member must exist.
        wanted = []
        for row in df.itertuples(index=False):
            for ch in (1, 2, 3):
                wanted.append(expected_raw_member(row, ch))
        missing_members = [x for x in wanted if x not in members]
        if missing_members:
            sample = missing_members[:20]
            raise RuntimeError(
                f"Missing {len(missing_members)} required source members. First entries: {sample}"
            )

        total = len(df)
        for i, row in enumerate(df.itertuples(index=False), start=1):
            rel = expected_out_rel(row)
            dst = out_root / rel

            if dst.exists():
                if existing_is_valid(dst):
                    skipped += 1
                    if args.progress_every and i % args.progress_every == 0:
                        elapsed = time.time() - t0
                        print(f"[{i}/{total}] created={created} skipped={skipped} elapsed={elapsed:.1f}s", flush=True)
                    continue
                invalid_existing += 1
                if not args.overwrite_invalid:
                    failures.append({"row": i, "path": str(dst), "error": "existing output invalid"})
                    break

            try:
                arr = reconstruct_rgb(zf, row)
                dst.parent.mkdir(parents=True, exist_ok=True)
                # PNG compression does not alter pixels; optimize=False keeps the write path simple.
                Image.fromarray(arr, mode="RGB").save(dst, format="PNG", optimize=False)
                created += 1
            except Exception as e:
                failures.append({"row": i, "path": str(dst), "error": repr(e)})
                break

            if args.progress_every and i % args.progress_every == 0:
                elapsed = time.time() - t0
                print(f"[{i}/{total}] created={created} skipped={skipped} elapsed={elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    report = {
        "zip": str(zpath),
        "metadata": str(mpath),
        "out_root": str(out_root),
        "rows_requested": int(len(df)),
        "created": int(created),
        "skipped_valid_existing": int(skipped),
        "invalid_existing_seen": int(invalid_existing),
        "failures": failures,
        "elapsed_seconds": elapsed,
        "complete": (not failures and created + skipped == len(df)),
        "rule": {
            "channels": [1, 2, 3],
            "crop": "center 256x256",
            "output": "RGB uint8 PNG",
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not report["complete"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
