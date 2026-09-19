from __future__ import annotations
import argparse, gzip, hashlib, json, tarfile
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-images", type=int, default=32)
    a = ap.parse_args()

    archive = Path(a.archive)
    out = Path(a.out_dir)
    (out/"metadata").mkdir(parents=True, exist_ok=True)
    (out/"sample_images").mkdir(parents=True, exist_ok=True)

    members, extracted = [], []
    image_count = 0
    truncated, err = False, None

    try:
        with tarfile.open(archive, "r|gz") as tf:
            for m in tf:
                members.append(m.name)
                if not m.isfile():
                    continue
                low = m.name.lower()
                target = None
                if low.endswith("metadata.csv") or low.endswith("categories.csv") or "release_v" in low:
                    target = out/"metadata"/Path(m.name).name
                elif low.endswith((".png", ".jpg", ".jpeg")) and image_count < a.max_images:
                    target = out/"sample_images"/Path(m.name).name
                    image_count += 1
                if target is not None:
                    f = tf.extractfile(m)
                    if f is not None:
                        target.write_bytes(f.read())
                        extracted.append({"member": m.name, "path": str(target), "bytes": target.stat().st_size})
    except (EOFError, gzip.BadGzipFile, tarfile.ReadError, OSError) as e:
        truncated, err = True, repr(e)

    report = {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "truncated": truncated,
        "error": err,
        "members_seen": len(members),
        "first_members": members[:30],
        "last_members": members[-30:],
        "extracted": extracted,
        "metadata_csv_recovered": any(Path(x["path"]).name == "metadata.csv" for x in extracted),
        "sample_images_recovered": sum(Path(x["path"]).suffix.lower() in {".png",".jpg",".jpeg"} for x in extracted),
    }
    (out/"partial_archive_recovery_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out/"members_seen.txt").write_text("\n".join(members), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
