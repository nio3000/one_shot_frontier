from __future__ import annotations
import argparse, gzip, json, tarfile
from pathlib import Path, PurePosixPath

def safe_rel(name: str) -> Path | None:
    p = PurePosixPath(name.replace("\\","/").lstrip("./"))
    if ".." in p.parts:
        return None
    if not p.parts or p.parts[0] != "images":
        return None
    if not str(p).lower().endswith(".png"):
        return None
    return Path(*p.parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--staging-root", required=True)
    ap.add_argument("--metadata", required=True)
    a = ap.parse_args()

    archive = Path(a.archive)
    staging = Path(a.staging_root)
    staging.mkdir(parents=True, exist_ok=True)

    # Preserve the exact authority metadata in the reconstructed root.
    md_src = Path(a.metadata)
    md_dst = staging / "metadata.csv"
    md_dst.write_bytes(md_src.read_bytes())

    n = 0
    truncated = False
    err = None
    try:
        with tarfile.open(archive, "r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                rel = safe_rel(m.name)
                if rel is None:
                    continue
                target = staging / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                target.write_bytes(data)
                n += 1
    except (EOFError, gzip.BadGzipFile, tarfile.ReadError, OSError) as e:
        truncated, err = True, repr(e)

    report = {
        "archive": str(archive),
        "staging_root": str(staging),
        "metadata": str(md_dst),
        "images_extracted": n,
        "truncated_archive": truncated,
        "stream_error": err,
    }
    out = staging / "_partial_extract_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
