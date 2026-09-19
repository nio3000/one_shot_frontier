from __future__ import annotations
import argparse, json, shutil, subprocess, zipfile
from pathlib import Path

URL = "https://storage.googleapis.com/rxrx/rxrx1/rxrx1-metadata.zip"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/phase5_tier1_recovery/rxrx1_original")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out/"rxrx1-metadata.zip"
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl not found")
    cmd = [curl, "--location", "--fail", "--show-error", "--retry", "20",
           "--retry-all-errors", "--output", str(zpath), URL]
    subprocess.run(cmd, check=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
        names = z.namelist()
    print(json.dumps({"url": URL, "zip": str(zpath), "files": names}, indent=2))

if __name__ == "__main__":
    main()
