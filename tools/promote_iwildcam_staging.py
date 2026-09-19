from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--staging-root",required=True)
    ap.add_argument("--metadata",required=True)
    ap.add_argument("--categories",required=True)
    ap.add_argument("--audit",required=True)
    ap.add_argument("--target-root",default="data/phase5_tier1_raw/iwildcam_v2.0")
    a=ap.parse_args()
    audit=json.loads(Path(a.audit).read_text(encoding="utf-8"))
    if not audit.get("full_complete"): raise RuntimeError("refusing promotion: audit not complete")
    src=Path(a.staging_root); dst=Path(a.target_root)
    if dst.exists(): raise RuntimeError(f"target exists: {dst}")
    dst.mkdir(parents=True)
    shutil.move(str(src/"train"),str(dst/"train"))
    shutil.copy2(a.metadata,dst/"metadata.csv"); shutil.copy2(a.categories,dst/"categories.csv")
    manifest={"decision":"IWILDCAM_WILDS_V2_FILEWISE_OFFICIAL_ACQUISITION_QUALIFIED",
              "source_bundle":"0x6313da2b204647e79a14b468131fcd64",
              "metadata_sha256":audit["metadata_sha256"],"categories_sha256":audit["categories_sha256"],
              "files":audit["expected_files"],"full_complete":True}
    (dst/"_phase5_acquisition_authority.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
