from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from frontier.phase2_baseline_adapter import validate_external_manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("manifest"); args=ap.parse_args()
    m=json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    qs=validate_external_manifest(m,require_frozen=False)
    out={q.method_id:{"qualified":q.qualified,"reasons":list(q.reasons)} for q in qs}
    print(json.dumps(out,indent=2))
    if not all(q.qualified for q in qs): raise SystemExit(2)
if __name__=="__main__": main()
