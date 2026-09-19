from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def canonical_hash(obj:dict)->str:
    payload=json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("output"); args=ap.parse_args()
    obj=json.loads(Path(args.input).read_text(encoding="utf-8"))
    from sys import path
    ROOT=Path(__file__).resolve().parents[1]; path.insert(0,str(ROOT/"src"))
    from frontier.phase2_baseline_adapter import validate_external_manifest
    qs=validate_external_manifest(obj,require_frozen=False)
    if not all(q.qualified for q in qs):
        raise RuntimeError("Cannot freeze: one or more adapters are not qualified")
    obj["status"]="FROZEN"; obj["manifest_payload_sha256"]=canonical_hash({k:v for k,v in obj.items() if k!="manifest_payload_sha256"})
    out=Path(args.output); out.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
    print(out)
if __name__=="__main__": main()
