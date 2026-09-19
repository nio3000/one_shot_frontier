from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",default=str(ROOT/"runs"/"phase5"))
    ap.add_argument("--out-dir",default=str(ROOT/"evidence"/"phase5"))
    args=ap.parse_args()
    run=Path(args.run_dir); out=Path(args.out_dir)
    required=["run_summary.json","gate_summary.json","pair_results.csv","risk_curves.csv","environment_manifest.json","protocol_snapshot.yaml"]
    for f in required:
        if not (run/f).exists(): raise FileNotFoundError(run/f)
    summary=json.loads((run/"run_summary.json").read_text(encoding="utf-8"))
    gate=json.loads((run/"gate_summary.json").read_text(encoding="utf-8"))
    if not summary.get("formal") or not summary.get("complete"):
        raise RuntimeError("Evidence freeze blocked: no complete formal Tier1 run")
    out.mkdir(parents=True,exist_ok=True)
    for f in required:
        shutil.copy2(run/f,out/f)
    reports=sorted((ROOT/"reports"/"phase5").glob("*_phase5_tier1-formal_experiment_report.md"))
    if not reports: raise RuntimeError("Formal report not found")
    shutil.copy2(reports[-1],out/"formal_experiment_report.md")
    files={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    payload={
        "freeze_type":"PHASE5_TIER1_FORMAL_EVIDENCE",
        "status":"FROZEN",
        "scientific_decision":gate["decision"],
        "code_git_head":summary["git_head"],
        "protocol_sha256":summary["protocol_sha256"],
        "feature_manifest_sha256":summary["feature_manifest_sha256"],
        "pair_manifest_sha256":summary["pair_manifest_sha256"],
        "authorization_sha256":summary["authorization_sha256"],
        "files":files,
    }
    canon=json.dumps(payload,sort_keys=True,separators=(",",":")).encode()
    payload["payload_sha256"]=hashlib.sha256(canon).hexdigest()
    (out/"evidence_manifest.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()
