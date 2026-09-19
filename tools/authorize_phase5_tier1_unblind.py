from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

AUTHORITY_FILES=[
    "configs/phase5_protocol.yaml",
    "configs/nature_external_validation_sap_v3.yaml",
    "docs/governance/nature/NATURE_TARGET_EXTERNAL_VALIDATION_SAP_V3.md",
    "docs/phases/phase5/PHASE5_SCIENTIFIC_FREEZE.md",
    "docs/phases/phase5/PHASE5_TIER1_DATASET_ADAPTER_CONTRACT.md",
    "docs/phases/phase5/PHASE5_RUNNER_DESIGN_CONTRACT.md",
    "src/frontier/phase5_featurebank.py",
    "src/frontier/phase5_wilds_adapters.py",
    "src/frontier/phase5_domain_tools.py",
    "src/frontier/phase5_summary_geometry.py",
    "src/frontier/phase5_pairing.py",
    "src/frontier/phase5_eval.py",
    "src/frontier/phase5_gates.py",
    "src/frontier/phase5_reporting.py",
    "tools/run_phase5.py",
    "tests/test_phase5_core.py",
]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def git_head() -> str:
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()

def tracked_clean(rel: str) -> bool:
    a=subprocess.run(["git","ls-files","--error-unmatch","--",rel],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if a.returncode != 0:
        return False
    b=subprocess.run(["git","diff","--quiet","HEAD","--",rel],cwd=ROOT)
    return b.returncode == 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default=str(ROOT/"configs"/"phase5_tier1_unblind_authorization.json"))
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():
        raise RuntimeError(f"Refusing overwrite: {out}")

    missing=[x for x in AUTHORITY_FILES if not (ROOT/x).exists()]
    if missing:
        raise RuntimeError(f"Missing authority files: {missing}")
    dirty=[x for x in AUTHORITY_FILES if not tracked_clean(x)]
    if dirty:
        raise RuntimeError("Unblind blocked: authority files must be tracked and clean in HEAD: "+", ".join(dirty))

    protocol=(ROOT/"configs"/"phase5_protocol.yaml").read_text(encoding="utf-8")
    if "1.0.0-FROZEN" not in protocol or "FROZEN_BEFORE_TIER1_OUTCOME_ACCESS" not in protocol:
        raise RuntimeError("Unblind blocked: Phase5 protocol not frozen")

    payload={
        "freeze_type":"PHASE5_TIER1_OUTCOME_UNBLIND_AUTHORIZATION",
        "status":"AUTHORIZED_AFTER_PHASE5_FREEZE",
        "code_authority_git_head":git_head(),
        "tier1_datasets":["camelyon17_v1.0","rxrx1_v1.0","iwildcam_v2.0"],
        "authority_sha256":{x:sha256(ROOT/x) for x in AUTHORITY_FILES},
        "external_outcome_access_now_authorized":True,
        "note":"Authorization is prospective and must be committed before loading Tier-1 labels/images.",
    }
    canon=json.dumps(payload,sort_keys=True,separators=(",",":")).encode()
    payload["payload_sha256"]=hashlib.sha256(canon).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()
