from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",default=str(ROOT/"runs"/"phase2r"))
    ap.add_argument("--out",default=str(ROOT/"configs"/"phase2r_winner_freeze.json"))
    args=ap.parse_args()

    run=Path(args.run_dir)
    gate=json.loads((run/"gate_summary.json").read_text(encoding="utf-8"))
    summary=json.loads((run/"run_summary.json").read_text(encoding="utf-8"))
    if gate.get("decision") != "SELECTOR_OBJECTIVE_REPAIR_SUPPORTED":
        raise RuntimeError("Winner freeze blocked: Phase 2-R did not reach Decision A")
    winner=gate.get("winner_objective_id")
    if not winner:
        raise RuntimeError("Winner freeze blocked: winner objective missing")

    payload={
        "freeze_type":"PHASE2R_SELECTOR_WINNER",
        "status":"FROZEN_FOR_EXTERNAL_VALIDATION",
        "winner_objective_id":winner,
        "development_git_head":summary["git_head"],
        "protocol_sha256":summary["protocol_sha256"],
        "development_manifest_sha256":summary["manifest_sha256"],
        "nature_external_plan_sha256":summary["external_plan_sha256"],
        "run_summary_sha256":sha256(run/"run_summary.json"),
        "gate_summary_sha256":sha256(run/"gate_summary.json"),
        "objective_results_sha256":sha256(run/"objective_results.csv"),
        "objective_curves_sha256":sha256(run/"objective_curves.csv"),
        "winner_metrics":gate["objective_metrics"][winner],
        "external_outcome_access_now_authorized":True,
    }
    canonical=json.dumps(payload,sort_keys=True,separators=(",",":")).encode()
    payload["winner_payload_sha256"]=hashlib.sha256(canonical).hexdigest()

    out=Path(args.out)
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite existing winner freeze: {out}")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
