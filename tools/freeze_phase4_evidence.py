from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "phase4"))
    ap.add_argument("--out-dir", default=str(ROOT / "evidence" / "phase4"))
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out_dir)
    summary_path = run / "run_summary.json"
    gate_path = run / "gate_summary.json"
    if not summary_path.exists() or not gate_path.exists():
        raise RuntimeError("Phase 4 evidence freeze blocked: formal run summary/gate missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mode") != "formal-development" or not summary.get("formal_development") or not summary.get("complete"):
        raise RuntimeError("Phase 4 evidence freeze blocked: run is not complete formal-development evidence")

    required = [
        "run_summary.json",
        "gate_summary.json",
        "pair_diagnostics.csv",
        "risk_curves.csv",
        "environment_manifest.json",
        "protocol_snapshot.yaml",
    ]
    for name in required:
        if not (run / name).exists():
            raise RuntimeError(f"Phase 4 evidence freeze blocked: missing {name}")

    reports = sorted((ROOT / "reports" / "phase4").glob("*_phase4_formal-development_experiment_report.md"))
    if not reports:
        raise RuntimeError("Phase 4 evidence freeze blocked: formal report missing")
    report = reports[-1]

    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty evidence directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in required:
        src = run / name
        dst = out / name
        shutil.copy2(src, dst)
        copied.append(dst)
    report_dst = out / "formal_experiment_report.md"
    shutil.copy2(report, report_dst)
    copied.append(report_dst)

    manifest = {
        "freeze_type": "PHASE4_FORMAL_DEVELOPMENT_EVIDENCE",
        "status": "FROZEN",
        "scientific_decision": json.loads(gate_path.read_text(encoding="utf-8")).get("decision"),
        "code_git_head": summary.get("git_head"),
        "protocol_sha256": summary.get("protocol_sha256"),
        "phase1_manifest_sha256": summary.get("manifest_sha256"),
        "authority_sha256": summary.get("authority_sha256"),
        "files": {p.name: sha256(p) for p in copied},
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    (out / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
