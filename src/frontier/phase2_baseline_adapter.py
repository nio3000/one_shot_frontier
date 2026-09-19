from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess


SCHEMA_VERSION = "phase2-external-baseline-v1"


@dataclass(frozen=True)
class AdapterQualification:
    method_id: str
    qualified: bool
    reasons: tuple[str, ...]


def validate_adapter_entry(entry: dict) -> AdapterQualification:
    reasons: list[str] = []
    if entry.get("execution_track") != "model_level_external":
        reasons.append("execution_track must be model_level_external")
    if not entry.get("official_repo"):
        reasons.append("official_repo missing")
    if not entry.get("pinned_commit"):
        reasons.append("pinned_commit missing")
    if not entry.get("license_verified"):
        reasons.append("license not verified")
    if not entry.get("strict_one_shot_verified"):
        reasons.append("strict one-shot semantics not verified")
    if not entry.get("environment_verified"):
        reasons.append("environment not verified")
    if not entry.get("wrapper_command"):
        reasons.append("wrapper_command missing")
    qualified = bool(entry.get("qualified")) and not reasons
    return AdapterQualification(str(entry.get("method_id", "UNKNOWN")), qualified, tuple(reasons))


def validate_external_manifest(manifest: dict, require_frozen: bool = False) -> list[AdapterQualification]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported external baseline manifest schema: {manifest.get('schema_version')}")
    if require_frozen and manifest.get("status") != "FROZEN":
        raise ValueError("External baseline manifest is not FROZEN")
    quals = [validate_adapter_entry(e) for e in manifest.get("adapters", [])]
    if require_frozen and not all(q.qualified for q in quals):
        raise ValueError("One or more external baseline adapters are not qualified")
    return quals


def run_external_adapter(entry: dict, input_json: Path, output_json: Path, cwd: Path | None = None) -> dict:
    q = validate_adapter_entry(entry)
    if not q.qualified:
        raise RuntimeError(f"Adapter {q.method_id} is not qualified: {', '.join(q.reasons)}")
    cmd = str(entry["wrapper_command"]).format(input_json=str(input_json), output_json=str(output_json))
    subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None, check=True)
    if not output_json.exists():
        raise RuntimeError(f"Adapter {q.method_id} did not produce {output_json}")
    obj = json.loads(output_json.read_text(encoding="utf-8"))
    required = ["method_id", "status", "balanced_accuracy", "communication_upload_bytes", "one_shot_rounds", "method_git_commit"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise RuntimeError(f"Adapter output missing fields: {missing}")
    if int(obj["one_shot_rounds"]) != 1:
        raise RuntimeError("External baseline violates one-shot rounds=1 contract")
    return obj
