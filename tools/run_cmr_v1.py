"""CMR-V1 formal runner.

Modes:

- ``--mode x1``   frozen X1 exact kernel-prototype collision (frozen authority);
- ``--mode feature-map``  freeze the eight Gaussian-RBF maps + feature-bank provenance
  (computed before X2 outcome access);
- ``--mode extract``      extract raw frozen layer features for one or all conditions;
- ``--mode x2-dry``       implementation dry validation on a small slice (no formal output);
- ``--mode x2``           the single formal X2 run (opens X2 outcome access).

Frozen authority:

- ``docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md``;
- ``configs/cross_mechanism_replication_protocol_v1.yaml``;
- ``docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv``;
- ``docs/phases/cmr_v1/CMR_V1_FREEZE_MANIFEST.json``.

X1 formal outputs:

- ``runs/cmr_v1/x1/run_summary.json``      (byte-stable scientific output);
- ``runs/cmr_v1/x1/rerun_audit.json``      (execution-history audit);
- ``evidence/cmr_v1/x1/{gate_summary,protocol_snapshot,evidence_manifest}``.

X2 formal outputs:

- ``configs/cmr_v1_feature_map_manifest.json``;
- ``configs/cmr_v1_feature_bank_manifest.json``;
- ``runs/cmr_v1/x2/{condition_results.csv,run_summary.json}``;
- ``evidence/cmr_v1/x2/{gate_summary,evidence_manifest,protocol_snapshot}``.

X3-X5 are never implemented and never authorized. Exit codes: 0 = stage PASS,
1 = stage implementation/decision block (artifacts preserved), 2 = local guard
failure (baseline mismatch / freeze-manifest hash mismatch / bad CLI usage).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.cmr_kernel_layer import (
    FLOAT_DTYPE_NAME,
    PROTOCOL_ID,
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
    X2_ACTION_RELEVANT_SPREAD_MIN,
    X2_AGGREGATION_TOLERANCE,
    X2_CONDITIONS,
    X2_DATASETS,
    X2_ENCODER_LAYERS,
    X2_RFF_DIMENSION,
    X2_RFF_MASTER_SEED,
    X2_RFF_SIGMA,
    X2_UNIQUE_BEST_MARGIN_MIN,
    aggregate_client_messages,
    apply_rff,
    array_sha256,
    bacc_from_predictions,
    canonical_dumps,
    class_message,
    evaluate_x1_exact_collision,
    evaluate_x2_gates,
    generate_rff,
    load_x2_protocol_constants,
    normalize_layer_features,
    predict_prototype_batch,
    reconstruct_prototypes,
    rff_seed,
    summarize_x2_condition,
)

from frontier.cmr_feature_extract import X2_RAW_LAYER_DIM

EXPECTED_BASELINE = "d409ee2e1009a876c773c48687c7d72ac6ed4102"

PROTOCOL_MD = ROOT / "docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md"
PROTOCOL_YAML = ROOT / "configs/cross_mechanism_replication_protocol_v1.yaml"
GATE_MATRIX = ROOT / "docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv"
FREEZE_MANIFEST = ROOT / "docs/phases/cmr_v1/CMR_V1_FREEZE_MANIFEST.json"
IMPL_PROMPT = ROOT / "docs/phases/cmr_v1/CMR_V1_CODEX_X1_IMPLEMENTATION_PROMPT.md"

MODULE_SRC = ROOT / "src/frontier/cmr_kernel_layer.py"
RUNNER_SRC = ROOT / "tools/run_cmr_v1.py"
TEST_X1 = ROOT / "tests/test_cmr_v1_x1.py"
TEST_PROTOCOL = ROOT / "tests/test_cmr_v1_protocol.py"

RUN_DIR = ROOT / "runs/cmr_v1/x1"
RUN_SUMMARY = RUN_DIR / "run_summary.json"
RERUN_AUDIT = RUN_DIR / "rerun_audit.json"

EVIDENCE_DIR = ROOT / "evidence/cmr_v1/x1"
GATE_SUMMARY = EVIDENCE_DIR / "gate_summary.json"
PROTOCOL_SNAPSHOT = EVIDENCE_DIR / "protocol_snapshot.yaml"
EVIDENCE_MANIFEST = EVIDENCE_DIR / "evidence_manifest.json"

FROZEN_FILE_EXPECTATIONS = (
    ("CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.0.md", PROTOCOL_MD),
    ("cross_mechanism_replication_protocol_v1.yaml", PROTOCOL_YAML),
    ("CMR_V1_GATE_MATRIX.csv", GATE_MATRIX),
    ("CMR_V1_CODEX_X1_IMPLEMENTATION_PROMPT.md", IMPL_PROMPT),
)

DECISION_PASS = "X1_PASS_EXACT_KERNEL_PROTOTYPE_LAYER_COLLISION"
DECISION_BLOCKED = "X1_IMPLEMENTATION_BLOCKED"

# --- X2 paths (data/ is repository-excluded scratch; never a formal deliverable) ---
FEATURE_MAP_MANIFEST = ROOT / "configs/cmr_v1_feature_map_manifest.json"
FEATURE_BANK_MANIFEST = ROOT / "configs/cmr_v1_feature_bank_manifest.json"

X2_RUN_DIR = ROOT / "runs/cmr_v1/x2"
X2_CONDITION_RESULTS = X2_RUN_DIR / "condition_results.csv"
X2_RUN_SUMMARY = X2_RUN_DIR / "run_summary.json"

X2_EVIDENCE_DIR = ROOT / "evidence/cmr_v1/x2"
X2_GATE_SUMMARY = X2_EVIDENCE_DIR / "gate_summary.json"
X2_PROTOCOL_SNAPSHOT = X2_EVIDENCE_DIR / "protocol_snapshot.yaml"
X2_EVIDENCE_MANIFEST = X2_EVIDENCE_DIR / "evidence_manifest.json"

X2_RAW_ROOT = ROOT / "data/cmr_v1/x2/raw"
X2_FEATURE_ROOT = ROOT / "data/cmr_v1/x2/features"

# Inference backend is an implementation choice, never a scientific axis. The
# CPU path stays the default; `--device cuda` selects the CUDA feature cache.
CUDA_FEATURE_ROOT = ROOT / "data/cmr_v1/x2_cuda/features"
CUDA_RAW_ROOT = ROOT / "data/cmr_v1/x2_cuda/raw"
INFERENCE_BACKENDS = ("cpu", "cuda")

FEATURE_EXTRACT_SRC = ROOT / "src/frontier/cmr_feature_extract.py"
TEST_X2 = ROOT / "tests/test_cmr_v1_x2.py"

X2_SOURCE_FILES = (MODULE_SRC, RUNNER_SRC, FEATURE_EXTRACT_SRC, TEST_X2, TEST_X1, TEST_PROTOCOL)

CONDITION_CSV_FIELDS = (
    "dataset",
    "encoder",
    "BACC_action_1",
    "BACC_action_2",
    "BACC_action_3",
    "BACC_action_4",
    "best_layer",
    "second_best_layer",
    "best_margin",
    "layer_spread",
    "action_relevant",
    "unique_best",
    "aggregation_recovery_max_abs",
    "finite_pass",
)

X2_DRY_DATASETS = ("cifar100", "pathmnist")
X2_DRY_MAX_RECORDS = 320
X2_DRY_MAX_TEST_RECORDS = 640

# Engineering extraction batch sizes per backend (never selected by model risk).
BACKEND_BATCH_SIZE = {"cpu": 128, "cuda": 128}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "NO_GIT_HEAD"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def check_baseline(expected: str) -> tuple[str, bool | None]:
    head = git_head()
    if expected == "":
        print("[CMR-V1 X1] WARNING: baseline check explicitly disabled (--expected-baseline '').")
        return head, None
    if head != expected:
        print("[CMR-V1 X1] BASELINE MISMATCH - formal run not started.")
        print(f"[CMR-V1 X1]   expected: {expected}")
        print(f"[CMR-V1 X1]   actual:   {head}")
        print("[CMR-V1 X1] Report this before rerunning. If a new baseline was")
        print("[CMR-V1 X1] authorized, pass it explicitly via --expected-baseline.")
        return head, False
    return head, True


def load_protocol_authority() -> dict:
    doc = yaml.safe_load(PROTOCOL_YAML.read_text(encoding="utf-8"))
    protocol = doc["protocol"]
    if protocol["id"] != PROTOCOL_ID:
        raise RuntimeError(f"protocol id mismatch: {protocol['id']!r}")
    if protocol["version"] != PROTOCOL_VERSION:
        raise RuntimeError(f"protocol version mismatch: {protocol['version']!r}")
    if protocol["status"] != PROTOCOL_STATUS:
        raise RuntimeError(f"protocol status mismatch: {protocol['status']!r}")

    x1 = doc["x1"]
    checks = (
        (x1["mode"] == "exact_analytic", "x1.mode"),
        (x1["feature_map"] == "[cos(theta), sin(theta)]", "x1.feature_map"),
        (int(x1["class_count"]) == 6, "x1.class_count"),
        (math.isclose(float(x1["a"]), 0.2, rel_tol=0.0, abs_tol=1e-15), "x1.a"),
        (math.isclose(float(x1["q_squared"]), 0.96, rel_tol=0.0, abs_tol=1e-15), "x1.q_squared"),
        (math.isclose(float(x1["expected_bacc_good"]), 1.0), "x1.expected_bacc_good"),
        (math.isclose(float(x1["expected_bacc_bad"]), 1.0 / 3.0), "x1.expected_bacc_bad"),
        (
            math.isclose(float(x1["expected_deterministic_regret"]), 2.0 / 3.0),
            "x1.expected_deterministic_regret",
        ),
        (
            math.isclose(float(x1["expected_randomized_regret"]), 1.0 / 3.0),
            "x1.expected_randomized_regret",
        ),
        (float(x1["numeric_tolerance"]) == 1e-12, "x1.numeric_tolerance"),
        (bool(x1["hard_gate"]) is True, "x1.hard_gate"),
    )
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise RuntimeError(f"frozen x1 protocol constants mismatch: {failed}")
    return {
        "id": protocol["id"],
        "version": protocol["version"],
        "status": protocol["status"],
        "x1": {
            "mode": x1["mode"],
            "feature_map": x1["feature_map"],
            "class_count": int(x1["class_count"]),
            "a": float(x1["a"]),
            "q_squared": float(x1["q_squared"]),
            "expected_bacc_good": float(x1["expected_bacc_good"]),
            "expected_bacc_bad": float(x1["expected_bacc_bad"]),
            "expected_deterministic_regret": float(x1["expected_deterministic_regret"]),
            "expected_randomized_regret": float(x1["expected_randomized_regret"]),
            "numeric_tolerance": float(x1["numeric_tolerance"]),
            "hard_gate": bool(x1["hard_gate"]),
        },
    }


def verify_freeze_manifest() -> dict:
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("status") != PROTOCOL_STATUS
    ):
        raise RuntimeError("freeze manifest identity mismatch")
    files = manifest["files"]
    verified = []
    for name, path in FROZEN_FILE_EXPECTATIONS:
        expected = files[name]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen file hash mismatch for {name}: expected {expected} actual {actual}"
            )
        verified.append(
            {
                "name": name,
                "path": rel(path),
                "sha256": actual,
                "expected_sha256": expected,
                "match": True,
            }
        )
    optional = []
    for name in ("README.md", "CMR_V1_REPOSITORY_LAYOUT.md"):
        expected = files[name]["sha256"]
        resolved = None
        for candidate in (
            ROOT / name,
            ROOT / "docs" / "phases" / "cmr_v1" / name,
            ROOT / "docs" / "governance" / "nature" / name,
            ROOT / "configs" / name,
        ):
            if candidate.exists() and sha256_file(candidate) == expected:
                resolved = candidate
                break
        optional.append(
            {
                "name": name,
                "expected_sha256": expected,
                "resolved": resolved is not None,
                "path": rel(resolved) if resolved is not None else None,
            }
        )
    return {
        "manifest_path": rel(FREEZE_MANIFEST),
        "manifest_sha256": sha256_file(FREEZE_MANIFEST),
        "verified_files": verified,
        "optional_entries": optional,
    }


def run_unit_tests(targets: list[Path] | None = None) -> dict:
    selected = list(targets) if targets is not None else [TEST_X1]
    if targets is None and TEST_PROTOCOL.exists():
        selected.append(TEST_PROTOCOL)
    selected = [target for target in selected if Path(target).exists()]
    rel_targets = [rel(Path(target)) for target in selected]
    command = [sys.executable, "-m", "pytest", *rel_targets, "-q", "-p", "no:cacheprovider"]
    proc = subprocess.run(
        command, cwd=str(ROOT), capture_output=True, text=True, timeout=36000
    )
    stdout = proc.stdout or ""
    passed_tests = None
    failed_tests = None
    match = re.search(r"(\d+) passed", stdout)
    if match:
        passed_tests = int(match.group(1))
    match = re.search(r"(\d+) failed", stdout)
    if match:
        failed_tests = int(match.group(1))
    return {
        "targets": rel_targets,
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "tests_passed": passed_tests,
        "tests_failed": failed_tests,
    }


# =============================================================================
# X2 orchestration
# =============================================================================
def condition_cache_path(dataset: str, encoder: str, device: str = "cpu", root: Path | None = None) -> Path:
    """Cache path for one structural condition under the selected backend root."""
    if root is not None:
        base = Path(root)
    else:
        base = CUDA_FEATURE_ROOT if device == "cuda" else X2_FEATURE_ROOT
    return base / f"{dataset}__{encoder}.npz"


def backend_metadata(device: str) -> dict:
    """Record the inference backend identity used for extraction."""
    import torch

    record = {
        "inference_backend": "CUDA" if device == "cuda" else "CPU",
        "device": device,
        "inference_dtype": "float32",
        "rff_dtype": FLOAT_DTYPE_NAME,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        import torchvision

        record["torchvision_version"] = torchvision.__version__
    except Exception:  # noqa: BLE001 - provenance best effort
        record["torchvision_version"] = "unknown"
    record["cuda_runtime_version"] = torch.version.cuda
    record["cudnn_version"] = torch.backends.cudnn.version()
    if torch.cuda.is_available():
        record["gpu_model"] = torch.cuda.get_device_name(0)
        record["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        try:
            record["driver_version"] = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip().splitlines()[0]
            )
        except Exception:  # noqa: BLE001 - provenance best effort
            record["driver_version"] = "unknown"
        record["tf32_matmul_allowed"] = bool(torch.backends.cuda.matmul.allow_tf32)
        record["tf32_cudnn_allowed"] = bool(torch.backends.cudnn.allow_tf32)
    return record


def rff_array_path(encoder: str, layer: str) -> Path:
    return ROOT / "data/cmr_v1/x2/rff" / f"{encoder}__{layer}.npz"


def _load_extract_module():
    sys.path.insert(0, str(ROOT / "src"))
    from frontier import cmr_feature_extract

    return cmr_feature_extract


def build_feature_map_manifest() -> dict:
    """Freeze the eight Gaussian-RBF maps and write the feature-map manifest.

    Computed entirely from frozen protocol constants, before any X2 BACC exists.
    """
    authority = load_x2_protocol_constants(PROTOCOL_YAML)
    head = git_head()
    code_sha = sha256_file(MODULE_SRC)
    entries = []
    for encoder, layer_tuple in X2_ENCODER_LAYERS.items():
        for layer in layer_tuple:
            # RFF input dimension is the frozen post-stage / post-block layer
            # width (never a pooled or projected dimension).
            input_dim = int(X2_RAW_LAYER_DIM[encoder][layer])
            w, b = generate_rff(encoder, layer, input_dim)
            if w.dtype != np.float64 or b.dtype != np.float64:
                raise RuntimeError("frozen RFF must be generated in float64")
            if w.shape != (X2_RFF_DIMENSION, input_dim) or b.shape != (X2_RFF_DIMENSION,):
                raise RuntimeError("frozen RFF shape mismatch")
            path = rff_array_path(encoder, layer)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with np.load(path) as frozen:
                    if not (
                        np.array_equal(frozen["W"], w) and np.array_equal(frozen["b"], b)
                    ):
                        raise RuntimeError(
                            f"frozen RFF array drift at {rel(path)}: regenerated map differs"
                        )
            else:
                np.savez(path, W=w, b=b)
            with np.load(path) as frozen:
                w_reloaded = np.asarray(frozen["W"], dtype=np.float64)
                b_reloaded = np.asarray(frozen["b"], dtype=np.float64)
            if array_sha256(w_reloaded) != array_sha256(w) or array_sha256(b_reloaded) != array_sha256(b):
                raise RuntimeError("frozen RFF array round-trip is not byte-identical")
            entries.append(
                {
                    "encoder": encoder,
                    "layer": layer,
                    "datasets_using_this_map": list(X2_DATASETS),
                    "input_dimension": int(input_dim),
                    "output_dimension": int(X2_RFF_DIMENSION),
                    "sigma": float(X2_RFF_SIGMA),
                    "master_seed": int(X2_RFF_MASTER_SEED),
                    "seed_derivation_string": f"{X2_RFF_MASTER_SEED}|{encoder}|{layer}",
                    "derived_seed": int(rff_seed(encoder, layer)),
                    "w_sha256": array_sha256(w_reloaded),
                    "b_sha256": array_sha256(b_reloaded),
                    "dtype": FLOAT_DTYPE_NAME,
                    "array_path": rel(path),
                    "generation_code_sha256": code_sha,
                    "git_head": head,
                }
            )
    manifest = {
        "manifest_type": "CMR_V1_FEATURE_MAP_MANIFEST",
        "protocol_id": authority["protocol_id"],
        "protocol_version": authority["protocol_version"],
        "protocol_status": authority["protocol_status"],
        "stage": "X2",
        "git_head": head,
        "rff": {
            "type": "gaussian_rbf_random_fourier_features",
            "formula": "sqrt(2/m) * cos(W u + b)",
            "dimension": int(X2_RFF_DIMENSION),
            "sigma": float(X2_RFF_SIGMA),
            "master_seed": int(X2_RFF_MASTER_SEED),
            "rng": authority["rff"]["rng"],
            "generation_dtype": FLOAT_DTYPE_NAME,
            "seed_derivation": {
                "string_template": "20261001|<encoder_id>|<layer_id>",
                "hash": "sha256",
                "bytes": "first_8",
                "byte_order": "big",
                "modulus": 2**32,
            },
        },
        "generation_code": rel(MODULE_SRC),
        "generation_code_sha256": code_sha,
        "frozen_before_x2_outcome_access": True,
        "tuning_prohibited": [
            "bandwidth",
            "dimension",
            "seed",
            "multi_seed_selection",
            "post_outcome_regeneration",
        ],
        "maps": entries,
    }
    write_json(FEATURE_MAP_MANIFEST, manifest)
    return manifest


def load_frozen_rff(manifest: dict, encoder: str, layer: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the frozen RFF array for a condition layer and verify its identity."""
    matches = [
        entry
        for entry in manifest["maps"]
        if entry["encoder"] == encoder and entry["layer"] == layer
    ]
    if len(matches) != 1:
        raise RuntimeError(f"feature-map manifest must contain exactly one map for {encoder}/{layer}")
    entry = matches[0]
    path = ROOT / entry["array_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as frozen:
        w = np.asarray(frozen["W"], dtype=np.float64)
        b = np.asarray(frozen["b"], dtype=np.float64)
    if array_sha256(w) != entry["w_sha256"] or array_sha256(b) != entry["b_sha256"]:
        raise RuntimeError(f"frozen RFF hash mismatch for {encoder}/{layer}")
    if array_sha256(w) != array_sha256(generate_rff(encoder, layer, entry["input_dimension"])[0]):
        raise RuntimeError(f"frozen RFF array is not reproducible for {encoder}/{layer}")
    return w, b


def resolve_raw_root(device: str, override: Path | None = None) -> Path:
    """Raw dataset root for a backend.

    Dataset identity is recorded per condition, so the authoritative downloaded
    raw data may be shared between backends instead of being duplicated. A
    backend-specific raw root is used only when it already exists.
    """
    if override is not None:
        return Path(override)
    candidate = CUDA_RAW_ROOT if device == "cuda" else X2_RAW_ROOT
    if candidate.is_dir() and any(candidate.iterdir()):
        return candidate
    return X2_RAW_ROOT


def extract_condition_to_cache(
    dataset: str,
    encoder: str,
    batch_size: int,
    num_workers: int,
    device: str,
    allow_download: bool,
    dry: bool = False,
    raw_root: Path | None = None,
    feature_root: Path | None = None,
) -> dict:
    extract = _load_extract_module()
    resolved_raw_root = resolve_raw_root(device, raw_root)
    condition = extract.extract_condition(
        dataset,
        encoder,
        raw_root=resolved_raw_root,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        allow_download=allow_download,
    )
    train, test = condition["train"], condition["test"]
    train_labels, test_labels = condition["train_labels"], condition["test_labels"]
    if dry:
        limit = int(X2_DRY_MAX_RECORDS)
        test_limit = int(X2_DRY_MAX_TEST_RECORDS)
        train = {layer: array[:limit] for layer, array in train.items()}
        test = {layer: array[:test_limit] for layer, array in test.items()}
        train_labels = train_labels[:limit]
        test_labels = test_labels[:test_limit]
    payload = {
        "dataset": np.asarray(dataset),
        "encoder": np.asarray(encoder),
        "train_labels": train_labels,
        "test_labels": test_labels,
        "provenance_json": np.asarray(canonical_dumps(condition["provenance"])),
        "backend_json": np.asarray(canonical_dumps(backend_metadata(device))),
    }
    for layer in X2_ENCODER_LAYERS[encoder]:
        payload[f"train__{layer}"] = train[layer]
        payload[f"test__{layer}"] = test[layer]
    path = condition_cache_path(dataset, encoder, device, feature_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    entry = {
        "condition_id": f"{dataset}__{encoder}",
        "dataset": dataset,
        "encoder": encoder,
        "device": device,
        "cache_path": rel(path),
        "cache_sha256": sha256_file(path),
        "cache_bytes": int(path.stat().st_size),
        "train_records": int(len(train_labels)),
        "test_records": int(len(test_labels)),
        "layers": list(X2_ENCODER_LAYERS[encoder]),
        "provenance": condition["provenance"],
        "backend": backend_metadata(device),
    }
    return entry


def build_feature_bank_manifest(device: str = "cpu", feature_root: Path | None = None) -> dict:
    """Record provenance for every extracted condition from the frozen cache."""
    authority = load_x2_protocol_constants(PROTOCOL_YAML)
    head = git_head()
    code_sha = sha256_file(FEATURE_EXTRACT_SRC)
    backend = backend_metadata(device)
    entries = []
    for dataset, encoder in X2_CONDITIONS:
        path = condition_cache_path(dataset, encoder, device, feature_root)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing extracted condition cache {rel(path)}; run --mode extract first"
            )
        with np.load(path, allow_pickle=False) as payload:
            provenance = json.loads(str(payload["provenance_json"]))
            train_labels = np.asarray(payload["train_labels"], dtype=np.int64)
            test_labels = np.asarray(payload["test_labels"], dtype=np.int64)
            raw_dim = {}
            for layer in X2_ENCODER_LAYERS[encoder]:
                train_layer = payload[f"train__{layer}"]
                test_layer = payload[f"test__{layer}"]
                if train_layer.shape[1] != test_layer.shape[1]:
                    raise RuntimeError(f"{dataset}/{encoder}/{layer} train/test dim mismatch")
                raw_dim[layer] = int(train_layer.shape[1])
                if train_layer.shape[0] != len(train_labels) or test_layer.shape[0] != len(test_labels):
                    raise RuntimeError(f"{dataset}/{encoder}/{layer} record count mismatch")
        entries.append(
            {
                "bank_id": f"{dataset}__{encoder}",
                "dataset": dataset,
                "encoder": encoder,
                "inference_backend": "CUDA" if device == "cuda" else "CPU",
                "device": device,
                "dataset_version_authority": provenance["dataset_source_authority"],
                "split": provenance["split"],
                "train_records": int(len(train_labels)),
                "test_records": int(len(test_labels)),
                "class_count": int(provenance["class_count"]),
                "canonical_sample_id_hash": provenance["canonical_sample_id_hash"],
                "train_class_counts": provenance["train_class_counts"],
                "test_class_counts": provenance["test_class_counts"],
                "pretrained_weights_enum": provenance["pretrained_weights_enum"],
                "pretrained_state_sha256": provenance["encoder_state_sha256"],
                "preprocessing_id": provenance["preprocessing_id"],
                "preprocessing_sha256": provenance["transform_sha256"],
                "layer_ids": list(X2_ENCODER_LAYERS[encoder]),
                "raw_layer_dim": raw_dim,
                "l2_normalization": "applied_before_frozen_rff",
                "storage_dtype": provenance["storage_dtype"],
                "inference_dtype": "float32",
                "rff_dtype": FLOAT_DTYPE_NAME,
                "rff_dimension": int(X2_RFF_DIMENSION),
                "rff_sigma": float(X2_RFF_SIGMA),
                "torch_version": provenance["torch_version"],
                "torchvision_version": provenance["torchvision_version"],
                "cuda_runtime_version": backend["cuda_runtime_version"],
                "cudnn_version": backend["cudnn_version"],
                "gpu_model": backend.get("gpu_model"),
                "driver_version": backend.get("driver_version"),
                "cache_path": rel(path),
                "cache_sha256": sha256_file(path),
                "cache_bytes": int(path.stat().st_size),
                "source_code_sha256": code_sha,
                "git_head": head,
            }
        )
    manifest = {
        "manifest_type": "CMR_V1_FEATURE_BANK_MANIFEST",
        "protocol_id": authority["protocol_id"],
        "protocol_version": authority["protocol_version"],
        "protocol_status": authority["protocol_status"],
        "stage": "X2",
        "git_head": head,
        "inference_backend": "CUDA" if device == "cuda" else "CPU",
        "device": device,
        "backend": backend,
        "cpu_premigration_cache_excluded": True,
        "source_code": rel(FEATURE_EXTRACT_SRC),
        "source_code_sha256": code_sha,
        "feature_map_manifest": rel(FEATURE_MAP_MANIFEST),
        "feature_map_manifest_sha256": sha256_file(FEATURE_MAP_MANIFEST),
        "conditions": len(entries),
        "banks": entries,
    }
    write_json(FEATURE_BANK_MANIFEST, manifest)
    return manifest


def verify_cache_against_bank(bank_entry: dict) -> None:
    path = ROOT / bank_entry["cache_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != bank_entry["cache_sha256"]:
        raise RuntimeError(f"feature cache drift for {bank_entry['bank_id']}")


def rff_on_device(features, w, b, device: str):
    """Apply the frozen RFF map on ``device`` and return CPU ``float64`` numpy.

    The frozen ``W``/``b`` arrays are device independent; only the evaluation
    device of the map application changes. Any CUDA result is cast back to
    ``float64`` on the host, so every downstream decision is computed in the
    protocol's frozen dtype.
    """
    if device != "cuda":
        return apply_rff(features, w, b)
    import torch

    with torch.inference_mode():
        x = torch.as_tensor(np.asarray(features, dtype=np.float64), device="cuda")
        w_t = torch.as_tensor(np.asarray(w, dtype=np.float64), device="cuda")
        b_t = torch.as_tensor(np.asarray(b, dtype=np.float64), device="cuda")
        phi = np.sqrt(2.0 / float(w_t.shape[0])) * torch.cos(x @ w_t.T + b_t)
        return phi.detach().cpu().numpy()


def verify_rff_dimensions(feature_map_manifest: dict, bank: dict) -> dict:
    """Fail closed unless every frozen RFF map matches its layer's stored width.

    This guard exists because an earlier implementation built the RFF maps from a
    hard-coded 512/768 width instead of the frozen per-layer width. No scientific
    quantity depends on the guard; it only prevents a silent shape mismatch.
    """
    verified = []
    for entry in bank["banks"]:
        encoder = entry["encoder"]
        checks = []
        for layer in X2_ENCODER_LAYERS[encoder]:
            stored = int(entry["raw_layer_dim"][layer])
            matches = [
                item
                for item in feature_map_manifest["maps"]
                if item["encoder"] == encoder and item["layer"] == layer
            ]
            if len(matches) != 1:
                raise RuntimeError(f"expected exactly one frozen RFF map for {encoder}/{layer}")
            declared = int(matches[0]["input_dimension"])
            frozen_dim = int(X2_RAW_LAYER_DIM[encoder][layer])
            if not (stored == declared == frozen_dim):
                raise RuntimeError(
                    f"frozen RFF input dimension mismatch for {encoder}/{layer}: "
                    f"stored={stored} declared={declared} frozen={frozen_dim}"
                )
            checks.append({"layer": layer, "stored": stored, "declared": declared, "frozen": frozen_dim})
        verified.append({"condition_id": entry["bank_id"], "layers": checks})
    return {"verified": verified, "passed": True}


def evaluate_x2_condition(
    dataset: str,
    encoder: str,
    feature_map_manifest: dict,
    bank_entry: dict,
    tolerance: float,
    device: str = "cpu",
) -> dict:
    """Evaluate the four frozen layer classifiers of one structural condition."""
    extract = _load_extract_module()
    verify_cache_against_bank(bank_entry)
    path = ROOT / bank_entry["cache_path"]
    with np.load(path, allow_pickle=False) as payload:
        train_labels = np.asarray(payload["train_labels"], dtype=np.int64)
        test_labels = np.asarray(payload["test_labels"], dtype=np.int64)
        clients = extract.client_partition(train_labels, f"{dataset}__{encoder}")
        test_clients = extract.client_partition(test_labels, f"{dataset}__{encoder}__test")
        bacc: dict[str, float] = {}
        recovery_max_abs = 0.0
        counts_equal_all = True
        for layer in X2_ENCODER_LAYERS[encoder]:
            w, b = load_frozen_rff(feature_map_manifest, encoder, layer)
            train_phi = rff_on_device(
                normalize_layer_features(np.asarray(payload[f"train__{layer}"], dtype=np.float64)),
                w,
                b,
                device,
            )
            test_phi = rff_on_device(
                normalize_layer_features(np.asarray(payload[f"test__{layer}"], dtype=np.float64)),
                w,
                b,
                device,
            )
            if not np.isfinite(train_phi).all() or not np.isfinite(test_phi).all():
                raise RuntimeError(f"non-finite RFF features at {dataset}/{encoder}/{layer}")
            _, recovery = aggregate_client_messages(
                train_phi, train_labels, clients, extract.X2_N_CLIENTS, tolerance
            )
            _, test_recovery = aggregate_client_messages(
                test_phi, test_labels, test_clients, extract.X2_N_CLIENTS, tolerance
            )
            recovery_max_abs = max(
                recovery_max_abs, float(recovery["max_abs"]), float(test_recovery["max_abs"])
            )
            counts_equal_all = counts_equal_all and bool(recovery["counts_equal"]) and bool(
                test_recovery["counts_equal"]
            )
            prototypes = reconstruct_prototypes(class_message(train_phi, train_labels))
            predictions = predict_prototype_batch(test_phi, prototypes)
            bacc[layer] = bacc_from_predictions(test_labels, predictions)
    recovery_record = {
        "counts_equal": counts_equal_all,
        "max_abs": recovery_max_abs,
        "tolerance": float(tolerance),
        "passed": bool(counts_equal_all and recovery_max_abs <= float(tolerance)),
    }
    if not bool(recovery_record["passed"]):
        raise RuntimeError(
            f"aggregation recovery failed at {dataset}/{encoder}: max_abs={recovery_max_abs}"
        )
    return summarize_x2_condition(dataset, encoder, bacc, recovery_record)


def write_condition_results_csv(conditions: list[dict]) -> None:
    X2_RUN_DIR.mkdir(parents=True, exist_ok=True)
    with X2_CONDITION_RESULTS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CONDITION_CSV_FIELDS)
        for condition in conditions:
            actions = X2_ENCODER_LAYERS[condition["encoder"]]
            row = [
                condition["dataset"],
                condition["encoder"],
                *[f"{condition['balanced_accuracy'][layer]:.12f}" for layer in actions],
                condition["best_layer"],
                condition["second_best_layer"],
                f"{condition['best_margin']:.12f}",
                f"{condition['layer_spread']:.12f}",
                int(bool(condition["action_relevant"])),
                condition["unique_best"] if condition["unique_best"] is not None else "NONE",
                f"{condition['aggregation_recovery_max_abs']:.3e}",
                int(bool(condition["finite_pass"])),
            ]
            writer.writerow(row)


def x2_outcome_already_opened() -> str | None:
    # Ordered by the stage at which each artifact appears: the feature-bank
    # manifest is written before any X2 BACC exists, the run artifacts after.
    for candidate in (FEATURE_BANK_MANIFEST, FEATURE_MAP_MANIFEST, X2_CONDITION_RESULTS, X2_GATE_SUMMARY, X2_RUN_SUMMARY):
        if candidate.exists():
            return rel(candidate)
    return None


def save_failed_x2_run(reason: str, detail: str) -> None:
    failed_dir = X2_RUN_DIR / "failed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        failed_dir / "INCOMPLETE_RUN.json",
        {
            "run_type": "CMR_V1_X2_INCOMPLETE",
            "reason": reason,
            "detail": detail,
            "git_head": git_head(),
            "outcome_access_completed": False,
            "note": (
                "Engineering failure before a complete formal X2 evaluation. "
                "No gate may be changed; a rerun must use identical frozen authority."
            ),
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CMR-V1 formal runner (x1, feature-map, extract, x2-dry, x2)."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["x1", "feature-map", "extract", "x2-dry", "x2"],
        help="stage to run; X3-X5 are not authorized",
    )
    parser.add_argument(
        "--expected-baseline",
        default=EXPECTED_BASELINE,
        help=(
            "expected git HEAD before the formal run; pass an authorized new "
            "baseline explicitly after a reviewed freeze commit, or '' to record "
            "the HEAD without blocking"
        ),
    )
    parser.add_argument("--dataset", default=None, help="condition dataset (extract mode)")
    parser.add_argument("--encoder", default=None, help="condition encoder (extract mode)")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch CPU threads for extraction; 0 keeps the torch default",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--extract-missing", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "reuse an already extracted condition cache instead of re-extracting it "
            "(deterministic given the frozen weights/preprocessing; the cache hash is "
            "still verified in the feature-bank manifest)"
        ),
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--allow-reopened-outcome", action="store_true")
    parser.add_argument("--save-failed-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.mode == "x1":
        return run_x1_mode(args)
    if args.mode == "feature-map":
        return run_feature_map_mode(args)
    if args.mode == "extract":
        return run_extract_mode(args)
    if args.mode == "x2-dry":
        return run_x2_dry_mode(args)
    return run_x2_formal_mode(args)


def run_x1_mode(args) -> int:
    head, baseline_match = check_baseline(args.expected_baseline)
    if baseline_match is False:
        return 2

    try:
        protocol_meta = load_protocol_authority()
    except Exception as exc:  # noqa: BLE001 - guarded startup failure
        print(f"[CMR-V1 X1] protocol authority check failed: {exc}", file=sys.stderr)
        return 2

    try:
        freeze = verify_freeze_manifest()
    except Exception as exc:  # noqa: BLE001 - guarded startup failure
        print(f"[CMR-V1 X1] freeze manifest verification failed: {exc}", file=sys.stderr)
        return 2

    result = evaluate_x1_exact_collision()
    result_rerun = evaluate_x1_exact_collision()
    in_process_rerun_identical = canonical_dumps(result) == canonical_dumps(result_rerun)

    unit_tests = run_unit_tests()

    gates = dict(result["gates"])
    gates["X1-G6"] = {
        "name": "deterministic rerun",
        "passed": bool(in_process_rerun_identical),
        "in_process_rerun_identical": bool(in_process_rerun_identical),
        "protocol_ref": "execution requirement: repeated formal evaluation is identical",
    }
    gates["X1-G7"] = {
        "name": "unit-test status",
        "passed": bool(unit_tests["passed"]),
        "targets": unit_tests["targets"],
        "returncode": unit_tests["returncode"],
        "tests_passed": unit_tests["tests_passed"],
        "tests_failed": unit_tests["tests_failed"],
        "protocol_ref": "protocol section 7.4 condition 8",
    }
    all_gates_pass = all(bool(gate["passed"]) for gate in gates.values())
    decision = DECISION_PASS if all_gates_pass else DECISION_BLOCKED

    payload = {
        "protocol": {
            "id": protocol_meta["id"],
            "version": protocol_meta["version"],
            "status": protocol_meta["status"],
        },
        "x1_result": result,
        "gates": gates,
        "decision": decision,
    }
    payload_sha = hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()

    previous_payload_sha = None
    if RUN_SUMMARY.exists():
        try:
            previous_payload_sha = json.loads(
                RUN_SUMMARY.read_text(encoding="utf-8")
            ).get("deterministic_payload_sha256")
        except Exception:  # noqa: BLE001 - preserve audit signal
            previous_payload_sha = "UNREADABLE"
    cross_run_payload_match = (
        None if previous_payload_sha is None else previous_payload_sha == payload_sha
    )

    source_files = [MODULE_SRC, RUNNER_SRC, TEST_X1]
    if TEST_PROTOCOL.exists():
        source_files.append(TEST_PROTOCOL)

    run_summary = {
        "run_type": "CMR_V1_X1_FORMAL",
        "stage": "X1",
        "protocol": payload["protocol"],
        "git_head_before_x1": head,
        "expected_baseline": args.expected_baseline,
        "baseline_match": baseline_match,
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "float_dtype": FLOAT_DTYPE_NAME,
        },
        "provenance": {
            "protocol_md_sha256": sha256_file(PROTOCOL_MD),
            "protocol_yaml_sha256": sha256_file(PROTOCOL_YAML),
            "gate_matrix_sha256": sha256_file(GATE_MATRIX),
            "implementation_prompt_sha256": sha256_file(IMPL_PROMPT),
            "freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST),
            "source_files": {rel(path): sha256_file(path) for path in source_files},
            "freeze_manifest_verification": freeze,
        },
        "x1_result": result,
        "gates": gates,
        "decision": decision,
        "deterministic_payload_sha256": payload_sha,
    }
    write_json(RUN_SUMMARY, run_summary)

    rerun_audit = {
        "run_type": "CMR_V1_X1_RERUN_AUDIT",
        "stage": "X1",
        "git_head_before_x1": head,
        "in_process_rerun_identical": bool(in_process_rerun_identical),
        "current_payload_sha256": payload_sha,
        "previous_run_summary_present": previous_payload_sha is not None,
        "previous_payload_sha256": previous_payload_sha,
        "cross_run_payload_match": cross_run_payload_match,
    }
    write_json(RERUN_AUDIT, rerun_audit)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROTOCOL_YAML, PROTOCOL_SNAPSHOT)
    if sha256_file(PROTOCOL_SNAPSHOT) != sha256_file(PROTOCOL_YAML):
        raise RuntimeError("protocol snapshot copy is not byte-identical")

    gate_summary = {
        "protocol_id": protocol_meta["id"],
        "protocol_version": protocol_meta["version"],
        "protocol_status": protocol_meta["status"],
        "stage": "X1",
        "git_head_before_x1": head,
        "provenance": {
            "protocol_md_sha256": sha256_file(PROTOCOL_MD),
            "protocol_yaml_sha256": sha256_file(PROTOCOL_YAML),
            "gate_matrix_sha256": sha256_file(GATE_MATRIX),
            "freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST),
            "module_sha256": sha256_file(MODULE_SRC),
            "runner_sha256": sha256_file(RUNNER_SRC),
        },
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "decision": decision,
    }
    write_json(GATE_SUMMARY, gate_summary)

    artifacts = [
        {
            "path": rel(GATE_SUMMARY),
            "role": "X1 gate summary (X1-G1..X1-G7)",
            "sha256": sha256_file(GATE_SUMMARY),
            "bytes": GATE_SUMMARY.stat().st_size,
        },
        {
            "path": rel(PROTOCOL_SNAPSHOT),
            "role": "frozen protocol snapshot (byte-identical copy of the frozen yaml)",
            "sha256": sha256_file(PROTOCOL_SNAPSHOT),
            "bytes": PROTOCOL_SNAPSHOT.stat().st_size,
        },
        {
            "path": rel(RUN_SUMMARY),
            "role": "formal X1 run summary",
            "sha256": sha256_file(RUN_SUMMARY),
            "bytes": RUN_SUMMARY.stat().st_size,
        },
    ]
    evidence_manifest = {
        "manifest_type": "CMR_V1_X1_EVIDENCE_MANIFEST",
        "protocol_id": protocol_meta["id"],
        "protocol_version": protocol_meta["version"],
        "protocol_status": protocol_meta["status"],
        "stage": "X1",
        "git_head_before_x1": head,
        "decision": decision,
        "artifacts": artifacts,
        "source_files": [
            {"path": rel(path), "sha256": sha256_file(path)} for path in source_files
        ],
        "protocol_files": [
            {"path": rel(path), "sha256": sha256_file(path)}
            for path in (PROTOCOL_MD, PROTOCOL_YAML, GATE_MATRIX, FREEZE_MANIFEST, IMPL_PROMPT)
        ],
        "notes": [
            "X1 stage implementation; no X2-X5 outcome was generated by this run.",
            "run_summary.json is byte-stable across reruns; rerun_audit.json records execution history.",
        ],
    }
    write_json(EVIDENCE_MANIFEST, evidence_manifest)

    bacc = result["balanced_accuracy"]
    print("=" * 72)
    print(f"[CMR-V1 X1] git HEAD: {head}")
    print(f"[CMR-V1 X1] message equality max_abs: {result['message_equality']['max_abs_vector_sum_diff']:.3e}")
    print(f"[CMR-V1 X1] prototype equality max_abs: {result['prototype_equality']['max_abs_diff']:.3e}")
    print(
        "[CMR-V1 X1] BACC "
        f"A.layer1={bacc['A']['layer1']:.12f} A.layer2={bacc['A']['layer2']:.12f} "
        f"B.layer1={bacc['B']['layer1']:.12f} B.layer2={bacc['B']['layer2']:.12f}"
    )
    print(
        f"[CMR-V1 X1] optimal sets: A={result['optimal_sets']['A']} B={result['optimal_sets']['B']}"
    )
    print(
        f"[CMR-V1 X1] deterministic regret: {result['deterministic_regret']['value']:.12f} "
        f"(expected {result['expected']['deterministic_regret']:.12f})"
    )
    print(
        f"[CMR-V1 X1] randomized regret:    {result['randomized_regret']['value']:.12f} "
        f"(expected {result['expected']['randomized_regret']:.12f})"
    )
    print(f"[CMR-V1 X1] in-process rerun identical: {in_process_rerun_identical}")
    print(
        f"[CMR-V1 X1] unit tests: returncode={unit_tests['returncode']} "
        f"passed={unit_tests['tests_passed']} failed={unit_tests['tests_failed']}"
    )
    print(f"[CMR-V1 X1] payload sha256: {payload_sha}")
    print(f"[CMR-V1 X1] decision: {decision}")
    print(f"[CMR-V1 X1] run summary: {rel(RUN_SUMMARY)}")
    print(f"[CMR-V1 X1] gate summary: {rel(GATE_SUMMARY)}")
    print("=" * 72)
    return 0 if all_gates_pass else 1


# --- X2 mode handlers --------------------------------------------------------
def guard_x2_outcome_access(mode: str) -> int | None:
    """Refuse to silently re-open X2 outcome access."""
    opened = x2_outcome_already_opened()
    if opened is None:
        return None
    print(
        f"[CMR-V1 {mode}] X2 outcome access already occurred (first artifact: {opened}).",
        file=sys.stderr,
    )
    print(
        "[CMR-V1 {m}] A second formal X2 run is only permitted after an audited "
        "engineering failure. Re-run with --allow-reopened-outcome, which must be "
        "recorded in the report.".format(m=mode),
        file=sys.stderr,
    )
    return 2


def run_feature_map_mode(args) -> int:
    try:
        authority = load_x2_protocol_constants(PROTOCOL_YAML)
    except Exception as exc:  # noqa: BLE001 - guarded startup failure
        print(f"[CMR-V1 feature-map] frozen x2 authority check failed: {exc}", file=sys.stderr)
        return 2
    head, baseline_match = check_baseline(args.expected_baseline)
    if baseline_match is False:
        return 2
    manifest = build_feature_map_manifest()
    print(f"[CMR-V1 feature-map] git HEAD: {head}")
    print(f"[CMR-V1 feature-map] protocol: {authority['protocol_id']} {authority['protocol_version']}")
    print(
        f"[CMR-V1 feature-map] frozen maps: {len(manifest['maps'])} unique "
        f"(encoder, layer) maps covering {len(X2_CONDITIONS)} structural conditions "
        f"(m={X2_RFF_DIMENSION}, sigma={X2_RFF_SIGMA}, master_seed={X2_RFF_MASTER_SEED})"
    )
    for entry in manifest["maps"]:
        print(
            f"[CMR-V1 feature-map]   {entry['encoder']:<28} {entry['layer']:<7} "
            f"dim={entry['input_dimension']:<3} seed={entry['derived_seed']:<10} "
            f"W={entry['w_sha256'][:12]} b={entry['b_sha256'][:12]}"
        )
    print(f"[CMR-V1 feature-map] manifest: {rel(FEATURE_MAP_MANIFEST)}")
    print("=" * 72)
    return 0


def run_extract_mode(args) -> int:
    # Extraction only materializes frozen features; it opens no outcome and is
    # therefore not gated on the formal-run baseline. The HEAD is still recorded.
    head, _baseline_match = check_baseline("")
    if args.dataset and args.encoder:
        if args.dataset not in X2_DATASETS:
            print(f"[CMR-V1 extract] dataset not in frozen authority: {args.dataset}", file=sys.stderr)
            return 2
        if args.encoder not in X2_ENCODER_LAYERS:
            print(f"[CMR-V1 extract] encoder not in frozen authority: {args.encoder}", file=sys.stderr)
            return 2
        pairs = [(args.dataset, args.encoder)]
    else:
        pairs = list(X2_CONDITIONS)
    device = args.device or "cpu"
    if device not in INFERENCE_BACKENDS:
        print(f"[CMR-V1 extract] unsupported device: {device}", file=sys.stderr)
        return 2
    backend = backend_metadata(device)
    print(
        f"[CMR-V1 extract] inference backend: {backend['inference_backend']} "
        f"(torch {backend['torch_version']}, torchvision {backend['torchvision_version']}"
        + (f", gpu {backend.get('gpu_model')}" if device == "cuda" else "")
        + ")"
    )
    if int(args.threads) > 0:
        import torch

        torch.set_num_threads(int(args.threads))
        print(f"[CMR-V1 extract] torch CPU threads: {torch.get_num_threads()}")
    failures = []
    batch_size = int(args.batch_size) or BACKEND_BATCH_SIZE[device]
    print(f"[CMR-V1 extract] batch_size={batch_size} workers={args.workers}")
    print(f"[CMR-V1 extract] raw dataset root: {rel(resolve_raw_root(device))}")
    for dataset, encoder in pairs:
        cache_path = condition_cache_path(dataset, encoder, device)
        if args.skip_existing and cache_path.is_file() and cache_path.stat().st_size > 0:
            print(
                f"[CMR-V1 extract] SKIP {dataset}__{encoder} (existing cache "
                f"{rel(cache_path)}, sha256={sha256_file(cache_path)[:16]})"
            )
            continue
        try:
            entry = extract_condition_to_cache(
                dataset,
                encoder,
                batch_size=batch_size,
                num_workers=args.workers,
                device=device,
                allow_download=args.allow_download,
            )
            print(
                f"[CMR-V1 extract] OK  {entry['condition_id']:<45} "
                f"train={entry['train_records']:<7} test={entry['test_records']:<6} "
                f"bytes={entry['cache_bytes']}"
            )
        except Exception as exc:  # noqa: BLE001 - one condition must not hide the others
            failures.append({"condition": f"{dataset}__{encoder}", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[CMR-V1 extract] FAIL {dataset}__{encoder}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"[CMR-V1 extract] git HEAD: {head}")
    if failures:
        print(f"[CMR-V1 extract] {len(failures)} condition(s) failed: {failures}", file=sys.stderr)
        return 1
    print(f"[CMR-V1 extract] completed {len(pairs)} condition(s)")
    print("=" * 72)
    return 0


def run_x2_dry_mode(args) -> int:
    """Implementation dry validation on cached feature slices; writes no formal output.

    The dry validation slices the already-extracted frozen caches and exercises
    the complete X2 path (frozen RFF, client partition, aggregation recovery,
    prototype classifier, BACC, spread, gates). It is outcome-free in the sense
    that no design decision, layer, seed or threshold depends on it.
    """
    try:
        authority = load_x2_protocol_constants(PROTOCOL_YAML)
    except Exception as exc:  # noqa: BLE001 - guarded startup failure
        print(f"[CMR-V1 x2-dry] frozen x2 authority check failed: {exc}", file=sys.stderr)
        return 2
    head, _baseline_match = check_baseline("")
    device = args.device or "cpu"
    print(f"[CMR-V1 x2-dry] git HEAD: {head}")
    print(f"[CMR-V1 x2-dry] protocol {authority['protocol_id']} {authority['protocol_version']}")

    # Deterministic downsampling indices for the dry slice.
    rng = np.random.default_rng(20261001)
    summaries = []
    device = args.device or "cpu"
    for dataset in X2_DRY_DATASETS:
        for encoder in X2_ENCODER_LAYERS:
            cache_path = condition_cache_path(dataset, encoder, device)
            if not cache_path.is_file():
                if not args.extract_missing:
                    print(
                        f"[CMR-V1 x2-dry] SKIP {dataset}__{encoder}: no frozen cache yet "
                        "(use --extract-missing to extract a dry slice)",
                        file=sys.stderr,
                    )
                    continue
                extract_condition_to_cache(
                    dataset,
                    encoder,
                    batch_size=int(args.batch_size) or BACKEND_BATCH_SIZE[device],
                    num_workers=args.workers,
                    device=device,
                    allow_download=args.allow_download,
                    dry=True,
                )
            with np.load(cache_path, allow_pickle=False) as payload:
                train_labels = np.asarray(payload["train_labels"], dtype=np.int64)
                test_labels = np.asarray(payload["test_labels"], dtype=np.int64)
                train_index = np.sort(
                    rng.choice(
                        len(train_labels),
                        size=min(int(X2_DRY_MAX_RECORDS), len(train_labels)),
                        replace=False,
                    )
                )
                test_index = np.sort(
                    rng.choice(
                        len(test_labels),
                        size=min(int(X2_DRY_MAX_TEST_RECORDS), len(test_labels)),
                        replace=False,
                    )
                )
                train_labels = train_labels[train_index]
                test_labels = test_labels[test_index]
                clients = _load_extract_module().client_partition(
                    train_labels, f"dry__{dataset}__{encoder}"
                )
                bacc: dict[str, float] = {}
                recovery_max_abs = 0.0
                counts_equal_all = True
                for layer in X2_ENCODER_LAYERS[encoder]:
                    w, b = load_frozen_rff_from_disk(encoder, layer)
                    train_phi = apply_rff(
                        normalize_layer_features(
                            np.asarray(payload[f"train__{layer}"], dtype=np.float64)[train_index]
                        ),
                        w,
                        b,
                    )
                    test_phi = apply_rff(
                        normalize_layer_features(
                            np.asarray(payload[f"test__{layer}"], dtype=np.float64)[test_index]
                        ),
                        w,
                        b,
                    )
                    _, recovery = aggregate_client_messages(
                        train_phi, train_labels, clients, 20, X2_AGGREGATION_TOLERANCE
                    )
                    recovery_max_abs = max(recovery_max_abs, float(recovery["max_abs"]))
                    counts_equal_all = counts_equal_all and bool(recovery["counts_equal"])
                    prototypes = reconstruct_prototypes(class_message(train_phi, train_labels))
                    bacc[layer] = bacc_from_predictions(
                        test_labels, predict_prototype_batch(test_phi, prototypes)
                    )
            recovery_record = {
                "counts_equal": counts_equal_all,
                "max_abs": recovery_max_abs,
                "passed": bool(counts_equal_all and recovery_max_abs <= X2_AGGREGATION_TOLERANCE),
            }
            summary = summarize_x2_condition(dataset, encoder, bacc, recovery_record)
            summaries.append(summary)
            print(
                f"[CMR-V1 x2-dry] {dataset:<10} {encoder:<28} "
                + " ".join(
                    f"{layer}={summary['balanced_accuracy'][layer]:.4f}"
                    for layer in X2_ENCODER_LAYERS[encoder]
                )
                + f" recovery_max_abs={summary['aggregation_recovery_max_abs']:.3e}"
                + f" finite={summary['finite_pass']}"
            )
    print(
        f"[CMR-V1 x2-dry] dry conditions evaluated: {len(summaries)} "
        "(NO formal artifact written; no design decision depends on these numbers)"
    )
    print("=" * 72)
    return 0


def load_frozen_rff_from_disk(encoder: str, layer: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the frozen RFF arrays straight from the frozen artifact."""
    path = rff_array_path(encoder, layer)
    if not path.is_file():
        raise FileNotFoundError(
            f"frozen RFF artifact missing for {encoder}/{layer}; run --mode feature-map first"
        )
    with np.load(path) as frozen:
        return (
            np.asarray(frozen["W"], dtype=np.float64),
            np.asarray(frozen["b"], dtype=np.float64),
        )


def run_x2_formal_mode(args) -> int:
    guard = guard_x2_outcome_access("x2")
    outcome_access = {
        "already_open_before_run": guard is not None,
        "first_stale_artifact": guard,
        "reopened_outcome_access": False,
        "justification": "initial formal run had no outcome access",
    }
    if guard is not None:
        if not args.allow_reopened_outcome:
            return guard
        outcome_access["reopened_outcome_access"] = True
        outcome_access["justification"] = (
            "documented engineering failure with no outcome access: "
            "frozen RFF input width was built from a hard-coded 512/768 instead of the "
            "frozen per-layer width, so the first attempt aborted with a shape error "
            "before any X2 BACC existed (see "
            "runs/cmr_v1/x2_cuda_migration/engineering_correction_rff_width.json)"
        )
        outcome_access["engineering_failure_record"] = (
            "runs/cmr_v1/x2_cuda_migration/engineering_correction_rff_width.json"
        )
        save_failed_x2_run(
            "ENGINEERING_FAILURE_NO_OUTCOME_ACCESS",
            "frozen RFF input width bug; fixed by deriving widths from the frozen action family",
        )
    try:
        authority = load_x2_protocol_constants(PROTOCOL_YAML)
    except Exception as exc:  # noqa: BLE001 - guarded startup failure
        print(f"[CMR-V1 x2] frozen x2 authority check failed: {exc}", file=sys.stderr)
        return 2
    try:
        freeze = verify_freeze_manifest()
    except Exception as exc:  # noqa: BLE001 - guarded startup failure
        print(f"[CMR-V1 x2] freeze manifest verification failed: {exc}", file=sys.stderr)
        return 2
    head, baseline_match = check_baseline(args.expected_baseline)
    if baseline_match is False:
        return 2
    device = args.device or "cpu"
    if device not in INFERENCE_BACKENDS:
        print(f"[CMR-V1 x2] unsupported device: {device}", file=sys.stderr)
        return 2
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("[CMR-V1 x2] CUDA requested but unavailable", file=sys.stderr)
            return 2
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    backend = backend_metadata(device)
    print(f"[CMR-V1 x2] git HEAD: {head}")
    print(f"[CMR-V1 x2] protocol: {authority['protocol_id']} {authority['protocol_version']}")
    print(
        f"[CMR-V1 x2] inference backend: {backend['inference_backend']} "
        f"(torch {backend['torch_version']}, torchvision {backend['torchvision_version']}"
        + (f", gpu {backend.get('gpu_model')}, cuda {backend['cuda_runtime_version']}" if device == "cuda" else "")
        + ")"
    )
    print(
        f"[CMR-V1 x2] RFF: m={X2_RFF_DIMENSION} sigma={X2_RFF_SIGMA} "
        f"master_seed={X2_RFF_MASTER_SEED} dtype={FLOAT_DTYPE_NAME}"
    )

    feature_map_manifest = build_feature_map_manifest()
    missing = [
        f"{dataset}__{encoder}"
        for dataset, encoder in X2_CONDITIONS
        if not condition_cache_path(dataset, encoder, device).is_file()
    ]
    if missing:
        save_failed_x2_run("MISSING_FEATURE_CACHE", str(missing))
        print(f"[CMR-V1 x2] missing extracted conditions: {missing}", file=sys.stderr)
        return 1

    # Freeze feature-bank provenance BEFORE any X2 BACC is observed.
    bank = build_feature_bank_manifest(device)
    rff_dimension_check = verify_rff_dimensions(feature_map_manifest, bank)
    print(
        "[CMR-V1 x2] frozen RFF input-dimension guard: PASS "
        f"({len(rff_dimension_check['verified'])} conditions)"
    )
    print(f"[CMR-V1 x2] feature-bank manifest frozen: {rel(FEATURE_BANK_MANIFEST)}")
    for entry in bank["banks"]:
        print(
            f"[CMR-V1 x2]   {entry['bank_id']:<45} train={entry['train_records']:<7} "
            f"test={entry['test_records']:<6} classes={entry['class_count']} "
            f"backend={entry['inference_backend']}"
        )

    try:
        conditions = []
        for dataset, encoder in X2_CONDITIONS:
            summary = evaluate_x2_condition(
                dataset,
                encoder,
                feature_map_manifest,
                next(e for e in bank["banks"] if e["bank_id"] == f"{dataset}__{encoder}"),
                X2_AGGREGATION_TOLERANCE,
                device,
            )
            conditions.append(summary)
            bacc = summary["balanced_accuracy"]
            print(
                f"[CMR-V1 x2] {dataset:<10} {encoder:<28} "
                + " ".join(f"{layer}={bacc[layer]:.6f}" for layer in X2_ENCODER_LAYERS[encoder])
                + f" spread={summary['layer_spread']:.6f} margin={summary['best_margin']:.6f} "
                f"best={summary['best_layer']} unique={summary['unique_best']} "
                f"recovery={summary['aggregation_recovery_max_abs']:.3e}"
            )
    except Exception as exc:  # noqa: BLE001 - preserve a failed-run artifact
        if args.save_failed_run:
            save_failed_x2_run("EVALUATION_ERROR", f"{type(exc).__name__}: {exc}")
        raise

    x2_evaluation = evaluate_x2_gates(conditions)
    unit_tests = run_unit_tests([TEST_X1, TEST_PROTOCOL, TEST_X2])
    write_condition_results_csv(conditions)

    run_summary = {
        "run_type": "CMR_V1_X2_FORMAL",
        "stage": "X2",
        "protocol": {
            "id": authority["protocol_id"],
            "version": authority["protocol_version"],
            "status": authority["protocol_status"],
        },
        "git_head": head,
        "expected_baseline": args.expected_baseline,
        "baseline_match": baseline_match,
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "device": device,
            "float_dtype": FLOAT_DTYPE_NAME,
        },
        "inference_backend": backend,
        "outcome_access": outcome_access,
        "rff_input_dimension_guard": rff_dimension_check,
        "provenance": {
            "protocol_md_sha256": sha256_file(PROTOCOL_MD),
            "protocol_yaml_sha256": sha256_file(PROTOCOL_YAML),
            "gate_matrix_sha256": sha256_file(GATE_MATRIX),
            "freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST),
            "feature_map_manifest_sha256": sha256_file(FEATURE_MAP_MANIFEST),
            "feature_bank_manifest_sha256": sha256_file(FEATURE_BANK_MANIFEST),
            "source_files": {
                rel(path): sha256_file(path) for path in X2_SOURCE_FILES if path.exists()
            },
            "freeze_manifest_verification": freeze,
        },
        "frozen_constants": {
            "datasets": list(X2_DATASETS),
            "encoders": {k: list(v) for k, v in X2_ENCODER_LAYERS.items()},
            "primary_metric": "balanced_accuracy",
            "action_relevant_spread_min": X2_ACTION_RELEVANT_SPREAD_MIN,
            "unique_best_margin_min": X2_UNIQUE_BEST_MARGIN_MIN,
            "aggregation_tolerance": X2_AGGREGATION_TOLERANCE,
            "client_partition": {"n_clients": 20, "dirichlet_alpha": 0.10, "seed": 20260908},
            "rff": {
                "dimension": X2_RFF_DIMENSION,
                "sigma": X2_RFF_SIGMA,
                "master_seed": X2_RFF_MASTER_SEED,
                "generation_dtype": FLOAT_DTYPE_NAME,
            },
        },
        "conditions": conditions,
        "gates": x2_evaluation["gates"],
        "all_gates_pass": x2_evaluation["all_gates_pass"],
        "decision": x2_evaluation["decision"],
        "cmr_x2_decision": x2_evaluation["decision"],
        "final_cmr_decision": x2_evaluation["final_cmr_decision"],
        "protocol_stops": x2_evaluation["stop"],
        "unit_tests": unit_tests,
        "prohibited_after_x2_outcome_access": authority[
            "prohibited_after_x2_outcome_access"
        ],
    }
    write_json(X2_RUN_SUMMARY, run_summary)

    X2_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROTOCOL_YAML, X2_PROTOCOL_SNAPSHOT)
    if sha256_file(X2_PROTOCOL_SNAPSHOT) != sha256_file(PROTOCOL_YAML):
        raise RuntimeError("protocol snapshot copy is not byte-identical")

    gate_summary = {
        "protocol_id": authority["protocol_id"],
        "protocol_version": authority["protocol_version"],
        "protocol_status": authority["protocol_status"],
        "stage": "X2",
        "git_head": head,
        "provenance": {
            "protocol_md_sha256": sha256_file(PROTOCOL_MD),
            "protocol_yaml_sha256": sha256_file(PROTOCOL_YAML),
            "gate_matrix_sha256": sha256_file(GATE_MATRIX),
            "freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST),
            "feature_map_manifest_sha256": sha256_file(FEATURE_MAP_MANIFEST),
            "feature_bank_manifest_sha256": sha256_file(FEATURE_BANK_MANIFEST),
            "module_sha256": sha256_file(MODULE_SRC),
            "runner_sha256": sha256_file(RUNNER_SRC),
            "feature_extract_sha256": sha256_file(FEATURE_EXTRACT_SRC),
        },
        "gates": x2_evaluation["gates"],
        "all_gates_pass": x2_evaluation["all_gates_pass"],
        "cmr_x2_decision": x2_evaluation["decision"],
        "final_cmr_decision": x2_evaluation["final_cmr_decision"],
        "protocol_stops": x2_evaluation["stop"],
        "condition_count": len(conditions),
        "action_relevant_count": int(
            sum(1 for condition in conditions if condition["action_relevant"])
        ),
    }
    write_json(X2_GATE_SUMMARY, gate_summary)

    artifacts = [
        {
            "path": rel(X2_CONDITION_RESULTS),
            "role": "per-condition X2 layer BACC, spread, margin and UNIQUE_BEST",
            "sha256": sha256_file(X2_CONDITION_RESULTS),
            "bytes": X2_CONDITION_RESULTS.stat().st_size,
        },
        {
            "path": rel(X2_GATE_SUMMARY),
            "role": "X2 gate summary (X2-G1..X2-G5) and decision",
            "sha256": sha256_file(X2_GATE_SUMMARY),
            "bytes": X2_GATE_SUMMARY.stat().st_size,
        },
        {
            "path": rel(X2_PROTOCOL_SNAPSHOT),
            "role": "frozen protocol snapshot (byte-identical copy of the frozen yaml)",
            "sha256": sha256_file(X2_PROTOCOL_SNAPSHOT),
            "bytes": X2_PROTOCOL_SNAPSHOT.stat().st_size,
        },
        {
            "path": rel(X2_RUN_SUMMARY),
            "role": "formal X2 run summary",
            "sha256": sha256_file(X2_RUN_SUMMARY),
            "bytes": X2_RUN_SUMMARY.stat().st_size,
        },
        {
            "path": rel(FEATURE_MAP_MANIFEST),
            "role": "frozen Gaussian-RBF feature-map manifest",
            "sha256": sha256_file(FEATURE_MAP_MANIFEST),
            "bytes": FEATURE_MAP_MANIFEST.stat().st_size,
        },
        {
            "path": rel(FEATURE_BANK_MANIFEST),
            "role": "feature-bank provenance manifest",
            "sha256": sha256_file(FEATURE_BANK_MANIFEST),
            "bytes": FEATURE_BANK_MANIFEST.stat().st_size,
        },
    ]
    evidence_manifest = {
        "manifest_type": "CMR_V1_X2_EVIDENCE_MANIFEST",
        "protocol_id": authority["protocol_id"],
        "protocol_version": authority["protocol_version"],
        "protocol_status": authority["protocol_status"],
        "stage": "X2",
        "git_head": head,
        "decision": x2_evaluation["decision"],
        "final_cmr_decision": x2_evaluation["final_cmr_decision"],
        "artifacts": artifacts,
        "source_files": [
            {"path": rel(path), "sha256": sha256_file(path)}
            for path in X2_SOURCE_FILES
            if path.exists()
        ],
        "protocol_files": [
            {"path": rel(path), "sha256": sha256_file(path)}
            for path in (PROTOCOL_MD, PROTOCOL_YAML, GATE_MATRIX, FREEZE_MANIFEST)
        ],
        "unit_tests": unit_tests,
        "notes": [
            "X2 only; no X3-X5 artifact was generated or authorized.",
            "Frozen RFF arrays are byte-identical on regeneration; downstream evaluation reads the frozen arrays only.",
            "action family, RFF dimension, sigma, seeds and thresholds were never modified after outcome access.",
        ],
    }
    write_json(X2_EVIDENCE_MANIFEST, evidence_manifest)

    print("=" * 72)
    for gate_id, gate in x2_evaluation["gates"].items():
        print(f"[CMR-V1 x2] {gate_id} {'PASS' if gate['passed'] else 'FAIL'} {gate['name']}")
    print(f"[CMR-V1 x2] unit tests: {unit_tests['tests_passed']} passed, {unit_tests['tests_failed']} failed")
    print(f"[CMR-V1 x2] action-relevant conditions: {gate_summary['action_relevant_count']}/8")
    print(f"[CMR-V1 x2] CMR_X2_DECISION: {x2_evaluation['decision']}")
    print(f"[CMR-V1 x2] final CMR decision: {x2_evaluation['final_cmr_decision']}")
    print(f"[CMR-V1 x2] condition results: {rel(X2_CONDITION_RESULTS)}")
    print(f"[CMR-V1 x2] gate summary: {rel(X2_GATE_SUMMARY)}")
    print("=" * 72)
    return 0 if x2_evaluation["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
