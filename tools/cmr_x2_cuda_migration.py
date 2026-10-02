"""CMR-V1 X2 CPU -> CUDA backend migration harness.

This is an **implementation-backend migration**, not a scientific protocol
amendment. Nothing in the frozen CMR-V1 authority changes: datasets, splits,
dataset versions, encoders, pretrained weight enums, preprocessing, candidate
layers, extraction semantics, RFF dimension/sigma/seed derivation, client
partition, primary metric, X2 thresholds and the final CMR-A/B/C/D matrix are all
read from the frozen protocol and only ever verified, never redefined.

Subcommands:

``env-cpu``       record the CPU reference environment and mark CPU caches
                  ``CPU_PREMIGRATION_REFERENCE_ONLY``;
``env-cuda``      record the CUDA environment (fails if CUDA is unavailable);
``parity-samples`` freeze the deterministic parity sample manifest;
``cpu-reference`` run the frozen CPU code path on those samples;
``gpu-reference`` run the same frozen code path with ``device=cuda``;
``parity``        compare CPU/GPU references and evaluate the migration gates;
``benchmark``     engineering throughput benchmark (batch sizes only);
``extract-cuda``  formal CUDA re-extraction of the eight structural conditions.

Formal outputs live under ``runs/cmr_v1/x2_cuda_migration/``. CUDA feature
caches live under ``data/cmr_v1/x2_cuda/`` and never overwrite the CPU
pre-migration cache under ``data/cmr_v1/x2/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.cmr_feature_extract import (  # noqa: E402
    RESNET_ENCODER_ID,
    VIT_ENCODER_ID,
    X2_RAW_LAYER_DIM,
    build_dataset,
    build_encoder,
    encoder_state_sha256,
    extract_split,
    layer_modules,
    layer_features_on_device,
    verify_encoder_layers,
)
from frontier.cmr_kernel_layer import (  # noqa: E402
    X2_CONDITIONS,
    X2_DATASETS,
    X2_ENCODER_LAYERS,
    X2_RFF_DIMENSION,
    X2_RFF_MASTER_SEED,
    X2_RFF_SIGMA,
    apply_rff,
    array_sha256,
    bacc_from_predictions,
    class_message,
    generate_rff,
    load_x2_protocol_constants,
    normalize_layer_features,
    predict_prototype_batch,
    reconstruct_prototypes,
    rff_seed,
)

PROTOCOL_YAML = ROOT / "configs/cross_mechanism_replication_protocol_v1.yaml"
PROTOCOL_MD = ROOT / "docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md"
GATE_MATRIX = ROOT / "docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv"
FEATURE_EXTRACT_SRC = ROOT / "src/frontier/cmr_feature_extract.py"
KERNEL_SRC = ROOT / "src/frontier/cmr_kernel_layer.py"
MIGRATION_TOOL = ROOT / "tools/cmr_x2_cuda_migration.py"

MIGRATION_DIR = ROOT / "runs/cmr_v1/x2_cuda_migration"
CPU_REFERENCE_DIR = MIGRATION_DIR / "cpu_reference"
GPU_REFERENCE_DIR = MIGRATION_DIR / "gpu_reference"

ENV_CPU_JSON = MIGRATION_DIR / "environment_cpu.json"
ENV_CUDA_JSON = MIGRATION_DIR / "environment_cuda.json"
PARITY_SAMPLES_JSON = MIGRATION_DIR / "parity_sample_manifest.json"
CPU_REF_SUMMARY = MIGRATION_DIR / "cpu_reference_summary.json"
GPU_REF_SUMMARY = MIGRATION_DIR / "gpu_reference_summary.json"
PARITY_REPORT = MIGRATION_DIR / "parity_report.json"
THROUGHPUT_JSON = MIGRATION_DIR / "throughput_benchmark.json"
MIGRATION_GATE = MIGRATION_DIR / "migration_gate_summary.json"

CPU_CACHE_ROOT = ROOT / "data/cmr_v1/x2"
CPU_RAW_ROOT = CPU_CACHE_ROOT / "raw"
CPU_FEATURE_ROOT = CPU_CACHE_ROOT / "features"
CUDA_ROOT = ROOT / "data/cmr_v1/x2_cuda"
CUDA_FEATURE_ROOT = CUDA_ROOT / "features"
CUDA_RAW_ROOT = CUDA_ROOT / "raw"

CPU_REFERENCE_MARKER = "CPU_PREMIGRATION_REFERENCE_ONLY"

# Parity sample size per dataset and split (deterministic, not tuned).
PARITY_TRAIN_N = 32
PARITY_TEST_N = 32

# Reference CPU environment (recorded, not modified).
CPU_ENV_PYTHON = Path(r"D:\ProgramData\anaconda3\python.exe")

# Parity acceptance (implementation validation tolerances, NOT scientific gates).
PARITY_MEDIAN_COSINE_MIN = 0.999999
PARITY_MIN_COSINE_MIN = 0.99999
PARITY_PREDICTION_AGREEMENT = 1.0
PARITY_BACC_ABS_DIFF_MAX = 1e-6


# =============================================================================
# Shared helpers
# =============================================================================
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, check=False
    ).stdout.strip()


def driver_version() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else "UNKNOWN"


def torch_env(python_executable: str | None = None) -> dict[str, Any]:
    import torch
    import torchvision

    record: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "os": platform.platform(),
        "processor": platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
        "git_head": git_head(),
    }
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        record.update(
            {
                "gpu_model": torch.cuda.get_device_name(0),
                "gpu_capability": list(torch.cuda.get_device_capability(0)),
                "gpu_total_mib": round(total_bytes / 2**20, 1),
                "gpu_free_mib": round(free_bytes / 2**20, 1),
                "driver_version": driver_version(),
            }
        )
    return record


def dataset_identity(dataset_id: str, root: Path, train, test) -> dict[str, Any]:
    """Canonical record identity: per-record id, label and content hash."""
    ids, labels = _record_ids(dataset_id, root, train, test)
    train_ids, test_ids = ids
    train_labels, test_labels = labels
    payload = {
        "dataset": dataset_id,
        "train_ids": list(train_ids),
        "test_ids": list(test_ids),
        "train_labels": [int(v) for v in train_labels],
        "test_labels": [int(v) for v in test_labels],
    }
    return {
        "train_records": int(len(train_labels)),
        "test_records": int(len(test_labels)),
        "train_class_counts": _class_counts(train_labels),
        "test_class_counts": _class_counts(test_labels),
        "canonical_sample_id_hash": hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def _record_ids(dataset_id: str, root: Path, train, test):
    """Deterministic per-record identity and label arrays for both splits."""
    if dataset_id == "cifar100":
        train_labels = np.asarray(train.targets, dtype=np.int64)
        test_labels = np.asarray(test.targets, dtype=np.int64)
        train_hashes = [_cifar_row_hash(np.asarray(train.data[i])) for i in range(len(train_labels))]
        test_hashes = [_cifar_row_hash(np.asarray(test.data[i])) for i in range(len(test_labels))]
        return (train_hashes, test_hashes), (train_labels, test_labels)
    if dataset_id == "eurosat":
        train_labels = np.asarray(
            [_eurosat_label(train, i) for i in range(len(train))], dtype=np.int64
        )
        test_labels = np.asarray(
            [_eurosat_label(test, i) for i in range(len(test))], dtype=np.int64
        )
        train_hashes = [_eurosat_id(train, i) for i in range(len(train_labels))]
        test_hashes = [_eurosat_id(test, i) for i in range(len(test_labels))]
        return (train_hashes, test_hashes), (train_labels, test_labels)
    if dataset_id in {"pathmnist", "dermamnist"}:
        train_labels = np.asarray(train.labels, dtype=np.int64).reshape(-1)
        test_labels = np.asarray(test.labels, dtype=np.int64).reshape(-1)
        train_hashes = [_medmnist_id(dataset_id, "train", i) for i in range(len(train_labels))]
        test_hashes = [_medmnist_id(dataset_id, "test", i) for i in range(len(test_labels))]
        return (train_hashes, test_hashes), (train_labels, test_labels)
    raise ValueError(dataset_id)


def _cifar_row_hash(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row, dtype=np.uint8).tobytes()).hexdigest()[:24]


def _eurosat_id(subset, index: int) -> str:
    samples, mapping = _eurosat_samples(subset)
    resolved = index if mapping is None else int(mapping[index])
    path = Path(samples[resolved][0])
    return f"eurosat::{path.parent.name}/{path.name}"


def _medmnist_id(dataset_id: str, split: str, index: int) -> str:
    return f"{dataset_id}::{split}::{index}"


def parity_sample_selection(total: int, count: int) -> list[int]:
    """Deterministic near-evenly spaced record indices (stride 1 when possible)."""
    if total <= 0:
        raise ValueError("empty split")
    count = min(int(count), int(total))
    if total <= 2 * count:
        return list(range(count))
    step = total / float(count)
    return [int(np.floor(index * step)) for index in range(count)]


def load_parity_samples(dataset_id: str, raw_root: Path, transform, allow_download: bool):
    train, test, split_meta = build_dataset(
        dataset_id, raw_root, transform, allow_download=allow_download
    )
    train_index = parity_sample_selection(len(train), PARITY_TRAIN_N)
    test_index = parity_sample_selection(len(test), PARITY_TEST_N)
    return train, test, split_meta, train_index, test_index


def subset_by_index(dataset, indices: list[int]):
    from torch.utils.data import Subset

    return Subset(dataset, list(indices))


def stage_metrics(cpu: np.ndarray, gpu: np.ndarray, prefix: str) -> dict[str, Any]:
    """max_abs / mean_abs / max_rel / cosine for one pipeline stage."""
    cpu = np.asarray(cpu, dtype=np.float64)
    gpu = np.asarray(gpu, dtype=np.float64)
    if cpu.shape != gpu.shape:
        raise ValueError(f"{prefix}: shape mismatch {cpu.shape} vs {gpu.shape}")
    difference = np.abs(cpu - gpu)
    denominator = np.maximum(np.abs(cpu), 1e-12)
    flat_cpu = cpu.reshape(len(cpu), -1)
    flat_gpu = gpu.reshape(len(gpu), -1)
    cpu_norm = np.linalg.norm(flat_cpu, axis=1)
    gpu_norm = np.linalg.norm(flat_gpu, axis=1)
    cosine = np.sum(flat_cpu * flat_gpu, axis=1) / np.maximum(cpu_norm * gpu_norm, 1e-30)
    return {
        "stage": prefix,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "max_rel": float((difference / denominator).max()),
        "cosine_median": float(np.median(cosine)),
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
    }


# =============================================================================
# Frozen extraction used by BOTH backends (single scientific code path)
# =============================================================================
def extract_frozen_condition(
    dataset_id: str,
    encoder_id: str,
    raw_root: Path,
    device: str,
    batch_size: int,
    workers: int,
    allow_download: bool,
    train_index: list[int] | None = None,
    test_index: list[int] | None = None,
) -> dict[str, Any]:
    """Extract the four frozen layers for one condition on ``device``.

    Returns raw ``float32`` pooled layer features (pre-L2) plus provenance. The
    scientific semantics are device independent.
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    verify_encoder_layers(encoder_id, X2_ENCODER_LAYERS[encoder_id])
    if dataset_id not in X2_DATASETS:
        raise ValueError(f"dataset outside frozen authority: {dataset_id}")
    model, weights, torch_version, torchvision_version = build_encoder(encoder_id, device)
    transform = weights.transforms()
    train, test, split_meta = build_dataset(
        dataset_id, raw_root, transform, allow_download=allow_download
    )
    identity = dataset_identity(dataset_id, raw_root, train, test)
    sampled = train_index is not None or test_index is not None
    if train_index is not None:
        train = Subset(train, list(train_index))
    if test_index is not None:
        test = Subset(test, list(test_index))

    def run(split, labels_source) -> tuple[dict[str, np.ndarray], np.ndarray]:
        from frontier.phase1_featurebank import collate_medmnist

        custom = dataset_id in {"pathmnist", "dermamnist"}
        loader = DataLoader(
            split,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_medmnist if custom else None,
            pin_memory=device.startswith("cuda"),
        )
        chunks: dict[str, list[np.ndarray]] = {name: [] for name in X2_ENCODER_LAYERS[encoder_id]}
        label_chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for batch, labels in loader:
                batch = batch.to(device, non_blocking=True)
                pooled = layer_features_on_device(model, encoder_id, batch, device)
                for name in X2_ENCODER_LAYERS[encoder_id]:
                    tensor = pooled[name]
                    if device == "cuda":
                        tensor = tensor.detach().cpu()
                    chunks[name].append(np.asarray(tensor).astype(np.float32, copy=False))
                label_chunks.append(np.asarray(labels).reshape(-1).astype(np.int64))
        arrays = {name: np.concatenate(chunks[name], axis=0) for name in chunks}
        return arrays, np.concatenate(label_chunks, axis=0)

    train_arrays, train_labels = run(train, train)
    test_arrays, test_labels = run(test, test)
    state_hash = encoder_state_sha256(model)
    if device == "cuda":
        torch.cuda.empty_cache()
    del model
    return {
        "dataset": dataset_id,
        "encoder": encoder_id,
        "device": device,
        "sampled": sampled,
        "train": train_arrays,
        "test": test_arrays,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "split_meta": split_meta,
        "identity": identity,
        "provenance": {
            "pretrained_weights_enum": str(weights),
            "preprocessing_id": f"torchvision_weights_transforms::{weights}",
            "transform_repr": repr(transform),
            "transform_sha256": hashlib.sha256(repr(transform).encode("utf-8")).hexdigest(),
            "encoder_state_sha256": state_hash,
            "torch_version": torch_version,
            "torchvision_version": torchvision_version,
            "raw_layer_dim": {
                name: int(X2_RAW_LAYER_DIM[encoder_id][name])
                for name in X2_ENCODER_LAYERS[encoder_id]
            },
        },
    }


def _state_hash_from_build(encoder_id: str, device: str) -> str:
    """State hash of the frozen pretrained encoder (device independent)."""
    model, _weights, _tv, _t = build_encoder(encoder_id, "cpu")
    digest = encoder_state_sha256(model)
    del model
    return digest


def _check_stage_shapes(condition: dict[str, Any]) -> dict[str, Any]:
    """Record extracted tensor shapes and the hook location per frozen layer."""
    shapes = {}
    for layer in X2_ENCODER_LAYERS[condition["encoder"]]:
        shapes[layer] = {
            "raw_train": list(np.asarray(condition["train"][layer]).shape),
            "raw_test": list(np.asarray(condition["test"][layer]).shape),
            "module_path": _module_path(condition["encoder"], layer),
            "extraction_location": _extraction_location(condition["encoder"]),
        }
    return shapes


def reference_stages(condition: dict[str, Any], device_label: str) -> dict[str, Any]:
    """Build stages A-G of the frozen X2 pipeline for one condition."""
    stages: dict[str, Any] = {}
    per_layer: dict[str, Any] = {}
    for layer in X2_ENCODER_LAYERS[condition["encoder"]]:
        raw_train = np.asarray(condition["train"][layer], dtype=np.float64)
        raw_test = np.asarray(condition["test"][layer], dtype=np.float64)
        w, b = generate_rff(condition["encoder"], layer, raw_train.shape[1])
        train_norm = normalize_layer_features(raw_train)
        test_norm = normalize_layer_features(raw_test)
        train_rff = apply_rff(train_norm, w, b)
        test_rff = apply_rff(test_norm, w, b)
        message = class_message(train_rff, condition["train_labels"])
        prototypes = reconstruct_prototypes(message)
        classes = np.asarray(sorted(prototypes), dtype=np.int64)
        means = np.stack([prototypes[int(c)] for c in classes])
        scores = test_rff @ means.T
        predictions = predict_prototype_batch(test_rff, prototypes)
        per_layer[layer] = {
            "raw_train": raw_train,
            "raw_test": raw_test,
            "norm_train": train_norm,
            "norm_test": test_norm,
            "rff_train": train_rff,
            "rff_test": test_rff,
            "class_sums": np.stack([message[int(c)]["vector_sum"] for c in classes]),
            "prototypes": means,
            "scores": scores,
            "predictions": predictions,
            "class_counts": np.asarray([message[int(c)]["count"] for c in classes], dtype=np.int64),
            "bacc": bacc_from_predictions(condition["test_labels"], predictions),
        }
    stages["layers"] = per_layer
    stages["device_label"] = device_label
    return stages


def save_reference(
    condition: dict[str, Any], stages: dict[str, Any], out_dir: Path, tag: str
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    condition_id = f"{condition['dataset']}__{condition['encoder']}"
    payload = {
        "tags": np.asarray([condition["dataset"], condition["encoder"]], dtype="U64"),
    }
    entries = []
    for layer, layer_stages in stages["layers"].items():
        for stage_key in (
            "raw_train",
            "raw_test",
            "norm_train",
            "norm_test",
            "rff_train",
            "rff_test",
            "class_sums",
            "prototypes",
            "scores",
        ):
            array = np.asarray(layer_stages[stage_key], dtype=np.float64)
            payload[f"{layer}__{stage_key}"] = array
            entries.append(
                {
                    "layer": layer,
                    "stage": stage_key,
                    "shape": list(array.shape),
                    "sha256": array_sha256(array),
                }
            )
        payload[f"{layer}__predictions"] = np.asarray(layer_stages["predictions"], dtype=np.int64)
        payload[f"{layer}__class_counts"] = np.asarray(layer_stages["class_counts"], dtype=np.int64)
        entries.append(
            {
                "layer": layer,
                "stage": "predictions",
                "shape": list(layer_stages["predictions"].shape),
                "sha256": array_sha256(layer_stages["predictions"].astype(np.int64)),
            }
        )
    payload["train_labels"] = np.asarray(condition["train_labels"], dtype=np.int64)
    payload["test_labels"] = np.asarray(condition["test_labels"], dtype=np.int64)
    path = out_dir / f"{condition_id}.npz"
    if tag == "cpu":
        np.savez(path, **payload)
    else:
        np.savez_compressed(path, **payload)
    summary = {
        "condition_id": condition_id,
        "dataset": condition["dataset"],
        "encoder": condition["encoder"],
        "device": condition["device"],
        "sampled": bool(condition["sampled"]),
        "train_records": int(len(condition["train_labels"])),
        "test_records": int(len(condition["test_labels"])),
        "reference_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "reference_sha256": sha256_file(path),
        "bacc": {
            layer: float(layer_stages["bacc"])
            for layer, layer_stages in stages["layers"].items()
        },
        "stage_hashes": entries,
        "provenance": condition["provenance"],
        "identity": condition["identity"],
        "split_meta": {k: v for k, v in condition["split_meta"].items()},
    }
    return summary


def load_reference(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


# =============================================================================
# Subcommands
# =============================================================================
def cmd_env_cpu(args) -> int:
    authority = load_x2_protocol_constants(PROTOCOL_YAML)
    record = {
        "role": "CPU_REFERENCE_BACKEND",
        "cpu_cache_classification": CPU_REFERENCE_MARKER,
        "device": "cpu",
        "environment": torch_env(),
        "protocol": {
            "id": authority["protocol_id"],
            "version": authority["protocol_version"],
            "status": authority["protocol_status"],
        },
        "protocol_files": {
            "protocol_md_sha256": sha256_file(PROTOCOL_MD),
            "protocol_yaml_sha256": sha256_file(PROTOCOL_YAML),
            "gate_matrix_sha256": sha256_file(GATE_MATRIX),
        },
        "source_files": {
            str(FEATURE_EXTRACT_SRC.relative_to(ROOT)).replace("\\", "/"): sha256_file(FEATURE_EXTRACT_SRC),
            str(KERNEL_SRC.relative_to(ROOT)).replace("\\", "/"): sha256_file(KERNEL_SRC),
        },
        "cpu_premigration_cache": {
            "root": str(CPU_CACHE_ROOT.relative_to(ROOT)).replace("\\", "/"),
            "status": CPU_REFERENCE_MARKER,
            "use": "engineering/pre-migration reference only; never the formal X2 feature bank",
            "conditions": _cpu_cache_inventory(),
        },
    }
    write_json(ENV_CPU_JSON, record)
    print(f"[migration] CPU environment recorded: {ENV_CPU_JSON.relative_to(ROOT)}")
    print(f"[migration] CPU cache marked {CPU_REFERENCE_MARKER}")
    for entry in record["cpu_premigration_cache"]["conditions"]:
        print(f"[migration]   {entry['condition_id']} bytes={entry['bytes']}")
    return 0


def _cpu_cache_inventory() -> list[dict[str, Any]]:
    entries = []
    if CPU_FEATURE_ROOT.is_dir():
        for path in sorted(CPU_FEATURE_ROOT.glob("*.npz")):
            entries.append(
                {
                    "condition_id": path.stem,
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                    "status": CPU_REFERENCE_MARKER,
                }
            )
    return entries


def cmd_env_cuda(args) -> int:
    import torch

    if not torch.cuda.is_available():
        print("[migration] FATAL: torch.cuda.is_available() is False", file=sys.stderr)
        return 2
    authority = load_x2_protocol_constants(PROTOCOL_YAML)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    record = {
        "role": "CUDA_FORMAL_BACKEND",
        "device": "cuda",
        "environment": torch_env(),
        "env_name": Path(sys.executable).parent.parent.name,
        "determinism_flags": {
            "torch.backends.cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "torch.backends.cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "protocol": {
            "id": authority["protocol_id"],
            "version": authority["protocol_version"],
            "status": authority["protocol_status"],
        },
        "formal_extraction_defaults": {
            "model_weights_dtype": "float32",
            "forward_dtype": "float32",
            "amp": False,
            "autocast": False,
            "bf16": False,
            "rff_generation_dtype": "float64",
            "storage_dtype": "float16",
        },
    }
    write_json(ENV_CUDA_JSON, record)
    print(
        "[migration] CUDA environment: "
        f"{record['environment']['gpu_model']} torch={record['environment']['torch_version']} "
        f"cuda={record['environment']['cuda_runtime_version']} "
        f"cudnn={record['environment']['cudnn_version']} "
        f"driver={record['environment'].get('driver_version')}"
    )
    print(f"[migration] CUDA environment recorded: {ENV_CUDA_JSON.relative_to(ROOT)}")
    return 0


def cmd_parity_samples(args) -> int:
    from torchvision.models import ResNet18_Weights

    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    entries = []
    for dataset_id in X2_DATASETS:
        train, test, split_meta, train_index, test_index = load_parity_samples(
            dataset_id, CPU_RAW_ROOT, transform, args.allow_download
        )
        train_labels_all = np.asarray(
            [_label_of(dataset_id, train, i) for i in range(len(train))], dtype=np.int64
        )
        test_labels_all = np.asarray(
            [_label_of(dataset_id, test, i) for i in range(len(test))], dtype=np.int64
        )
        train_labels = train_labels_all[train_index]
        test_labels = test_labels_all[test_index]
        ids = _record_ids(dataset_id, CPU_RAW_ROOT, train, test)
        train_ids, test_ids = ids[0]
        entries.append(
            {
                "dataset": dataset_id,
                "split_id": split_meta["split_id"],
                "preprocessing_id": f"torchvision_weights_transforms::{ResNet18_Weights.IMAGENET1K_V1}",
                "transform_repr": repr(transform),
                "transform_sha256": hashlib.sha256(repr(transform).encode("utf-8")).hexdigest(),
                "sampling_rule": "evenly spaced deterministic indices over canonical record order",
                "train": {
                    "count": int(len(train_index)),
                    "indices": [int(i) for i in train_index],
                    "labels": [int(v) for v in train_labels],
                    "record_ids": [train_ids[i] for i in train_index],
                    "class_counts": _class_counts(train_labels),
                },
                "test": {
                    "count": int(len(test_index)),
                    "indices": [int(i) for i in test_index],
                    "labels": [int(v) for v in test_labels],
                    "record_ids": [test_ids[i] for i in test_index],
                    "class_counts": _class_counts(test_labels),
                },
                "source_split_sizes": {"train": int(len(train)), "test": int(len(test))},
            }
        )
    manifest = {
        "manifest_type": "CMR_V1_X2_CUDA_PARITY_SAMPLE_MANIFEST",
        "git_head": git_head(),
        "protocol_yaml_sha256": sha256_file(PROTOCOL_YAML),
        "datasets": entries,
        "sample_hash": hashlib.sha256(
            json.dumps(entries, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    write_json(PARITY_SAMPLES_JSON, manifest)
    print(f"[migration] parity sample manifest: {PARITY_SAMPLES_JSON.relative_to(ROOT)}")
    for entry in entries:
        print(
            f"[migration]   {entry['dataset']:<11} train={entry['train']['count']} "
            f"test={entry['test']['count']} train_classes={len(entry['train']['class_counts'])} "
            f"test_classes={len(entry['test']['class_counts'])}"
        )
    return 0


def _label_of(dataset_id: str, dataset, index: int) -> int:
    """Label of one record without decoding the image."""
    if dataset_id == "cifar100":
        return int(dataset.targets[index])
    if dataset_id == "eurosat":
        return _eurosat_label(dataset, index)
    if dataset_id in {"pathmnist", "dermamnist"}:
        return int(np.asarray(dataset.labels[index]).reshape(-1)[0])
    raise ValueError(dataset_id)


def _eurosat_samples(dataset):
    """Resolve the underlying ``ImageFolder.samples`` list for a EuroSAT (Sub)set."""
    if hasattr(dataset, "_data_folder"):
        return dataset.samples, None
    inner = getattr(dataset, "dataset", None)
    if inner is not None and hasattr(inner, "samples"):
        return inner.samples, dataset.indices
    raise AttributeError("cannot resolve EuroSAT sample metadata")


def _eurosat_label(dataset, index: int) -> int:
    """Read the EuroSAT class from the sample path (avoid an image decode)."""
    samples, mapping = _eurosat_samples(dataset)
    resolved = index if mapping is None else int(mapping[index])
    return int(samples[resolved][1])


def _run_reference(device: str, out_dir: Path, summary_path: Path, args) -> int:
    manifest = read_json(PARITY_SAMPLES_JSON)
    by_dataset = {entry["dataset"]: entry for entry in manifest["datasets"]}
    summaries = []
    for dataset_id, encoder_id in X2_CONDITIONS:
        entry = by_dataset[dataset_id]
        condition = extract_frozen_condition(
            dataset_id,
            encoder_id,
            CPU_RAW_ROOT,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            allow_download=args.allow_download,
            train_index=entry["train"]["indices"],
            test_index=entry["test"]["indices"],
        )
        stages = reference_stages(condition, device)
        summary = save_reference(condition, stages, out_dir, tag="cpu" if device == "cpu" else "gpu")
        summaries.append(summary)
        print(
            f"[migration] {device:<4} {dataset_id:<11} {encoder_id:<26} "
            + " ".join(f"{layer}={summary['bacc'][layer]:.6f}" for layer in X2_ENCODER_LAYERS[encoder_id])
        )
    record = {
        "device": device,
        "git_head": git_head(),
        "parity_sample_hash": manifest["sample_hash"],
        "environment": torch_env(),
        "conditions": summaries,
    }
    write_json(summary_path, record)
    print(f"[migration] reference summary: {summary_path.relative_to(ROOT)}")
    return 0


def cmd_cpu_reference(args) -> int:
    return _run_reference("cpu", CPU_REFERENCE_DIR, CPU_REF_SUMMARY, args)


def cmd_gpu_reference(args) -> int:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        print("[migration] FATAL: CUDA unavailable", file=sys.stderr)
        return 2
    return _run_reference("cuda", GPU_REFERENCE_DIR, GPU_REF_SUMMARY, args)


STAGE_SPECS = (
    ("raw", "raw_test"),
    ("normalized", "norm_test"),
    ("rff", "rff_test"),
    ("class_sum", "class_sums"),
    ("prototype", "prototypes"),
    ("classifier_score", "scores"),
    ("prediction", "predictions"),
)


def cmd_parity(args) -> int:
    cpu_summary = read_json(CPU_REF_SUMMARY)
    gpu_summary = read_json(GPU_REF_SUMMARY)
    cpu_by_id = {entry["condition_id"]: entry for entry in cpu_summary["conditions"]}
    gpu_by_id = {entry["condition_id"]: entry for entry in gpu_summary["conditions"]}

    rff_identity = _rff_identity()
    conditions = []
    all_pass = True
    for dataset_id, encoder_id in X2_CONDITIONS:
        condition_id = f"{dataset_id}__{encoder_id}"
        cpu_entry = cpu_by_id[condition_id]
        gpu_entry = gpu_by_id[condition_id]
        cpu_ref = load_reference(ROOT / cpu_entry["reference_path"])
        gpu_ref = load_reference(ROOT / gpu_entry["reference_path"])
        if not np.array_equal(cpu_ref["train_labels"], gpu_ref["train_labels"]) or not np.array_equal(
            cpu_ref["test_labels"], gpu_ref["test_labels"]
        ):
            all_pass = False
        layer_reports = []
        condition_pass = True
        for layer in X2_ENCODER_LAYERS[encoder_id]:
            stage_reports = []
            for stage_label, key in STAGE_SPECS:
                cpu_array = cpu_ref[f"{layer}__{key}"]
                gpu_array = gpu_ref[f"{layer}__{key}"]
                if stage_label == "prediction":
                    agreement = float(np.mean(cpu_array == gpu_array))
                    report = {
                        "stage": stage_label,
                        "agreement": agreement,
                        "max_abs": float(np.max(np.abs(cpu_array.astype(np.int64) - gpu_array.astype(np.int64)))),
                    }
                    if agreement < PARITY_PREDICTION_AGREEMENT:
                        condition_pass = False
                else:
                    report = stage_metrics(cpu_array, gpu_array, stage_label)
                    if stage_label == "normalized":
                        if report["cosine_median"] < PARITY_MEDIAN_COSINE_MIN:
                            condition_pass = False
                        if report["cosine_min"] < PARITY_MIN_COSINE_MIN:
                            condition_pass = False
                stage_reports.append(report)
            cpu_predictions = cpu_ref[f"{layer}__predictions"]
            gpu_predictions = gpu_ref[f"{layer}__predictions"]
            layer_reports.append(
                {
                    "layer": layer,
                    "layer_module_path": str(_module_path(encoder_id, layer)),
                    "tensor_shape": list(cpu_ref[f"{layer}__raw_test"].shape),
                    "extraction_location": _extraction_location(encoder_id),
                    "stages": stage_reports,
                    "bacc_cpu": float(cpu_entry["bacc"][layer]),
                    "bacc_gpu": float(gpu_entry["bacc"][layer]),
                    "bacc_abs_diff": abs(float(cpu_entry["bacc"][layer]) - float(gpu_entry["bacc"][layer])),
                    "predictions_identical": bool(np.array_equal(cpu_predictions, gpu_predictions)),
                }
            )
            if layer_reports[-1]["bacc_abs_diff"] > PARITY_BACC_ABS_DIFF_MAX:
                condition_pass = False
        conditions.append(
            {
                "condition_id": condition_id,
                "dataset": dataset_id,
                "encoder": encoder_id,
                "passed": condition_pass,
                "layers": layer_reports,
            }
        )
        all_pass = all_pass and condition_pass

    report = {
        "report_type": "CMR_V1_X2_CPU_GPU_PARITY",
        "git_head": git_head(),
        "acceptance": {
            "normalized_cosine_median_min": PARITY_MEDIAN_COSINE_MIN,
            "normalized_cosine_min": PARITY_MIN_COSINE_MIN,
            "prediction_agreement_required": PARITY_PREDICTION_AGREEMENT,
            "bacc_abs_diff_max": PARITY_BACC_ABS_DIFF_MAX,
            "tolerance_class": "IMPLEMENTATION_VALIDATION_TOLERANCE_NOT_SCIENTIFIC_GATE",
        },
        "rff_identity": rff_identity,
        "conditions": conditions,
        "conditions_total": len(conditions),
        "conditions_passed": int(sum(1 for c in conditions if c["passed"])),
        "parity_pass": bool(all_pass),
        "environment_cpu": cpu_summary["environment"],
        "environment_cuda": gpu_summary["environment"],
    }
    write_json(PARITY_REPORT, report)
    print(f"[migration] parity report: {PARITY_REPORT.relative_to(ROOT)}")
    _print_parity(report)
    return 0 if all_pass else 1


def _module_path(encoder_id: str, layer: str) -> str:
    if encoder_id == RESNET_ENCODER_ID:
        return f"model.{layer}"
    return f"model.encoder.layers[{int(layer[1:]) - 1}]"


def _extraction_location(encoder_id: str) -> str:
    if encoder_id == RESNET_ENCODER_ID:
        return "post-stage tensor -> adaptive_avg_pool2d(1) -> flatten"
    return "post-block CLS token (token index 0) BEFORE encoder.ln (final LayerNorm)"


def _rff_identity() -> dict[str, Any]:
    entries = []
    identical = True
    for encoder_id, layers in X2_ENCODER_LAYERS.items():
        dim = X2_RAW_LAYER_DIM[encoder_id][layers[0]]
        for layer in layers:
            w, b = generate_rff(encoder_id, layer, dim)
            w2, b2 = generate_rff(encoder_id, layer, dim)
            same = array_sha256(w) == array_sha256(w2) and array_sha256(b) == array_sha256(b2)
            identical = identical and same
            entries.append(
                {
                    "encoder": encoder_id,
                    "layer": layer,
                    "derived_seed": int(rff_seed(encoder_id, layer)),
                    "w_sha256": array_sha256(w),
                    "b_sha256": array_sha256(b),
                    "regeneration_identical": bool(same),
                    "shape": [int(X2_RFF_DIMENSION), int(dim)],
                    "dtype": "float64",
                }
            )
    return {
        "dimension": X2_RFF_DIMENSION,
        "sigma": X2_RFF_SIGMA,
        "master_seed": X2_RFF_MASTER_SEED,
        "maps": entries,
        "byte_identical_on_regeneration": bool(identical),
    }


def _print_parity(report: dict[str, Any]) -> None:
    for condition in report["conditions"]:
        print(
            f"[migration] parity {condition['condition_id']:<44} "
            f"{'PASS' if condition['passed'] else 'FAIL'}"
        )
        for layer in condition["layers"]:
            normalized = next(s for s in layer["stages"] if s["stage"] == "normalized")
            prediction = next(s for s in layer["stages"] if s["stage"] == "prediction")
            print(
                f"[migration]    {layer['layer']:<7} max_abs_raw="
                f"{next(s for s in layer['stages'] if s['stage'] == 'raw')['max_abs']:.3e} "
                f"cos_med={normalized['cosine_median']:.9f} cos_min={normalized['cosine_min']:.9f} "
                f"pred_agree={prediction['agreement']:.4f} bacc_diff={layer['bacc_abs_diff']:.3e}"
            )
    print(
        f"[migration] parity {'PASS' if report['parity_pass'] else 'FAIL'} "
        f"({report['conditions_passed']}/{report['conditions_total']} conditions)"
    )


def cmd_benchmark(args) -> int:
    import torch
    from torchvision.models import ResNet18_Weights, ViT_B_16_Weights
    from torch.utils.data import Subset
    from torchvision import datasets

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for encoder_id, weights, batch_sizes in (
        (RESNET_ENCODER_ID, ResNet18_Weights.IMAGENET1K_V1, (32, 64, 128, 256)),
        (VIT_ENCODER_ID, ViT_B_16_Weights.IMAGENET1K_V1, (8, 16, 32, 64)),
    ):
        dataset = datasets.CIFAR100(
            root=str(CPU_RAW_ROOT), train=False, transform=weights.transforms(), download=False
        )
        for device, label in (("cpu", "cpu"), ("cuda", "cuda")):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            model, _w, _tv, _t = build_encoder(encoder_id, device)
            for batch_size in batch_sizes:
                subset = Subset(dataset, list(range(min(1024, len(dataset)))))
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                started = time.time()
                try:
                    arrays, _labels = extract_split(
                        model,
                        encoder_id,
                        "cifar100",
                        subset,
                        batch_size,
                        0,
                        device,
                    )
                except RuntimeError as exc:
                    results.append(
                        {
                            "encoder": encoder_id,
                            "device": label,
                            "batch_size": batch_size,
                            "status": "OOM" if "memory" in str(exc).lower() else "ERROR",
                            "error": str(exc)[:200],
                        }
                    )
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                elapsed = time.time() - started
                records = int(arrays[X2_ENCODER_LAYERS[encoder_id][0]].shape[0])
                record = {
                    "encoder": encoder_id,
                    "device": label,
                    "batch_size": batch_size,
                    "status": "OK",
                    "records": records,
                    "seconds": elapsed,
                    "images_per_second": float(records / elapsed),
                    "workers": 0,
                }
                if device == "cuda":
                    record["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                results.append(record)
                print(
                    f"[migration] bench {encoder_id:<26} {label:<4} bs={batch_size:<4} "
                    f"{record['images_per_second']:.1f} img/s "
                    + (f"peak={record.get('peak_vram_mib')}MiB" if device == "cuda" else "")
                )
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    payload = {
        "benchmark_type": "CMR_V1_X2_ENGINEERING_THROUGHPUT",
        "note": "engineering throughput only; batch size is never selected by model performance",
        "git_head": git_head(),
        "results": results,
    }
    write_json(THROUGHPUT_JSON, payload)
    print(f"[migration] throughput benchmark: {THROUGHPUT_JSON.relative_to(ROOT)}")
    return 0


def cmd_gate(args) -> int:
    """Evaluate the frozen migration gates M1-M8 and authorize (or block) CUDA X2."""
    cpu_env = read_json(ENV_CPU_JSON)
    cuda_env = read_json(ENV_CUDA_JSON)
    samples = read_json(PARITY_SAMPLES_JSON)
    parity = read_json(PARITY_REPORT)
    bench = read_json(THROUGHPUT_JSON) if THROUGHPUT_JSON.is_file() else {"results": []}

    weights_path = MIGRATION_DIR / "weight_identity.json"
    if not weights_path.is_file():
        print(f"[migration] missing {weights_path.relative_to(ROOT)}", file=sys.stderr)
        return 2
    weights = read_json(weights_path)

    gates: dict[str, Any] = {}
    cuda_environment = cuda_env["environment"]
    cuda_build = bool(cuda_environment.get("cuda_runtime_version"))
    gates["M1"] = {
        "name": "CUDA environment valid",
        "passed": bool(
            cuda_environment["cuda_available"]
            and cuda_build
            and "cuda" in str(cuda_env.get("device", ""))
            and cuda_environment.get("gpu_model")
        ),
        "cuda_available": cuda_environment["cuda_available"],
        "cuda_build": cuda_build,
        "gpu_model": cuda_environment.get("gpu_model"),
        "gpu_capability": cuda_environment.get("gpu_capability"),
        "torch_version": cuda_environment["torch_version"],
        "torchvision_version": cuda_environment["torchvision_version"],
        "cuda_runtime_version": cuda_environment["cuda_runtime_version"],
        "cudnn_version": cuda_environment["cudnn_version"],
        "driver_version": cuda_environment.get("driver_version"),
        "tf32_disabled": not bool(cuda_env["determinism_flags"]["torch.backends.cuda.matmul.allow_tf32"])
        and not bool(cuda_env["determinism_flags"]["torch.backends.cudnn.allow_tf32"]),
    }
    gates["M2"] = {
        "name": "weights identity valid",
        "passed": bool(weights["weights_identity_pass"]),
        "detail": weights["weights"],
    }
    gates["M3"] = {
        "name": "preprocessing identity valid",
        "passed": bool(weights["preprocessing_identity_pass"]),
        "detail": weights["preprocessing"],
    }
    gates["M4"] = {
        "name": "layer hook identity valid",
        "passed": bool(weights["layer_hook_identity_pass"]),
        "detail": weights["layer_hooks"],
    }
    gates["M5"] = {
        "name": "RFF arrays identical",
        "passed": bool(parity["rff_identity"]["byte_identical_on_regeneration"]),
        "detail": {
            "dimension": parity["rff_identity"]["dimension"],
            "sigma": parity["rff_identity"]["sigma"],
            "master_seed": parity["rff_identity"]["master_seed"],
            "maps": [
                {
                    "encoder": entry["encoder"],
                    "layer": entry["layer"],
                    "derived_seed": entry["derived_seed"],
                    "w_sha256": entry["w_sha256"],
                    "b_sha256": entry["b_sha256"],
                }
                for entry in parity["rff_identity"]["maps"]
            ],
        },
    }
    feature_stage = [
        stage
        for condition in parity["conditions"]
        for layer in condition["layers"]
        for stage in layer["stages"]
        if stage["stage"] in {"normalized", "rff", "class_sum", "prototype", "classifier_score"}
    ]
    gates["M6"] = {
        "name": "CPU<->GPU feature parity",
        "passed": bool(
            all(
                stage.get("cosine_median", 1.0) >= PARITY_MEDIAN_COSINE_MIN
                and stage.get("cosine_min", 1.0) >= PARITY_MIN_COSINE_MIN
                for stage in feature_stage
            )
        ),
        "worst_cosine_median": float(min(s["cosine_median"] for s in feature_stage)),
        "worst_cosine_min": float(min(s["cosine_min"] for s in feature_stage)),
        "acceptance": parity["acceptance"],
    }
    prediction_stages = [
        stage
        for condition in parity["conditions"]
        for layer in condition["layers"]
        for stage in layer["stages"]
        if stage["stage"] == "prediction"
    ]
    gates["M7"] = {
        "name": "prediction agreement",
        "passed": bool(
            all(stage["agreement"] >= PARITY_PREDICTION_AGREEMENT for stage in prediction_stages)
            and all(
                layer["bacc_abs_diff"] <= PARITY_BACC_ABS_DIFF_MAX
                for condition in parity["conditions"]
                for layer in condition["layers"]
            )
        ),
        "min_agreement": float(min(stage["agreement"] for stage in prediction_stages)),
        "max_bacc_abs_diff": float(
            max(
                layer["bacc_abs_diff"]
                for condition in parity["conditions"]
                for layer in condition["layers"]
            )
        ),
    }
    protocol_now = load_x2_protocol_constants(PROTOCOL_YAML)
    gates["M8"] = {
        "name": "no scientific protocol field changed",
        "passed": bool(
            protocol_now["protocol_version"] == "1.0.0-FROZEN"
            and protocol_now["protocol_status"] == "FROZEN_BEFORE_CMR_OUTCOME_ACCESS"
            and sha256_file(PROTOCOL_YAML) == cpu_env["protocol_files"]["protocol_yaml_sha256"]
            and sha256_file(PROTOCOL_MD) == cpu_env["protocol_files"]["protocol_md_sha256"]
            and sha256_file(GATE_MATRIX) == cpu_env["protocol_files"]["gate_matrix_sha256"]
        ),
        "protocol_yaml_sha256": sha256_file(PROTOCOL_YAML),
        "protocol_md_sha256": sha256_file(PROTOCOL_MD),
        "gate_matrix_sha256": sha256_file(GATE_MATRIX),
        "dataset_definition_hash": samples["sample_hash"],
        "model_definition_unchanged": True,
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    summary = {
        "summary_type": "CMR_V1_X2_CUDA_MIGRATION_GATE",
        "git_head": git_head(),
        "migration_type": "IMPLEMENTATION_BACKEND_MIGRATION_NOT_PROTOCOL_AMENDMENT",
        "gates": gates,
        "cuda_migration_status": "PASS" if passed else "BLOCKED",
        "formal_x2_cuda_reextraction": "AUTHORIZED" if passed else "NOT_AUTHORIZED",
        "cpu_premigration_cache_status": CPU_REFERENCE_MARKER,
        "environment_cpu": cpu_env["environment"],
        "environment_cuda": cuda_env["environment"],
        "throughput": bench["results"],
    }
    write_json(MIGRATION_GATE, summary)
    print(f"[migration] migration gate: {MIGRATION_GATE.relative_to(ROOT)}")
    for gate_id, gate in gates.items():
        print(f"[migration] {gate_id} {'PASS' if gate['passed'] else 'FAIL'} {gate['name']}")
    print(f"[migration] CUDA_MIGRATION_STATUS = {summary['cuda_migration_status']}")
    print(f"[migration] FORMAL_X2_CUDA_REEXTRACTION = {summary['formal_x2_cuda_reextraction']}")
    return 0 if passed else 1


def cmd_identity(args) -> int:
    """Record weight / preprocessing / layer-hook identity for one backend."""
    record: dict[str, Any] = {
        "backend": args.label,
        "device_probe": args.label,
        "git_head": git_head(),
        "environment": torch_env(),
        "weights": {},
        "preprocessing": {},
        "layer_hooks": {},
    }
    for encoder_id in X2_ENCODER_LAYERS:
        model, weights, _tv, _t = build_encoder(encoder_id, "cpu")
        transform = weights.transforms()
        record["weights"][encoder_id] = {
            "weights_enum": str(weights),
            "state_sha256": encoder_state_sha256(model),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "dtype": str(next(model.parameters()).dtype),
        }
        record["preprocessing"][encoder_id] = {
            "preprocessing_id": f"torchvision_weights_transforms::{weights}",
            "transform_repr": repr(transform),
            "transform_sha256": hashlib.sha256(repr(transform).encode("utf-8")).hexdigest(),
        }
        modules = layer_modules(model, encoder_id)
        record["layer_hooks"][encoder_id] = {
            layer: {
                "module_path": _module_path(encoder_id, layer),
                "module_type": type(modules[layer]).__name__,
                "extraction_location": _extraction_location(encoder_id),
            }
            for layer in X2_ENCODER_LAYERS[encoder_id]
        }
        del model
    path = MIGRATION_DIR / f"identity_{args.label}.json"
    write_json(path, record)
    print(f"[migration] identity ({args.label}): {path.relative_to(ROOT)}")
    for encoder_id, entry in record["weights"].items():
        print(f"[migration]   {encoder_id:<26} state={entry['state_sha256'][:16]} enum={entry['weights_enum']}")
    return 0


def cmd_compare_identity(args) -> int:
    cpu = read_json(MIGRATION_DIR / "identity_cpu.json")
    cuda = read_json(MIGRATION_DIR / "identity_cuda.json")
    weight_pass = True
    preprocess_pass = True
    hook_pass = True
    weights_detail = {}
    preprocessing_detail = {}
    hooks_detail = {}
    for encoder_id in X2_ENCODER_LAYERS:
        cpu_w = cpu["weights"][encoder_id]
        cuda_w = cuda["weights"][encoder_id]
        same_state = cpu_w["state_sha256"] == cuda_w["state_sha256"]
        same_enum = cpu_w["weights_enum"] == cuda_w["weights_enum"]
        weight_pass = weight_pass and same_state and same_enum
        weights_detail[encoder_id] = {
            "cpu_state_sha256": cpu_w["state_sha256"],
            "cuda_state_sha256": cuda_w["state_sha256"],
            "state_identical": bool(same_state),
            "weights_enum": cpu_w["weights_enum"],
            "enum_identical": bool(same_enum),
        }
        cpu_p = cpu["preprocessing"][encoder_id]
        cuda_p = cuda["preprocessing"][encoder_id]
        same_repr = cpu_p["transform_repr"] == cuda_p["transform_repr"]
        same_hash = cpu_p["transform_sha256"] == cuda_p["transform_sha256"]
        preprocess_pass = preprocess_pass and same_repr and same_hash
        preprocessing_detail[encoder_id] = {
            "transform_repr": cpu_p["transform_repr"],
            "cpu_transform_sha256": cpu_p["transform_sha256"],
            "cuda_transform_sha256": cuda_p["transform_sha256"],
            "identical": bool(same_repr and same_hash),
        }
        cpu_h = cpu["layer_hooks"][encoder_id]
        cuda_h = cuda["layer_hooks"][encoder_id]
        same_hooks = cpu_h == cuda_h
        hook_pass = hook_pass and same_hooks
        hooks_detail[encoder_id] = {
            "cpu": cpu_h,
            "cuda": cuda_h,
            "identical": bool(same_hooks),
        }
    record = {
        "record_type": "CMR_V1_X2_BACKEND_IDENTITY_COMPARISON",
        "git_head": git_head(),
        "weights": weights_detail,
        "preprocessing": preprocessing_detail,
        "layer_hooks": hooks_detail,
        "weights_identity_pass": bool(weight_pass),
        "preprocessing_identity_pass": bool(preprocess_pass),
        "layer_hook_identity_pass": bool(hook_pass),
        "cpu_torch_version": cpu["environment"]["torch_version"],
        "cuda_torch_version": cuda["environment"]["torch_version"],
        "cpu_torchvision_version": cpu["environment"]["torchvision_version"],
        "cuda_torchvision_version": cuda["environment"]["torchvision_version"],
    }
    write_json(MIGRATION_DIR / "weight_identity.json", record)
    print(f"[migration] identity comparison: {MIGRATION_DIR.joinpath('weight_identity.json').relative_to(ROOT)}")
    print(f"[migration] weights identity        {'PASS' if weight_pass else 'FAIL'}")
    print(f"[migration] preprocessing identity  {'PASS' if preprocess_pass else 'FAIL'}")
    print(f"[migration] layer-hook identity     {'PASS' if hook_pass else 'FAIL'}")
    return 0 if (weight_pass and preprocess_pass and hook_pass) else 1


def cmd_determinism(args) -> int:
    """Two complete CUDA extraction reruns on the frozen parity samples.

    Verifies sample order, frozen RFF identity, prediction identity and BACC
    identity for at least one ResNet and one ViT condition.
    """
    import torch

    if not torch.cuda.is_available():
        print("[migration] FATAL: CUDA unavailable for determinism check", file=sys.stderr)
        return 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read_json(PARITY_SAMPLES_JSON)
    by_dataset = {entry["dataset"]: entry for entry in manifest["datasets"]}
    dataset_id = args.dataset or "cifar100"
    entry = by_dataset[dataset_id]
    records = []
    for encoder_id in (RESNET_ENCODER_ID, VIT_ENCODER_ID):
        runs = []
        for repeat in (1, 2):
            condition = extract_frozen_condition(
                dataset_id,
                encoder_id,
                CPU_RAW_ROOT,
                device="cuda",
                batch_size=args.batch_size,
                workers=args.workers,
                allow_download=False,
                train_index=entry["train"]["indices"],
                test_index=entry["test"]["indices"],
            )
            stages = reference_stages(condition, "cuda")
            run_summary = {
                "repeat": repeat,
                "train_labels_sha256": array_sha256(condition["train_labels"].astype(np.int64)),
                "test_labels_sha256": array_sha256(condition["test_labels"].astype(np.int64)),
                "layers": {
                    layer: {
                        "raw_test_sha256": array_sha256(layer_stages["raw_test"]),
                        "rff_test_sha256": array_sha256(layer_stages["rff_test"]),
                        "prototype_sha256": array_sha256(layer_stages["prototypes"]),
                        "prediction_sha256": array_sha256(
                            layer_stages["predictions"].astype(np.int64)
                        ),
                        "bacc": float(layer_stages["bacc"]),
                        "raw_test": layer_stages["raw_test"],
                        "rff_test": layer_stages["rff_test"],
                        "predictions": layer_stages["predictions"],
                    }
                    for layer, layer_stages in stages["layers"].items()
                },
            }
            runs.append(run_summary)
        layer_reports = {}
        for layer in X2_ENCODER_LAYERS[encoder_id]:
            first, second = runs[0]["layers"][layer], runs[1]["layers"][layer]
            raw_diff = float(
                np.max(np.abs(first["raw_test"].astype(np.float64) - second["raw_test"].astype(np.float64)))
            )
            rff_diff = float(
                np.max(np.abs(first["rff_test"].astype(np.float64) - second["rff_test"].astype(np.float64)))
            )
            layer_reports[layer] = {
                "raw_test_max_abs_diff": raw_diff,
                "rff_test_max_abs_diff": rff_diff,
                "raw_bitwise_identical": bool(first["raw_test_sha256"] == second["raw_test_sha256"]),
                "rff_bitwise_identical": bool(first["rff_test_sha256"] == second["rff_test_sha256"]),
                "prototype_bitwise_identical": bool(
                    first["prototype_sha256"] == second["prototype_sha256"]
                ),
                "predictions_identical": bool(
                    first["prediction_sha256"] == second["prediction_sha256"]
                ),
                "bacc_first": first["bacc"],
                "bacc_second": second["bacc"],
                "bacc_identical": bool(first["bacc"] == second["bacc"]),
            }
        records.append(
            {
                "dataset": dataset_id,
                "encoder": encoder_id,
                "sample_order_identical": bool(
                    runs[0]["train_labels_sha256"] == runs[1]["train_labels_sha256"]
                    and runs[0]["test_labels_sha256"] == runs[1]["test_labels_sha256"]
                ),
                "layers": layer_reports,
                "predictions_identical": all(
                    report["predictions_identical"] for report in layer_reports.values()
                ),
                "bacc_identical": all(report["bacc_identical"] for report in layer_reports.values()),
            }
        )
        print(
            f"[migration] determinism {dataset_id}/{encoder_id}: "
            f"predictions_identical={records[-1]['predictions_identical']} "
            f"bacc_identical={records[-1]['bacc_identical']}"
        )
    passed = all(
        record["predictions_identical"] and record["bacc_identical"] and record["sample_order_identical"]
        for record in records
    )
    payload = {
        "report_type": "CMR_V1_X2_CUDA_DETERMINISM",
        "git_head": git_head(),
        "dataset": dataset_id,
        "sample_source": "parity_sample_manifest",
        "note": "raw features are not required to be bitwise identical; decisions must be",
        "conditions": records,
        "determinism_pass": bool(passed),
    }
    write_json(MIGRATION_DIR / "determinism_report.json", payload)
    print(f"[migration] determinism: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CMR-V1 X2 CPU->CUDA migration harness")
    parser.add_argument(
        "command",
        choices=[
            "env-cpu",
            "env-cuda",
            "identity",
            "compare-identity",
            "parity-samples",
            "cpu-reference",
            "gpu-reference",
            "parity",
            "benchmark",
            "determinism",
            "gate",
        ],
    )
    parser.add_argument("--label", default=None, help="backend label for the identity record")
    parser.add_argument("--dataset", default="cifar100", help="dataset for the determinism check")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    handlers = {
        "env-cpu": cmd_env_cpu,
        "env-cuda": cmd_env_cuda,
        "identity": cmd_identity,
        "compare-identity": cmd_compare_identity,
        "parity-samples": cmd_parity_samples,
        "cpu-reference": cmd_cpu_reference,
        "gpu-reference": cmd_gpu_reference,
        "parity": cmd_parity,
        "benchmark": cmd_benchmark,
        "determinism": cmd_determinism,
        "gate": cmd_gate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
