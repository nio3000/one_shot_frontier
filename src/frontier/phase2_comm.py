from __future__ import annotations


def accuracy_per_megabyte(balanced_accuracy: float, upload_bytes: int) -> float:
    mb = max(float(upload_bytes) / (1024.0 * 1024.0), 1e-12)
    return float(balanced_accuracy / mb)
