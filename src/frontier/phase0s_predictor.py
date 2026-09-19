from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _feature_frame(cells: pd.DataFrame) -> pd.DataFrame:
    H = cells["actual_H_sigma_mean"].astype(float).to_numpy()
    S = cells["effective_support_mean"].astype(float).to_numpy()
    logS = np.log2(S)
    return pd.DataFrame({"H": H, "log2_Seff": logS, "H_x_log2_Seff": H * logS}, index=cells.index)


def _fit_spec(cells: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    X = _feature_frame(cells)
    y = cells["tau_0_5"].to_numpy(float)
    m = LinearRegression().fit(X[features], y)
    return {
        "features": features,
        "intercept": float(m.intercept_),
        "coefficients": {f: float(v) for f, v in zip(features, m.coef_)},
    }


def predict_spec(cells: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    X = _feature_frame(cells)
    z = np.full(len(cells), float(spec["intercept"]), dtype=float)
    for f in spec["features"]:
        z += float(spec["coefficients"][f]) * X[f].to_numpy(float)
    return np.clip(z, 0.0, 1.0)


def freeze_predictor_from_phase0h(
    *,
    phase0h_predictor_fit: Path,
    phase0h_cells: Path,
    phase0h_run_summary: Path,
    output: Path,
    fitting_git_head: str,
) -> dict[str, Any]:
    cells = pd.read_csv(phase0h_cells)
    fit = json.loads(phase0h_predictor_fit.read_text(encoding="utf-8")) if phase0h_predictor_fit.exists() else {}
    mold = None
    try:
        mold = fit["models"]["M_old"]["all_discovery_fit"]
    except Exception:
        mold = None
    source_mode = "phase0h_predictor_fit_json"
    if mold is None:
        mold = _fit_spec(cells, ["H", "log2_Seff", "H_x_log2_Seff"])
        source_mode = "deterministic_refit_phase0h_cells"

    B0 = {"features": [], "intercept": float(cells["tau_0_5"].mean()), "coefficients": {}}
    B1 = _fit_spec(cells, ["H"])
    B2 = _fit_spec(cells, ["log2_Seff"])
    run_summary = json.loads(phase0h_run_summary.read_text(encoding="utf-8")) if phase0h_run_summary.exists() else {}

    payload = {
        "freeze_type": "PHASE0S_PREDICTOR_FREEZE",
        "primary_model": "M_old",
        "source_mode": source_mode,
        "source_phase0h_git_head": run_summary.get("git_head"),
        "source_phase0h_protocol_sha256": run_summary.get("protocol_sha256"),
        "source_phase0h_cells_sha256": file_sha256(phase0h_cells),
        "source_phase0h_predictor_fit_sha256": file_sha256(phase0h_predictor_fit) if phase0h_predictor_fit.exists() else None,
        "source_row_count": int(len(cells)),
        "fitting_git_head": fitting_git_head,
        "models": {"M_old": mold, "B0_constant": B0, "B1_H_only": B1, "B2_Seff_only": B2},
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["predictor_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
