from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "phase5_nature_sensitivity_addendum.yaml"
DEFAULT_PHASE5_PROTOCOL = ROOT / "configs" / "phase5_protocol.yaml"
DEFAULT_PAIR_MANIFEST = ROOT / "configs" / "phase5_tier1_pair_manifest.json"
DEFAULT_FEATURE_MANIFEST = ROOT / "configs" / "phase5_tier1_feature_manifest.json"
DEFAULT_PRIMARY_RESULTS = ROOT / "evidence" / "phase5" / "pair_results.csv"
DEFAULT_OUT = ROOT / "runs" / "phase5_nature_sensitivity"

EXPECTED_PAIR_MANIFEST_STATUS = "FROZEN_BEFORE_RISK_EVALUATION"
EXPECTED_FEATURE_MANIFEST_STATUS = "FROZEN_AFTER_AUTHORIZED_UNBLIND"
EXPECTED_SENSITIVITY_STATUS = "POST_PRIMARY_FROZEN_BEFORE_SENSITIVITY_EXECUTION"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "NO_GIT_HEAD"


def tracked_clean(path: Path) -> bool:
    try:
        rel = str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return False
    a = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if a.returncode != 0:
        return False
    b = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", rel],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return b.returncode == 0


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _project_modules():
    sys.path.insert(0, str(ROOT / "src"))
    from frontier.phase1_projection import project_features
    from frontier.phase5_featurebank import load_tier1_feature_bank
    from frontier.phase5_domain_tools import deterministic_group_roles, no_group_leakage
    from frontier.phase5_summary_geometry import remap_classes
    from frontier.phase5_eval import domain_risk_curve
    from frontier.phase4_eval import pair_decision_metrics
    return {
        "project_features": project_features,
        "load_tier1_feature_bank": load_tier1_feature_bank,
        "deterministic_group_roles": deterministic_group_roles,
        "no_group_leakage": no_group_leakage,
        "remap_classes": remap_classes,
        "domain_risk_curve": domain_risk_curve,
        "pair_decision_metrics": pair_decision_metrics,
    }


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    gate_family: str
    alpha0: float
    projection_seed: int
    split_seed: int
    is_primary_dense: bool = False


def dense_tau_grid(cfg: dict) -> list[float]:
    spec = cfg["dense_tau_grid"]
    start, stop, step = float(spec["start"]), float(spec["stop"]), float(spec["step"])
    n = int(round((stop - start) / step))
    vals = [round(start + i * step, 12) for i in range(n + 1)]
    if abs(vals[-1] - stop) > 1e-10:
        raise ValueError("dense_tau_grid does not land exactly on stop")
    return vals


def build_scenarios(cfg: dict) -> list[Scenario]:
    alpha = [float(x) for x in cfg["alpha0_values"]]
    proj = [int(x) for x in cfg["projection_seeds"]]
    split = [int(x) for x in cfg["group_split_seeds"]]
    primary_alpha = 0.10
    primary_proj = proj[0]
    primary_split = split[0]
    if primary_alpha not in alpha:
        raise ValueError("alpha0_values must include primary alpha0=0.10")
    scenarios = [
        Scenario(
            "C1_dense_primary", "C1",
            primary_alpha, primary_proj, primary_split, True
        )
    ]
    for a in alpha:
        if abs(a - primary_alpha) > 1e-12:
            scenarios.append(
                Scenario(f"C2_alpha_{str(a).replace('.', 'p')}", "C2", a, primary_proj, primary_split)
            )
    for seed in proj[1:]:
        scenarios.append(
            Scenario(f"C3_projection_{seed}", "C3", primary_alpha, seed, primary_split)
        )
    for seed in split[1:]:
        scenarios.append(
            Scenario(f"C4_split_{seed}", "C4", primary_alpha, primary_proj, seed)
        )
    ids = [s.scenario_id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario ids")
    return scenarios


def _pair_key(condition_id: str, pair_id: str) -> str:
    return f"{condition_id}||{pair_id}"


def _primary_robust_keys(primary_results: pd.DataFrame, expected_count: int) -> set[str]:
    if len(primary_results) != 56:
        raise RuntimeError(f"Primary Phase5 authority mismatch: expected 56 rows, got {len(primary_results)}")
    flag = primary_results["robust_decision_critical"].astype(str).str.lower().eq("true")
    robust = primary_results[flag].copy()
    if len(robust) != int(expected_count):
        raise RuntimeError(
            f"Primary robust count mismatch: config={expected_count}, authority={len(robust)}"
        )
    return {
        _pair_key(str(r.condition_id), str(r.pair_id))
        for r in robust.itertuples(index=False)
    }


def _manifest_pair_count(pair_manifest: dict) -> int:
    return int(sum(int(c["n_selected_pairs"]) for c in pair_manifest["conditions"]))


def _feature_entry_map(feature_manifest: dict) -> dict[str, dict]:
    return {str(e["bank_id"]): dict(e) for e in feature_manifest["banks"]}


def _validate_preflight(
    cfg: dict,
    phase5_cfg: dict,
    pair_manifest: dict,
    feature_manifest: dict,
    primary_results: pd.DataFrame,
    formal_git_guard: bool,
    protocol_path: Path,
) -> tuple[set[str], dict[str, Any]]:
    if cfg.get("status") != EXPECTED_SENSITIVITY_STATUS:
        raise RuntimeError(f"Sensitivity protocol status must be {EXPECTED_SENSITIVITY_STATUS}")
    if bool(cfg.get("pair_manifest_mutable", True)):
        raise RuntimeError("Sensitivity protocol must set pair_manifest_mutable=false")
    if pair_manifest.get("status") != EXPECTED_PAIR_MANIFEST_STATUS:
        raise RuntimeError("Original pair manifest is not the frozen Phase5 authority")
    if feature_manifest.get("status") != EXPECTED_FEATURE_MANIFEST_STATUS:
        raise RuntimeError("Feature manifest status is not authoritative")
    if _manifest_pair_count(pair_manifest) != 56:
        raise RuntimeError("Frozen pair manifest must contain exactly 56 selected pair rows")
    original_robust = _primary_robust_keys(
        primary_results, int(cfg["primary_robust_pair_count"])
    )
    if formal_git_guard:
        for p in [
            protocol_path, DEFAULT_PHASE5_PROTOCOL, DEFAULT_PAIR_MANIFEST,
            DEFAULT_FEATURE_MANIFEST, DEFAULT_PRIMARY_RESULTS,
        ]:
            if not tracked_clean(p):
                raise RuntimeError(f"Formal run blocked: authority must be Git-tracked and clean: {p}")
    authority = {
        "sensitivity_protocol_sha256": sha256(protocol_path),
        "phase5_protocol_sha256": sha256(DEFAULT_PHASE5_PROTOCOL),
        "pair_manifest_sha256": sha256(DEFAULT_PAIR_MANIFEST),
        "feature_manifest_sha256": sha256(DEFAULT_FEATURE_MANIFEST),
        "primary_pair_results_sha256": sha256(DEFAULT_PRIMARY_RESULTS),
        "original_selected_pair_rows": 56,
        "original_robust_pair_rows": len(original_robust),
    }
    return original_robust, authority


def _support_check(bank, pair: dict, roles: np.ndarray, phase5_cfg: dict) -> tuple[bool, str, dict[str, int]]:
    ds = phase5_cfg["tier1"]["datasets"][bank.dataset_id]
    min_fit = int(ds["min_fit_per_class"])
    min_eval = int(ds["min_eval_per_class"])
    a, b = int(pair["domain_a"]), int(pair["domain_b"])
    classes = [int(c) for c in pair["shared_classes"]]
    counts: dict[str, int] = {}
    failures: list[str] = []
    for dom_label, dom in [("a", a), ("b", b)]:
        for c in classes:
            fit_n = int(np.sum((bank.domain_id == dom) & (roles == 0) & (bank.y == c)))
            eval_n = int(np.sum((bank.domain_id == dom) & (roles == 1) & (bank.y == c)))
            counts[f"{dom_label}_class_{c}_fit"] = fit_n
            counts[f"{dom_label}_class_{c}_eval"] = eval_n
            if fit_n < min_fit:
                failures.append(f"{dom_label}:class={c}:fit={fit_n}<{min_fit}")
            if eval_n < min_eval:
                failures.append(f"{dom_label}:class={c}:eval={eval_n}<{min_eval}")
    if failures:
        return False, ";".join(failures), counts
    return True, "", counts


def _eval_pair(
    bank,
    Xp: np.ndarray,
    pair: dict,
    roles: np.ndarray,
    taus: list[float],
    alpha0: float,
    modules: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    remap_classes = modules["remap_classes"]
    domain_risk_curve = modules["domain_risk_curve"]
    pair_decision_metrics = modules["pair_decision_metrics"]

    classes = [int(c) for c in pair["shared_classes"]]
    a, b = int(pair["domain_a"]), int(pair["domain_b"])

    af = (bank.domain_id == a) & (roles == 0) & np.isin(bank.y, classes)
    ae = (bank.domain_id == a) & (roles == 1) & np.isin(bank.y, classes)
    bf = (bank.domain_id == b) & (roles == 0) & np.isin(bank.y, classes)
    be = (bank.domain_id == b) & (roles == 1) & np.isin(bank.y, classes)

    Xaf, yaf, _ = remap_classes(Xp[af], bank.y[af], classes)
    Xae, yae, _ = remap_classes(Xp[ae], bank.y[ae], classes)
    Xbf, ybf, _ = remap_classes(Xp[bf], bank.y[bf], classes)
    Xbe, ybe, _ = remap_classes(Xp[be], bank.y[be], classes)

    ca, _ = domain_risk_curve(Xaf, yaf, Xae, yae, taus, float(alpha0))
    cb, _ = domain_risk_curve(Xbf, ybf, Xbe, ybe, taus, float(alpha0))
    metrics = pair_decision_metrics(ca, cb, taus)

    curves = []
    for task, curve in [("domain_a", ca), ("domain_b", cb)]:
        for tau, bacc in zip(taus, curve):
            curves.append({
                "task": task,
                "tau": float(tau),
                "balanced_accuracy": float(bacc),
            })
    return metrics, curves


def _scenario_map(cfg: dict) -> dict[str, Scenario]:
    return {s.scenario_id: s for s in build_scenarios(cfg)}


def _eval_condition(
    cond: dict,
    bank,
    cfg: dict,
    phase5_cfg: dict,
    original_robust: set[str],
    modules: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    project_features = modules["project_features"]
    deterministic_group_roles = modules["deterministic_group_roles"]
    no_group_leakage = modules["no_group_leakage"]

    taus = dense_tau_grid(cfg)
    scenarios = build_scenarios(cfg)
    primary_proj_seed = int(cfg["projection_seeds"][0])
    primary_split_seed = int(cfg["group_split_seeds"][0])
    fit_fraction = float(phase5_cfg["fixed"]["group_fit_fraction"])
    dim = int(cond["projection_dim"])

    # Projection cache is condition-local: at most three arrays, released after this condition.
    projection_cache: dict[int, np.ndarray] = {}
    roles_cache: dict[int, np.ndarray] = {}

    def get_projection(seed: int) -> np.ndarray:
        if seed not in projection_cache:
            projection_cache[seed] = project_features(
                bank.X, dim, bank.encoder_id, int(seed)
            )
        return projection_cache[seed]

    def get_roles(seed: int) -> np.ndarray:
        if seed not in roles_cache:
            roles_cache[seed] = deterministic_group_roles(
                bank.dataset_id, bank.domain_id, bank.group_id,
                int(seed), fit_fraction
            )
            if not no_group_leakage(bank.domain_id, bank.group_id, roles_cache[seed]):
                raise RuntimeError(f"Group leakage detected for {cond['condition_id']} split_seed={seed}")
        return roles_cache[seed]

    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []

    for scenario in scenarios:
        Xp = get_projection(scenario.projection_seed)
        roles = get_roles(scenario.split_seed)
        for pair in cond["selected_pairs"]:
            key = _pair_key(cond["condition_id"], pair["pair_id"])
            supported, reason, counts = _support_check(bank, pair, roles, phase5_cfg)
            base = {
                "scenario_id": scenario.scenario_id,
                "gate_family": scenario.gate_family,
                "condition_id": cond["condition_id"],
                "dataset_id": cond["dataset_id"],
                "bank_id": cond["bank_id"],
                "encoder_id": cond["encoder_id"],
                "projection_dim": dim,
                "pair_id": pair["pair_id"],
                "domain_a": int(pair["domain_a"]),
                "domain_b": int(pair["domain_b"]),
                "n_shared_classes": int(pair["n_shared_classes"]),
                "alpha0": float(scenario.alpha0),
                "projection_seed": int(scenario.projection_seed),
                "group_split_seed": int(scenario.split_seed),
                "primary_robust": key in original_robust,
                "supported": bool(supported),
                "unsupported_reason": reason,
            }
            if not supported:
                rows.append({
                    **base,
                    "optimal_sets_disjoint": False,
                    "deterministic_pair_minimax_regret": np.nan,
                    "randomized_pair_minimax_regret": np.nan,
                    "curve_linf_shift": np.nan,
                    "best_bacc_original": np.nan,
                    "best_bacc_counterfactual": np.nan,
                    "robust_under_sensitivity": False,
                    "retained_primary_robust": False,
                })
                continue

            metrics, pair_curves = _eval_pair(
                bank, Xp, pair, roles, taus, scenario.alpha0, modules
            )
            robust = bool(
                float(metrics["deterministic_pair_minimax_regret"])
                >= float(cfg["robust_pair"]["deterministic_regret_min"])
                and bool(metrics["optimal_sets_disjoint"])
            )
            retained = bool((key in original_robust) and robust)
            rows.append({
                **base,
                **metrics,
                "robust_under_sensitivity": robust,
                "retained_primary_robust": retained,
            })
            for cr in pair_curves:
                curves.append({
                    **{k: base[k] for k in [
                        "scenario_id", "gate_family", "condition_id", "dataset_id",
                        "bank_id", "encoder_id", "projection_dim", "pair_id",
                        "alpha0", "projection_seed", "group_split_seed"
                    ]},
                    **cr,
                })
    return rows, curves


def _median_int(values: list[int]) -> float:
    return float(np.median(np.asarray(values, dtype=float))) if values else float("nan")


def summarize_gates(df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    expected_primary = int(cfg["primary_robust_pair_count"])
    primary = df[df["scenario_id"] == "C1_dense_primary"]
    c1_retained = int(primary["retained_primary_robust"].sum())
    c1_threshold = int(cfg["gates"]["C1_dense_grid"]["retained_primary_robust_min"])
    c1_pass = c1_retained >= c1_threshold

    alpha_counts: dict[str, int] = {}
    for a in [float(x) for x in cfg["alpha0_values"]]:
        if abs(a - 0.10) <= 1e-12:
            z = primary
        else:
            sid = f"C2_alpha_{str(a).replace('.', 'p')}"
            z = df[df["scenario_id"] == sid]
        alpha_counts[str(a)] = int(z["retained_primary_robust"].sum())
    c2_min = int(cfg["gates"]["C2_regularization"]["retained_primary_robust_min_per_setting"])
    c2_npass = sum(v >= c2_min for v in alpha_counts.values())
    c2_pass = c2_npass >= int(cfg["gates"]["C2_regularization"]["min_settings_passing"])

    proj_counts: dict[str, int] = {}
    for seed in [int(x) for x in cfg["projection_seeds"]]:
        if seed == int(cfg["projection_seeds"][0]):
            z = primary
        else:
            z = df[df["scenario_id"] == f"C3_projection_{seed}"]
        proj_counts[str(seed)] = int(z["retained_primary_robust"].sum())
    c3_median = _median_int(list(proj_counts.values()))
    c3_pass = c3_median >= float(cfg["gates"]["C3_projection"]["median_retained_primary_robust_min"])

    split_counts: dict[str, int] = {}
    split_supported_primary: dict[str, int] = {}
    for seed in [int(x) for x in cfg["group_split_seeds"]]:
        if seed == int(cfg["group_split_seeds"][0]):
            z = primary
        else:
            z = df[df["scenario_id"] == f"C4_split_{seed}"]
        split_counts[str(seed)] = int(z["retained_primary_robust"].sum())
        split_supported_primary[str(seed)] = int((z["primary_robust"] & z["supported"]).sum())
    c4_median = _median_int(list(split_counts.values()))
    c4_pass = c4_median >= float(cfg["gates"]["C4_group_split"]["median_retained_primary_robust_min"])

    overall = bool(c1_pass and c2_pass and c3_pass and c4_pass)
    return {
        "authority": {
            "expected_primary_robust_pairs": expected_primary,
            "evaluated_rows": int(len(df)),
            "scenario_count": int(df["scenario_id"].nunique()),
        },
        "C1_dense_grid": {
            "pass": c1_pass,
            "retained_primary_robust": c1_retained,
            "required": c1_threshold,
        },
        "C2_regularization": {
            "pass": c2_pass,
            "retained_by_alpha0": alpha_counts,
            "settings_passing": c2_npass,
            "required_settings_passing": int(cfg["gates"]["C2_regularization"]["min_settings_passing"]),
            "required_retained_per_setting": c2_min,
        },
        "C3_projection": {
            "pass": c3_pass,
            "retained_by_seed": proj_counts,
            "median_retained": c3_median,
            "required_median": float(cfg["gates"]["C3_projection"]["median_retained_primary_robust_min"]),
        },
        "C4_group_split": {
            "pass": c4_pass,
            "retained_by_seed": split_counts,
            "supported_primary_robust_by_seed": split_supported_primary,
            "median_retained": c4_median,
            "required_median": float(cfg["gates"]["C4_group_split"]["median_retained_primary_robust_min"]),
        },
        "overall_pass": overall,
        "decision": "PHASE5_NATURE_SENSITIVITY_PASS" if overall else "PHASE5_NATURE_SENSITIVITY_FAIL",
    }


def effect_size_summary(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    primary = df[(df["scenario_id"] == "C1_dense_primary") & df["supported"]]
    robust = primary[primary["robust_under_sensitivity"]]
    for name, z in [("primary_dense_all_supported", primary), ("primary_dense_robust", robust)]:
        block: dict[str, Any] = {"n": int(len(z))}
        for col in [
            "deterministic_pair_minimax_regret",
            "randomized_pair_minimax_regret",
            "curve_linf_shift",
        ]:
            vals = pd.to_numeric(z[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals):
                block[col] = {
                    "min": float(np.min(vals)),
                    "q1": float(np.quantile(vals, 0.25)),
                    "median": float(np.median(vals)),
                    "q3": float(np.quantile(vals, 0.75)),
                    "max": float(np.max(vals)),
                }
        out[name] = block
    return out


def _formal(
    cfg: dict,
    phase5_cfg: dict,
    pair_manifest: dict,
    feature_manifest: dict,
    primary_results: pd.DataFrame,
    protocol_path: Path,
    out: Path,
    resume: bool,
) -> None:
    modules = _project_modules()
    original_robust, authority = _validate_preflight(
        cfg, phase5_cfg, pair_manifest, feature_manifest, primary_results,
        formal_git_guard=True, protocol_path=protocol_path,
    )

    entries = _feature_entry_map(feature_manifest)
    conditions = {str(c["condition_id"]): c for c in pair_manifest["conditions"]}
    by_bank: dict[str, list[dict]] = {}
    for cond in conditions.values():
        by_bank.setdefault(str(cond["bank_id"]), []).append(cond)

    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = sha256(protocol_path)

    for bank_id in sorted(by_bank):
        entry = entries[bank_id]
        feature_path = ROOT / entry["path"]
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        observed_hash = sha256(feature_path)
        if observed_hash != entry["sha256"]:
            raise RuntimeError(f"Feature bank hash mismatch: {bank_id}")
        bank = modules["load_tier1_feature_bank"](feature_path, entry)

        for cond in sorted(by_bank[bank_id], key=lambda c: c["condition_id"]):
            shard = shard_dir / f"{cond['condition_id']}.json"
            if resume and shard.exists():
                z = _load_json(shard)
                if z.get("sensitivity_protocol_sha256") != cfg_hash:
                    raise RuntimeError(f"Resume shard protocol mismatch: {shard}")
                rows.extend(z["rows"])
                curves.extend(z["curves"])
                print(f"[resume] {cond['condition_id']}")
                continue

            r, c = _eval_condition(
                cond, bank, cfg, phase5_cfg, original_robust, modules
            )
            payload = {
                "sensitivity_protocol_sha256": cfg_hash,
                "condition_id": cond["condition_id"],
                "rows": r,
                "curves": c,
            }
            _atomic_json(shard, payload)
            rows.extend(r)
            curves.extend(c)
            print(f"[done] {cond['condition_id']} rows={len(r)} curves={len(c)}")

        del bank

    df = pd.DataFrame(rows)
    cv = pd.DataFrame(curves)
    expected_scenarios = len(build_scenarios(cfg))
    expected_rows = 56 * expected_scenarios
    if len(df) != expected_rows:
        raise RuntimeError(f"Incomplete sensitivity table: expected {expected_rows}, got {len(df)}")
    if df["scenario_id"].nunique() != expected_scenarios:
        raise RuntimeError("Scenario count mismatch")

    gates = summarize_gates(df, cfg)
    effects = effect_size_summary(df)

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sensitivity_pair_results.csv", index=False)
    cv.to_csv(out / "sensitivity_risk_curves.csv", index=False)
    _atomic_json(out / "gate_summary.json", gates)
    _atomic_json(out / "effect_size_summary.json", effects)

    summary = {
        "mode": "formal",
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "authority": authority,
        "n_scenarios": expected_scenarios,
        "n_pair_scenario_rows": int(len(df)),
        "n_risk_curve_rows": int(len(cv)),
        "dense_tau_count": len(dense_tau_grid(cfg)),
        "gate_decision": gates["decision"],
    }
    _atomic_json(out / "run_summary.json", summary)
    _atomic_json(out / "environment_manifest.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    })
    (out / "protocol_snapshot.yaml").write_text(
        protocol_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    report = ROOT / "reports" / "phase5" / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase5_nature_sensitivity_formal_report.md"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Phase 5 Nature Sensitivity Formal Report\n\n"
        f"- Git HEAD: `{summary['git_head']}`\n"
        f"- Pair-scenario rows: {len(df)}\n"
        f"- Risk-curve rows: {len(cv)}\n"
        f"- Decision: **{gates['decision']}**\n\n"
        "## Gate summary\n\n```json\n"
        + json.dumps(gates, indent=2)
        + "\n```\n\n## Effect-size summary\n\n```json\n"
        + json.dumps(effects, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(json.dumps(gates, indent=2))
    print(f"REPORT={report}")


def _plan(cfg: dict) -> None:
    scenarios = build_scenarios(cfg)
    print("Dense tau count:", len(dense_tau_grid(cfg)))
    print("Scenarios:", len(scenarios))
    for s in scenarios:
        print(
            f"{s.scenario_id:28s} gate={s.gate_family} "
            f"alpha0={s.alpha0} projection_seed={s.projection_seed} split_seed={s.split_seed}"
        )
    print("Expected pair-scenario rows:", 56 * len(scenarios))
    print("Expected maximum risk-curve rows:",
          56 * len(scenarios) * 2 * len(dense_tau_grid(cfg)))


def _smoke(cfg: dict) -> None:
    scenarios = build_scenarios(cfg)
    assert len(scenarios) == 7
    assert len(dense_tau_grid(cfg)) == 51
    rows = []
    # Synthetic gate table with exactly 11 primary robust keys retained under known counts.
    for s in scenarios:
        retained = {
            "C1_dense_primary": 10,
            "C2_alpha_0p03": 8,
            "C2_alpha_0p3": 6,
            "C3_projection_20260921": 7,
            "C3_projection_20260922": 8,
            "C4_split_20260921": 7,
            "C4_split_20260922": 6,
        }[s.scenario_id]
        for i in range(56):
            primary = i < 11
            rows.append({
                "scenario_id": s.scenario_id,
                "primary_robust": primary,
                "supported": True,
                "retained_primary_robust": primary and i < retained,
            })
    gates = summarize_gates(pd.DataFrame(rows), cfg)
    assert gates["C1_dense_grid"]["pass"] is True
    assert gates["C2_regularization"]["pass"] is True
    assert gates["C3_projection"]["pass"] is True
    assert gates["C4_group_split"]["pass"] is True
    assert gates["overall_pass"] is True
    print(json.dumps({"smoke": "PASS", "gates": gates}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["plan", "smoke", "formal"])
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    protocol_path = Path(args.protocol)
    cfg = _load_yaml(protocol_path)

    if args.mode == "plan":
        _plan(cfg)
        return
    if args.mode == "smoke":
        _smoke(cfg)
        return

    phase5_cfg = _load_yaml(DEFAULT_PHASE5_PROTOCOL)
    pair_manifest = _load_json(DEFAULT_PAIR_MANIFEST)
    feature_manifest = _load_json(DEFAULT_FEATURE_MANIFEST)
    primary_results = pd.read_csv(DEFAULT_PRIMARY_RESULTS)
    _formal(
        cfg, phase5_cfg, pair_manifest, feature_manifest, primary_results,
        protocol_path, Path(args.out_dir), bool(args.resume)
    )


if __name__ == "__main__":
    main()
