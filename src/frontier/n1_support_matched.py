from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class N1Columns:
    record_id: str = "record_id"
    institution: str = "institution"
    age: str = "age"
    age_band: str = "age_band"
    y_true: str = "y_true"
    prob_seed42: str = "prob_seed42"
    prob_seed52: str = "prob_seed52"
    prob_seed62: str = "prob_seed62"
    prob_mean: str = "prob_mean"
    y_pred: str = "y_pred"


CANONICAL_COLUMNS = N1Columns()
AGE_ORDER = ["<40", "40-59", "60-74", "75+"]


def balanced_accuracy_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("Balanced accuracy is undefined when a class is absent")
    tpr = np.mean(y_pred[pos] == 1)
    tnr = np.mean(y_pred[neg] == 0)
    return float(0.5 * (tpr + tnr))


def derive_age_band(age: pd.Series | np.ndarray) -> np.ndarray:
    a = pd.to_numeric(pd.Series(age), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("age must be finite numeric for all authoritative rows")
    out = np.empty(len(a), dtype=object)
    out[a < 40] = "<40"
    out[(a >= 40) & (a < 60)] = "40-59"
    out[(a >= 60) & (a < 75)] = "60-74"
    out[a >= 75] = "75+"
    return out.astype(str)


def _binary_series(s: pd.Series, name: str) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isin(x, [0.0, 1.0]).all():
        bad = pd.Series(x)[~pd.Series(x).isin([0.0, 1.0])].head(10).tolist()
        raise ValueError(f"{name} must be binary 0/1; examples={bad}")
    return x.astype(np.int8)


def prepare_authoritative_rows(df: pd.DataFrame, protocol: dict, columns: N1Columns = CANONICAL_COLUMNS) -> tuple[pd.DataFrame, dict]:
    required = list(protocol["authoritative_input"]["required_columns"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing authoritative N1 columns: {missing}")
    if df.empty:
        raise ValueError("N1 authoritative input is empty")
    if df[required].isna().any().any():
        bad_cols = df[required].columns[df[required].isna().any()].tolist()
        raise ValueError(f"Missing values in authoritative columns: {bad_cols}")
    if protocol["authoritative_input"].get("record_id_unique", True) and not df[columns.record_id].is_unique:
        raise ValueError("record_id must be unique")

    institutions = df[columns.institution].astype(str).to_numpy()
    expected_inst = set(map(str, protocol["context_construction"]["institutions"]))
    actual_inst = set(institutions)
    if actual_inst != expected_inst:
        raise ValueError(f"Institution set mismatch expected={sorted(expected_inst)} actual={sorted(actual_inst)}")

    derived_band = derive_age_band(df[columns.age])
    supplied_band = df[columns.age_band].astype(str).to_numpy()
    band_ok = supplied_band == derived_band
    if not band_ok.all():
        idx = np.flatnonzero(~band_ok)[:10]
        examples = [(str(df.iloc[i][columns.record_id]), float(df.iloc[i][columns.age]), supplied_band[i], derived_band[i]) for i in idx]
        raise ValueError(f"age_band disagrees with frozen age boundaries; examples={examples}")

    yt = _binary_series(df[columns.y_true], columns.y_true)
    prob_cols = protocol["prediction_rule"]["seed_columns"]
    probs = df[prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Frozen seed probabilities must all be finite and in [0,1]")
    prob_mean = probs.mean(axis=1)
    tol = float(protocol["prediction_rule"].get("probability_tolerance", 1e-12))
    supplied_mean = pd.to_numeric(df[columns.prob_mean], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(supplied_mean).all() or not np.allclose(prob_mean, supplied_mean, rtol=0.0, atol=tol):
        diff = float(np.nanmax(np.abs(prob_mean - supplied_mean)))
        raise ValueError(f"prob_mean does not equal arithmetic mean of frozen seed probabilities; max_abs_diff={diff}")

    threshold = float(protocol["prediction_rule"]["threshold"])
    derived_pred = (prob_mean >= threshold).astype(np.int8)
    supplied_pred = _binary_series(df[columns.y_pred], columns.y_pred)
    if not np.array_equal(derived_pred, supplied_pred):
        n = int(np.sum(derived_pred != supplied_pred))
        raise ValueError(f"y_pred violates frozen probability-mean threshold rule; mismatched_rows={n}")

    out = pd.DataFrame({
        "sample_id": df[columns.record_id].astype(str).to_numpy(),
        "institution_id": institutions,
        "age": pd.to_numeric(df[columns.age], errors="raise").to_numpy(dtype=float),
        "age_band": derived_band,
        "context_id": np.char.add(np.char.add(institutions.astype(str), "|"), derived_band.astype(str)),
        "y_true": yt,
        "y_pred": derived_pred,
        "prob_mean": prob_mean,
    })
    audit = {
        "n_rows": int(len(out)),
        "record_id_unique": bool(out["sample_id"].is_unique),
        "institution_set": sorted(actual_inst),
        "age_band_recomputed_exact": True,
        "prob_mean_recomputed_exact_within_tolerance": True,
        "y_pred_recomputed_exact": True,
        "prediction_threshold": threshold,
    }
    return out, audit


def context_support_table(rows: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    institutions = list(map(str, protocol["context_construction"]["institutions"]))
    age_bands = list(map(str, protocol["context_construction"]["age_bands"]))
    order = pd.MultiIndex.from_product([institutions, age_bands], names=["institution_id", "age_band"])

    grouped = []
    for (inst, band), g in rows.groupby(["institution_id", "age_band"], sort=False):
        n_pos = int((g["y_true"] == 1).sum())
        n_neg = int((g["y_true"] == 0).sum())
        grouped.append({
            "institution_id": str(inst),
            "age_band": str(band),
            "context_id": f"{inst}|{band}",
            "N": int(len(g)),
            "n_positive": n_pos,
            "n_negative": n_neg,
        })
    tab = pd.DataFrame(grouped).set_index(["institution_id", "age_band"]).reindex(order)
    if tab[["N", "n_positive", "n_negative"]].isna().any().any():
        missing = tab[tab["N"].isna()].index.tolist()
        raise ValueError(f"Missing frozen institution-by-age cells: {missing}")
    tab = tab.reset_index()
    tab["context_id"] = tab["institution_id"].astype(str) + "|" + tab["age_band"].astype(str)
    for c in ["N", "n_positive", "n_negative"]:
        tab[c] = tab[c].astype(int)
    elig = protocol["context_construction"]["eligibility"]
    tab["eligible"] = (
        (tab["N"] >= int(elig["min_N"]))
        & (tab["n_positive"] >= int(elig["min_positive"]))
        & (tab["n_negative"] >= int(elig["min_negative"]))
    )
    return tab


def verify_frozen_support(tab: pd.DataFrame, protocol: dict) -> dict:
    expected = protocol["context_construction"]["expected_support"]
    exp = pd.DataFrame(expected).rename(columns={"institution": "institution_id"})
    cols = ["institution_id", "age_band", "N", "n_positive", "n_negative", "eligible"]
    a = tab[cols].copy().sort_values(["institution_id", "age_band"]).reset_index(drop=True)
    b = exp[cols].copy().sort_values(["institution_id", "age_band"]).reset_index(drop=True)
    for c in ["N", "n_positive", "n_negative"]:
        b[c] = b[c].astype(int)
    b["eligible"] = b["eligible"].astype(bool)
    if not a.equals(b):
        merged = a.merge(b, on=["institution_id", "age_band"], how="outer", suffixes=("_actual", "_expected"), indicator=True)
        raise ValueError("Frozen 16-cell support fingerprint mismatch:\n" + merged.to_string(index=False))
    n_eligible = int(tab["eligible"].sum())
    if len(tab) != int(protocol["context_construction"]["expected_all_contexts"]):
        raise ValueError("Unexpected number of all institution-by-age contexts")
    if n_eligible != int(protocol["context_construction"]["expected_eligible_contexts"]):
        raise ValueError(f"Expected 13 eligible contexts, found {n_eligible}")
    return {
        "all_context_support_fingerprint_match": True,
        "n_all_contexts": int(len(tab)),
        "n_eligible_contexts": n_eligible,
        "eligible_context_ids": tab.loc[tab["eligible"], "context_id"].tolist(),
    }


def eligible_analysis_rows(rows: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    keep = set(support.loc[support["eligible"], "context_id"].astype(str))
    out = rows[rows["context_id"].isin(keep)].copy()
    if out.empty:
        raise ValueError("No eligible N1 rows")
    return out


def natural_context_table(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (inst, band, ctx), g in rows.groupby(["institution_id", "age_band", "context_id"], sort=True):
        yt = g["y_true"].to_numpy(dtype=np.int8)
        yp = g["y_pred"].to_numpy(dtype=np.int8)
        bacc = balanced_accuracy_binary(yt, yp)
        records.append({
            "institution_id": str(inst),
            "age_band": str(band),
            "context_id": str(ctx),
            "N": int(len(g)),
            "n_positive": int((yt == 1).sum()),
            "n_negative": int((yt == 0).sum()),
            "balanced_accuracy": bacc,
            "risk": float(1.0 - bacc),
        })
    return pd.DataFrame(records)


def t_het_from_context_risks(contexts: pd.DataFrame, ddof: int = 0) -> float:
    vals = []
    for _, g in contexts.groupby("institution_id", sort=True):
        r = g["risk"].to_numpy(dtype=float)
        if len(r) < 2:
            raise ValueError("T_HET requires >=2 eligible contexts per institution")
        vals.append(float(np.var(r, ddof=ddof)))
    if not vals:
        raise ValueError("No institutions available for T_HET")
    return float(np.mean(vals))


def _sequential_binary_partition_correct_counts(*, total_correct: int, total_incorrect: int, slot_sizes: np.ndarray, replicates: int, rng: np.random.Generator) -> np.ndarray:
    slot_sizes = np.asarray(slot_sizes, dtype=np.int64)
    if slot_sizes.sum() != total_correct + total_incorrect:
        raise ValueError("Slot sizes must exhaust the class-specific finite population")
    if (slot_sizes < 0).any():
        raise ValueError("Negative slot size")
    out = np.zeros((replicates, len(slot_sizes)), dtype=np.int64)
    good = np.full(replicates, int(total_correct), dtype=np.int64)
    bad = np.full(replicates, int(total_incorrect), dtype=np.int64)
    for i, n in enumerate(slot_sizes[:-1]):
        draw = rng.hypergeometric(good, bad, int(n))
        out[:, i] = draw
        good -= draw
        bad -= int(n) - draw
    out[:, -1] = good
    return out


def _sequential_multicategory_partition(*, category_counts: Iterable[int], slot_sizes: np.ndarray, replicates: int, rng: np.random.Generator) -> np.ndarray:
    base = np.asarray(list(category_counts), dtype=np.int64)
    slot_sizes = np.asarray(slot_sizes, dtype=np.int64)
    if base.sum() != slot_sizes.sum():
        raise ValueError("Slot sizes must exhaust finite population")
    if (base < 0).any() or (slot_sizes < 0).any():
        raise ValueError("Negative counts are invalid")
    k = len(base)
    s = len(slot_sizes)
    out = np.zeros((replicates, s, k), dtype=np.int64)
    remaining = np.repeat(base[None, :], replicates, axis=0)
    for si, n in enumerate(slot_sizes[:-1]):
        slots_left = np.full(replicates, int(n), dtype=np.int64)
        for ci in range(k - 1):
            good = remaining[:, ci]
            bad = remaining[:, ci + 1 :].sum(axis=1)
            draw = rng.hypergeometric(good, bad, slots_left)
            out[:, si, ci] = draw
            remaining[:, ci] -= draw
            slots_left -= draw
        out[:, si, -1] = slots_left
        remaining[:, -1] -= slots_left
    out[:, -1, :] = remaining
    return out


def simulate_arm_c_exact_support(rows: pd.DataFrame, contexts: pd.DataFrame, *, replicates: int, seed: int, ddof: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    inst_values: list[np.ndarray] = []
    for inst, slots in contexts.groupby("institution_id", sort=True):
        g = rows[rows["institution_id"].astype(str) == str(inst)]
        yt = g["y_true"].to_numpy(dtype=np.int8)
        yp = g["y_pred"].to_numpy(dtype=np.int8)
        pos = yt == 1
        neg = yt == 0
        pos_correct = int((yp[pos] == 1).sum())
        neg_correct = int((yp[neg] == 0).sum())
        n_pos = slots["n_positive"].to_numpy(dtype=np.int64)
        n_neg = slots["n_negative"].to_numpy(dtype=np.int64)
        if n_pos.sum() != int(pos.sum()) or n_neg.sum() != int(neg.sum()):
            raise ValueError(f"Eligible context slots do not exhaust eligible institution support: {inst}")
        tp = _sequential_binary_partition_correct_counts(
            total_correct=pos_correct,
            total_incorrect=int(pos.sum()) - pos_correct,
            slot_sizes=n_pos,
            replicates=replicates,
            rng=rng,
        )
        tn = _sequential_binary_partition_correct_counts(
            total_correct=neg_correct,
            total_incorrect=int(neg.sum()) - neg_correct,
            slot_sizes=n_neg,
            replicates=replicates,
            rng=rng,
        )
        bacc = 0.5 * (tp / n_pos[None, :] + tn / n_neg[None, :])
        risk = 1.0 - bacc
        inst_values.append(np.var(risk, axis=1, ddof=ddof))
    return np.mean(np.vstack(inst_values), axis=0)


def simulate_arm_b_n_matched(rows: pd.DataFrame, contexts: pd.DataFrame, *, replicates: int, seed: int, ddof: int = 0) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    yt = rows["y_true"].to_numpy(dtype=np.int8)
    yp = rows["y_pred"].to_numpy(dtype=np.int8)
    counts = [
        int(((yt == 1) & (yp == 1)).sum()),
        int(((yt == 1) & (yp == 0)).sum()),
        int(((yt == 0) & (yp == 0)).sum()),
        int(((yt == 0) & (yp == 1)).sum()),
    ]
    slot_n = contexts["N"].to_numpy(dtype=np.int64)
    alloc = _sequential_multicategory_partition(
        category_counts=counts,
        slot_sizes=slot_n,
        replicates=replicates,
        rng=rng,
    )
    tp, fn, tn, fp = [alloc[:, :, i] for i in range(4)]
    pos_n = tp + fn
    neg_n = tn + fp
    valid = ((pos_n > 0) & (neg_n > 0)).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        risk = 1.0 - 0.5 * (tp / pos_n + tn / neg_n)
    inst_vars = []
    labels = contexts["institution_id"].astype(str).to_numpy()
    for inst in sorted(set(labels)):
        idx = np.flatnonzero(labels == inst)
        inst_vars.append(np.var(risk[:, idx], axis=1, ddof=ddof))
    t = np.mean(np.vstack(inst_vars), axis=0)
    t[~valid] = np.nan
    return t, float(np.mean(valid))


def monte_carlo_upper_p(null_values: np.ndarray, observed: float) -> float:
    z = np.asarray(null_values, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        raise ValueError("No finite null values")
    return float((1 + np.sum(z >= observed)) / (len(z) + 1))


def evaluate_n1_gate(*, natural_t_het: float, arm_c_null: np.ndarray, alpha: float = 0.05) -> dict:
    z = np.asarray(arm_c_null, dtype=float)
    if not np.isfinite(z).all():
        raise ValueError("Arm-C null contains non-finite values")
    p = monte_carlo_upper_p(z, natural_t_het)
    med = float(np.median(z))
    pass_p = bool(p <= alpha)
    pass_med = bool(natural_t_het > med)
    passed = bool(pass_p and pass_med)
    return {
        "primary_statistic": "T_HET",
        "T_HET_natural": float(natural_t_het),
        "arm_c_replicates": int(len(z)),
        "arm_c_null_median": med,
        "arm_c_null_mean": float(np.mean(z)),
        "arm_c_null_q025": float(np.quantile(z, 0.025)),
        "arm_c_null_q975": float(np.quantile(z, 0.975)),
        "p_HET": p,
        "alpha": float(alpha),
        "p_condition_pass": pass_p,
        "median_condition_pass": pass_med,
        "pass": passed,
        "decision": "N1_PASS_STRUCTURE_BEYOND_SUPPORT" if passed else "N1_SUPPORT_DOMINATED",
    }
