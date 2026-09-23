from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

AGE_ORDER = ["<40", "40-59", "60-74", "75+"]


@dataclass(frozen=True)
class N3Columns:
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


COL = N3Columns()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _institution_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not token:
        raise ValueError(f"Empty institution identifier after normalization: {value!r}")
    return token


def canonicalize_institutions(values: pd.Series, expected: Iterable[str]) -> tuple[np.ndarray, dict[str, str]]:
    expected = [str(x) for x in expected]
    expected_by_token: dict[str, str] = {}
    for name in expected:
        token = _institution_token(name)
        if token in expected_by_token and expected_by_token[token] != name:
            raise ValueError(f"Frozen institution labels collide under representation normalization: {name!r}")
        expected_by_token[token] = name

    raw = values.astype(str)
    raw_unique = sorted(raw.unique().tolist())
    raw_to_canonical: dict[str, str] = {}
    reverse: dict[str, list[str]] = {name: [] for name in expected}
    unknown: list[str] = []
    for name in raw_unique:
        canonical = expected_by_token.get(_institution_token(name))
        if canonical is None:
            unknown.append(name)
        else:
            raw_to_canonical[name] = canonical
            reverse[canonical].append(name)
    if unknown:
        raise ValueError(f"Unknown institution identifiers after representation normalization: {unknown}")
    missing = [x for x, aliases in reverse.items() if not aliases]
    if missing:
        raise ValueError(f"Missing frozen institutions: {missing}")
    ambiguous = {x: aliases for x, aliases in reverse.items() if len(aliases) != 1}
    if ambiguous:
        raise ValueError(f"Multiple raw institution aliases map to one frozen institution: {ambiguous}")
    mapped = raw.map(raw_to_canonical).to_numpy(dtype=str)
    return mapped, raw_to_canonical


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


def _binary(s: pd.Series, name: str) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isin(x, [0.0, 1.0]).all():
        raise ValueError(f"{name} must be binary 0/1")
    return x.astype(np.int8)


def prepare_authoritative_rows(df: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, dict]:
    req = list(protocol["inputs"]["n1_authoritative_rows"]["required_columns"])
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing N1 authoritative columns: {missing}")
    if len(df) == 0:
        raise ValueError("N1 authoritative rows are empty")
    if df[req].isna().any().any():
        bad = df[req].columns[df[req].isna().any()].tolist()
        raise ValueError(f"Missing values in authoritative columns: {bad}")
    if not df[COL.record_id].is_unique:
        raise ValueError("record_id must be unique")

    institutions, inst_map = canonicalize_institutions(df[COL.institution], protocol["context"]["institutions"])
    derived_band = derive_age_band(df[COL.age])
    supplied_band = df[COL.age_band].astype(str).to_numpy()
    if not np.array_equal(derived_band, supplied_band):
        idx = np.flatnonzero(derived_band != supplied_band)[:10]
        ex = [(str(df.iloc[i][COL.record_id]), supplied_band[i], derived_band[i]) for i in idx]
        raise ValueError(f"age_band disagrees with frozen age boundaries; examples={ex}")

    y_true = _binary(df[COL.y_true], COL.y_true)
    seed_cols = protocol["prediction"]["seed_columns"]
    probs = df[seed_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Seed probabilities must be finite and in [0,1]")
    prob_mean = probs.mean(axis=1)
    supplied_mean = pd.to_numeric(df[COL.prob_mean], errors="coerce").to_numpy(dtype=float)
    tol = float(protocol["prediction"]["probability_tolerance"])
    if not np.isfinite(supplied_mean).all() or not np.allclose(prob_mean, supplied_mean, rtol=0.0, atol=tol):
        diff = float(np.nanmax(np.abs(prob_mean - supplied_mean)))
        raise ValueError(f"prob_mean mismatch; max_abs_diff={diff}")
    threshold = float(protocol["prediction"]["threshold"])
    y_pred = (prob_mean >= threshold).astype(np.int8)
    supplied_pred = _binary(df[COL.y_pred], COL.y_pred)
    if not np.array_equal(y_pred, supplied_pred):
        raise ValueError(f"y_pred violates frozen mean-probability threshold; mismatched_rows={int(np.sum(y_pred != supplied_pred))}")

    out = pd.DataFrame({
        "record_id": df[COL.record_id].astype(str).to_numpy(),
        "institution_id": institutions,
        "age": pd.to_numeric(df[COL.age], errors="raise").to_numpy(dtype=float),
        "age_band": derived_band,
        "context_id": np.char.add(np.char.add(institutions.astype(str), "|"), derived_band.astype(str)),
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_mean": prob_mean,
    })
    audit = {
        "n_rows": int(len(out)),
        "record_id_unique": bool(out.record_id.is_unique),
        "institution_set": sorted(set(out.institution_id)),
        "institution_raw_to_canonical": inst_map,
        "age_band_recomputed_exact": True,
        "prob_mean_recomputed_exact_within_tolerance": True,
        "y_pred_recomputed_exact": True,
        "prediction_threshold": threshold,
    }
    return out, audit


def _canonical_integer_string(v: object, name: str) -> str:
    if pd.isna(v):
        raise ValueError(f"Missing {name}")
    s = str(v).strip()
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", s):
        return str(int(float(s)))
    return s


def _path_token(v: object) -> str:
    s = str(v).strip().replace("\\", "/").lower()
    while s.endswith((".hea", ".mat", ".dat")):
        s = s.rsplit(".", 1)[0]
    s = re.sub(r"/+", "/", s)
    return s


def _record_aliases(v: object) -> set[str]:
    s = _path_token(v)
    out = {s}
    if "/" in s:
        out.add(s.rsplit("/", 1)[-1])
    expanded = set(out)
    for x in list(out):
        y = re.sub(r"_(lr|hr)$", "", x)
        expanded.add(y)
        m = re.search(r"(\d+)$", y)
        if m:
            expanded.add(str(int(m.group(1))))
            expanded.add(m.group(1).lstrip("0") or "0")
    return {x for x in expanded if x != ""}


def prepare_ptbxl_metadata(meta: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, dict]:
    req = list(protocol["inputs"]["ptbxl_metadata"]["required_columns"])
    missing = [c for c in req if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing PTB-XL metadata columns: {missing}")
    if meta[req].isna().any().any():
        bad = meta[req].columns[meta[req].isna().any()].tolist()
        raise ValueError(f"Missing values in PTB-XL identity columns: {bad}")
    d = meta.copy()
    d["ecg_id_canon"] = d["ecg_id"].map(lambda x: _canonical_integer_string(x, "ecg_id"))
    d["patient_id_canon"] = d["patient_id"].map(lambda x: _canonical_integer_string(x, "patient_id"))
    if not d["ecg_id_canon"].is_unique:
        raise ValueError("PTB-XL ecg_id must be unique")
    audit = {
        "rows": int(len(d)),
        "unique_ecg_id": int(d["ecg_id_canon"].nunique()),
        "unique_patient_id": int(d["patient_id_canon"].nunique()),
    }
    return d, audit


def build_ptb_alias_map(meta: pd.DataFrame) -> dict[str, str]:
    alias_to_ids: dict[str, set[str]] = {}
    for _, r in meta.iterrows():
        eid = str(r["ecg_id_canon"])
        aliases = {eid}
        if eid.isdigit():
            aliases.add(f"{int(eid):05d}")
        for col in ("filename_lr", "filename_hr"):
            if col in meta.columns and not pd.isna(r[col]):
                aliases.update(_record_aliases(r[col]))
        for a in aliases:
            alias_to_ids.setdefault(a, set()).add(eid)
    collisions = {k: sorted(v) for k, v in alias_to_ids.items() if len(v) > 1}
    if collisions:
        # Generic numeric aliases should still be unique for PTB-XL. Any collision is unsafe.
        ex = dict(list(collisions.items())[:10])
        raise ValueError(f"PTB-XL metadata alias collision: {ex}")
    return {k: next(iter(v)) for k, v in alias_to_ids.items()}


def bind_ptb_patients(rows: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ptb = rows[rows.institution_id == "PTB-XL"].copy()
    alias_map = build_ptb_alias_map(meta)
    meta_by_ecg = meta.set_index("ecg_id_canon", drop=False)
    resolved: list[str | None] = []
    methods: list[str] = []
    for rid in ptb.record_id.astype(str):
        ids = {alias_map[a] for a in _record_aliases(rid) if a in alias_map}
        if not ids:
            # Conservative final fallback: last numeric run in the record basename.
            token = _path_token(rid).rsplit("/", 1)[-1]
            token = re.sub(r"_(lr|hr)$", "", token)
            m = re.search(r"(\d+)$", token)
            if m:
                eid = str(int(m.group(1)))
                if eid in meta_by_ecg.index:
                    ids = {eid}
                    methods.append("numeric_terminal_fallback")
                else:
                    methods.append("unresolved")
            else:
                methods.append("unresolved")
        else:
            methods.append("metadata_alias")
        if len(ids) > 1:
            raise ValueError(f"Ambiguous PTB-XL binding for record_id={rid!r}: {sorted(ids)}")
        resolved.append(next(iter(ids)) if ids else None)
    ptb["ecg_id"] = resolved
    missing = int(ptb.ecg_id.isna().sum())
    if missing:
        ex = ptb.loc[ptb.ecg_id.isna(), "record_id"].head(20).tolist()
        raise ValueError(f"PTB-XL patient linkage incomplete: missing={missing}; examples={ex}")
    if ptb.ecg_id.duplicated().any():
        ex = ptb.loc[ptb.ecg_id.duplicated(keep=False), ["record_id", "ecg_id"]].head(20).to_dict(orient="records")
        raise ValueError(f"Multiple authoritative rows bind to the same PTB-XL ecg_id; examples={ex}")
    patient_map = meta_by_ecg["patient_id_canon"].to_dict()
    ptb["patient_id"] = ptb.ecg_id.map(patient_map)
    if ptb.patient_id.isna().any():
        raise ValueError("Resolved PTB-XL ecg_id lacks patient_id")
    audit = {
        "bound_rows": int(len(ptb)),
        "linked_rows": int(ptb.patient_id.notna().sum()),
        "linkage_fraction": float(ptb.patient_id.notna().mean()),
        "binding_method_counts": pd.Series(methods).value_counts().sort_index().to_dict(),
    }
    return ptb, audit


def identity_fingerprint(ptb: pd.DataFrame) -> dict:
    counts = ptb.groupby("patient_id", sort=False).size()
    repeated = counts[counts > 1]
    band_counts = ptb.groupby("patient_id", sort=False)["age_band"].nunique()
    return {
        "bound_ptbxl_rows": int(len(ptb)),
        "unique_patients": int(ptb.patient_id.nunique()),
        "repeated_patients": int(len(repeated)),
        "repeated_patient_records": int(repeated.sum()),
        "max_ecgs_per_patient": int(counts.max()),
        "cross_age_patients": int((band_counts > 1).sum()),
    }


def verify_identity_fingerprint(observed: dict, expected: dict) -> bool:
    keys = [
        "bound_ptbxl_rows", "unique_patients", "repeated_patients",
        "repeated_patient_records", "max_ecgs_per_patient", "cross_age_patients",
    ]
    mismatches = {k: {"expected": int(expected[k]), "actual": int(observed[k])} for k in keys if int(observed[k]) != int(expected[k])}
    if mismatches:
        raise ValueError(f"Frozen N3-1 identity fingerprint mismatch: {mismatches}")
    return True


def binding_digest(ptb: pd.DataFrame) -> str:
    d = ptb[["record_id", "ecg_id", "patient_id", "age_band"]].astype(str).sort_values("record_id")
    payload = "".join("\t".join(r) + "\n" for r in d.itertuples(index=False, name=None)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    pos = y_true == 1
    neg = y_true == 0
    if not pos.any() or not neg.any():
        raise ValueError("Balanced accuracy requires both classes")
    return float(0.5 * (np.mean(y_pred[pos] == 1) + np.mean(y_pred[neg] == 0)))


def wilson_interval(successes: int, n: int, z: float) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("Wilson interval requires n>0")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def classify_drop_interval(drop_lower: float, drop_upper: float, margin: float) -> str:
    if drop_lower > margin:
        return "DEFINITE_FAILURE"
    if drop_upper < margin:
        return "DEFINITE_NONFAILURE"
    return "AMBIGUOUS"


def wilson_status(rows: pd.DataFrame, source_bacc: float, margin: float, z: float) -> dict:
    yt = rows.y_true.to_numpy(dtype=np.int8)
    yp = rows.y_pred.to_numpy(dtype=np.int8)
    pos = yt == 1
    neg = yt == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Wilson BACC diagnostic requires both classes")
    tp = int(np.sum(yp[pos] == 1))
    tn = int(np.sum(yp[neg] == 0))
    sens_lo, sens_hi = wilson_interval(tp, n_pos, z)
    spec_lo, spec_hi = wilson_interval(tn, n_neg, z)
    bacc = 0.5 * (tp / n_pos + tn / n_neg)
    bacc_lo = 0.5 * (sens_lo + spec_lo)
    bacc_hi = 0.5 * (sens_hi + spec_hi)
    drop = source_bacc - bacc
    drop_lo = source_bacc - bacc_hi
    drop_hi = source_bacc - bacc_lo
    return {
        "N": int(len(rows)), "n_positive": n_pos, "n_negative": n_neg,
        "balanced_accuracy": float(bacc), "performance_drop_bacc": float(drop),
        "bacc_ci_lower": float(bacc_lo), "bacc_ci_upper": float(bacc_hi),
        "drop_ci_lower": float(drop_lo), "drop_ci_upper": float(drop_hi),
        "status": classify_drop_interval(drop_lo, drop_hi, margin),
    }


def cluster_bootstrap_status(
    rows: pd.DataFrame,
    *,
    source_bacc: float,
    margin: float,
    replicates: int,
    rng: np.random.Generator,
    qlo: float = 0.025,
    qhi: float = 0.975,
    batch_size: int = 256,
) -> dict:
    if "patient_id" not in rows.columns:
        raise ValueError("patient_id required for cluster bootstrap")
    grouped = []
    for _, g in rows.groupby("patient_id", sort=True):
        yt = g.y_true.to_numpy(dtype=np.int8)
        yp = g.y_pred.to_numpy(dtype=np.int8)
        grouped.append([
            int(np.sum((yt == 1) & (yp == 1))),
            int(np.sum((yt == 1) & (yp == 0))),
            int(np.sum((yt == 0) & (yp == 0))),
            int(np.sum((yt == 0) & (yp == 1))),
        ])
    c = np.asarray(grouped, dtype=np.int64)
    n_patients = int(len(c))
    if n_patients < 2:
        raise ValueError("Cluster bootstrap requires at least two patients")
    out = np.empty(replicates, dtype=float)
    at = 0
    while at < replicates:
        b = min(batch_size, replicates - at)
        idx = rng.integers(0, n_patients, size=(b, n_patients), endpoint=False)
        totals = c[idx].sum(axis=1)
        tp, fn, tn, fp = [totals[:, j].astype(float) for j in range(4)]
        pos_n = tp + fn
        neg_n = tn + fp
        if np.any(pos_n <= 0) or np.any(neg_n <= 0):
            raise RuntimeError("Cluster bootstrap produced a single-class replicate")
        out[at:at+b] = 0.5 * (tp / pos_n + tn / neg_n)
        at += b
    bacc_lo = float(np.quantile(out, qlo))
    bacc_hi = float(np.quantile(out, qhi))
    drop_lo = float(source_bacc - bacc_hi)
    drop_hi = float(source_bacc - bacc_lo)
    return {
        "n_patients": n_patients,
        "replicates": int(replicates),
        "bacc_bootstrap_mean": float(np.mean(out)),
        "bacc_ci_lower": bacc_lo,
        "bacc_ci_upper": bacc_hi,
        "drop_ci_lower": drop_lo,
        "drop_ci_upper": drop_hi,
        "status": classify_drop_interval(drop_lo, drop_hi, margin),
    }


def patient_selection_hash(patient_id: str, record_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{patient_id}|{record_id}".encode("utf-8")).hexdigest()


def select_one_ecg_per_patient(ptb: pd.DataFrame, seed: int, rule: str = "hash") -> pd.DataFrame:
    d = ptb.copy()
    if rule == "hash":
        d["_sel"] = [patient_selection_hash(str(p), str(r), seed) for p, r in zip(d.patient_id, d.record_id)]
        d = d.sort_values(["patient_id", "_sel", "record_id"], kind="mergesort")
    elif rule in {"min_ecg_id", "max_ecg_id"}:
        numeric = pd.to_numeric(d.ecg_id, errors="coerce")
        if numeric.isna().any():
            raise ValueError("ecg_id must be numeric for min/max deterministic sensitivity")
        d["_ecg_numeric"] = numeric.astype(np.int64)
        asc = rule == "min_ecg_id"
        d = d.sort_values(["patient_id", "_ecg_numeric", "record_id"], ascending=[True, asc, True], kind="mergesort")
    else:
        raise ValueError(f"Unknown one-ECG selection rule: {rule}")
    out = d.drop_duplicates("patient_id", keep="first").copy()
    return out.drop(columns=[c for c in ["_sel", "_ecg_numeric"] if c in out.columns])


def context_support(rows: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    insts = list(map(str, protocol["context"]["institutions"]))
    bands = list(map(str, protocol["context"]["age_bands"]))
    order = pd.MultiIndex.from_product([insts, bands], names=["institution_id", "age_band"])
    records = []
    for (inst, band), g in rows.groupby(["institution_id", "age_band"], sort=False):
        records.append({
            "institution_id": str(inst), "age_band": str(band),
            "N": int(len(g)), "n_positive": int((g.y_true == 1).sum()), "n_negative": int((g.y_true == 0).sum()),
        })
    tab = pd.DataFrame(records).set_index(["institution_id", "age_band"]).reindex(order)
    if tab[["N", "n_positive", "n_negative"]].isna().any().any():
        missing = tab[tab.N.isna()].index.tolist()
        raise ValueError(f"Missing institution-age cells: {missing}")
    tab = tab.reset_index()
    for c in ["N", "n_positive", "n_negative"]:
        tab[c] = tab[c].astype(int)
    tab["context_id"] = tab.institution_id.astype(str) + "|" + tab.age_band.astype(str)
    s = protocol["context"]["support"]
    tab["eligible"] = (
        (tab.N >= int(s["min_N"])) &
        (tab.n_positive >= int(s["min_positive"])) &
        (tab.n_negative >= int(s["min_negative"]))
    )
    return tab


def natural_contexts(rows: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    keep = set(support.loc[support.eligible, "context_id"].astype(str))
    d = rows[rows.context_id.isin(keep)].copy()
    records = []
    for (inst, band, ctx), g in d.groupby(["institution_id", "age_band", "context_id"], sort=True):
        bacc = balanced_accuracy(g.y_true.to_numpy(), g.y_pred.to_numpy())
        records.append({
            "institution_id": str(inst), "age_band": str(band), "context_id": str(ctx),
            "N": int(len(g)), "n_positive": int((g.y_true == 1).sum()), "n_negative": int((g.y_true == 0).sum()),
            "balanced_accuracy": bacc, "risk": float(1.0 - bacc),
        })
    return pd.DataFrame(records)


def t_het(contexts: pd.DataFrame, ddof: int = 0) -> float:
    vals = []
    for _, g in contexts.groupby("institution_id", sort=True):
        r = g.risk.to_numpy(dtype=float)
        if len(r) < 2:
            raise ValueError("T_HET requires at least two eligible contexts per institution")
        vals.append(float(np.var(r, ddof=ddof)))
    return float(np.mean(vals))


def _sequential_binary_correct_counts(total_correct: int, total_incorrect: int, slot_sizes: np.ndarray, replicates: int, rng: np.random.Generator) -> np.ndarray:
    slot_sizes = np.asarray(slot_sizes, dtype=np.int64)
    if int(slot_sizes.sum()) != int(total_correct + total_incorrect):
        raise ValueError("Class slot sizes do not exhaust finite population")
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


def exact_support_matched_null(rows: pd.DataFrame, contexts: pd.DataFrame, replicates: int, seed: int, ddof: int = 0) -> np.ndarray:
    keep = set(contexts.context_id.astype(str))
    d = rows[rows.context_id.isin(keep)].copy()
    rng = np.random.default_rng(seed)
    inst_vars = []
    for inst, slots in contexts.groupby("institution_id", sort=True):
        g = d[d.institution_id == inst]
        yt = g.y_true.to_numpy(dtype=np.int8)
        yp = g.y_pred.to_numpy(dtype=np.int8)
        pos, neg = yt == 1, yt == 0
        pos_correct = int(np.sum(yp[pos] == 1))
        neg_correct = int(np.sum(yp[neg] == 0))
        n_pos = slots.n_positive.to_numpy(dtype=np.int64)
        n_neg = slots.n_negative.to_numpy(dtype=np.int64)
        if int(n_pos.sum()) != int(pos.sum()) or int(n_neg.sum()) != int(neg.sum()):
            raise ValueError(f"Context slots do not exhaust institution support: {inst}")
        tp = _sequential_binary_correct_counts(pos_correct, int(pos.sum()) - pos_correct, n_pos, replicates, rng)
        tn = _sequential_binary_correct_counts(neg_correct, int(neg.sum()) - neg_correct, n_neg, replicates, rng)
        bacc = 0.5 * (tp / n_pos[None, :] + tn / n_neg[None, :])
        risk = 1.0 - bacc
        inst_vars.append(np.var(risk, axis=1, ddof=ddof))
    return np.mean(np.vstack(inst_vars), axis=0)


def monte_carlo_upper_p(null_values: np.ndarray, observed: float) -> float:
    z = np.asarray(null_values, dtype=float)
    if len(z) == 0 or not np.isfinite(z).all():
        raise ValueError("Null must be non-empty and finite")
    return float((1 + np.sum(z >= observed)) / (len(z) + 1))


def status_polarity_reversal(reference: str, candidate: str) -> bool:
    definite = {"DEFINITE_FAILURE", "DEFINITE_NONFAILURE"}
    return reference in definite and candidate in definite and reference != candidate


def evaluate_primary_gate(
    *, linkage_fraction: float, fingerprint_match: bool, same_eligible_contexts: bool,
    cluster_concordance: int, dedup_concordance: int,
    cluster_polarity_reversals: int, dedup_polarity_reversals: int,
    replay_p: float, natural_t: float, null_median: float, protocol: dict,
) -> dict:
    alpha = float(protocol["class_support_replay"]["alpha"])
    checks = {
        "gate1_linkage_100pct": bool(np.isclose(linkage_fraction, 1.0, rtol=0, atol=0)),
        "gate2_identity_fingerprint_match": bool(fingerprint_match),
        "gate3_same_13_eligible_contexts": bool(same_eligible_contexts),
        "gate4_cluster_status_concordance_3of3": int(cluster_concordance) == 3,
        "gate5_dedup_status_concordance_3of3": int(dedup_concordance) == 3,
        "gate6_zero_polarity_reversals": int(cluster_polarity_reversals) == 0 and int(dedup_polarity_reversals) == 0,
        "gate7_replay_p_and_median": bool(replay_p <= alpha and natural_t > null_median),
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "pass": passed,
        "decision": protocol["primary_gate"]["pass_label"] if passed else protocol["primary_gate"]["fail_label"],
    }
