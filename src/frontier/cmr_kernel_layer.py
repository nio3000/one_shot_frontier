"""CMR-V1 kernel-prototype computations for the frozen X1 and X2 stages.

Frozen authority (read-only; never tuned by this implementation):

- ``docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md`` (sections 5 and 7-8);
- ``configs/cross_mechanism_replication_protocol_v1.yaml`` (keys ``x1``, ``x2``, ``rff``);
- ``docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv`` (stages X1 and X2).

The X1 witness is analytic and exact: two class-conditional finite Fourier-feature
patterns whose class counts and class vector sums collide exactly, while the
downstream representation-layer risk is inverted between the two tasks.

X2 machinery in this module is arithmetic only: frozen Gaussian-RBF maps, the
one-shot class-sum message, prototype reconstruction and the frozen X2 gates. It
performs no dataset access and no encoder forward pass; feature extraction lives
in :mod:`frontier.cmr_feature_extract`. All arithmetic uses ``float64``; the
frozen X1 tolerance ``1e-12`` must never be relaxed.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROTOCOL_ID = "CMR-V1"
PROTOCOL_VERSION = "1.0.0-FROZEN"
PROTOCOL_STATUS = "FROZEN_BEFORE_CMR_OUTCOME_ACCESS"

FLOAT_DTYPE = np.float64
FLOAT_DTYPE_NAME = "float64"

# --- Frozen X1 constants (protocol section 7.2) ------------------------------
X1_A = Fraction(1, 5)
X1_Q_SQUARED = Fraction(24, 25)
X1_A_FLOAT = 1.0 / 5.0
X1_Q_FLOAT = math.sqrt(24.0 / 25.0)

X1_CLASS_COUNT = 6
X1_CLASSES: tuple[int, ...] = (0, 1)
X1_LAYERS: tuple[str, ...] = ("layer1", "layer2")

X1_NUMERIC_TOLERANCE = 1e-12
OPTIMAL_SET_TOLERANCE = 1e-12

X1_TASK_LAYER_PATTERNS: dict[str, dict[str, str]] = {
    "A": {"layer1": "good", "layer2": "bad"},
    "B": {"layer1": "bad", "layer2": "good"},
}

X1_EXPECTED_BACC: dict[str, dict[str, float]] = {
    "A": {"layer1": 1.0, "layer2": 1.0 / 3.0},
    "B": {"layer1": 1.0 / 3.0, "layer2": 1.0},
}
X1_EXPECTED_DETERMINISTIC_REGRET = 2.0 / 3.0
X1_EXPECTED_RANDOMIZED_REGRET = 1.0 / 3.0

# Symbolic recipes: point = (x_coeff * a, y_coeff * q), multiplicity.
# These encode the frozen Good/Bad patterns with exact rational arithmetic.
_SYMBOLIC_PATTERNS: dict[str, dict[int, tuple[tuple[int, int, int], ...]]] = {
    "good": {
        0: ((-1, 1, 3), (-1, -1, 3)),
        1: ((1, 1, 3), (1, -1, 3)),
    },
    "bad": {
        0: ((1, 1, 2), (1, -1, 2), (-5, 0, 2)),
        1: ((-1, 1, 2), (-1, -1, 2), (5, 0, 2)),
    },
}


# --- Serialization helpers ---------------------------------------------------
def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / Fraction objects into JSON-safe values."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Fraction):
        return str(obj)
    return obj


def canonical_dumps(obj: Any) -> str:
    """Deterministic JSON serialization, used for rerun equality checks."""
    return json.dumps(
        to_jsonable(obj), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )


# --- Pattern construction ----------------------------------------------------
def construct_x1_patterns() -> dict[str, dict[int, np.ndarray]]:
    """Return the frozen Good/Bad patterns as float64 arrays.

    Good: class 0 has ``(-a, +q)`` and ``(-a, -q)`` repeated three times;
    class 1 has ``(+a, +q)`` and ``(+a, -q)`` repeated three times.

    Bad: class 0 has two ``(+a, +q)``, two ``(+a, -q)`` and two ``(-1, 0)``;
    class 1 has two ``(-a, +q)``, two ``(-a, -q)`` and two ``(+1, 0)``.
    """
    a = X1_A_FLOAT
    q = X1_Q_FLOAT
    good = {
        0: np.tile(np.array([[-a, q], [-a, -q]], dtype=FLOAT_DTYPE), (3, 1)),
        1: np.tile(np.array([[a, q], [a, -q]], dtype=FLOAT_DTYPE), (3, 1)),
    }
    bad = {
        0: np.concatenate(
            [
                np.tile(np.array([[a, q]], dtype=FLOAT_DTYPE), (2, 1)),
                np.tile(np.array([[a, -q]], dtype=FLOAT_DTYPE), (2, 1)),
                np.tile(np.array([[-1.0, 0.0]], dtype=FLOAT_DTYPE), (2, 1)),
            ],
            axis=0,
        ),
        1: np.concatenate(
            [
                np.tile(np.array([[-a, q]], dtype=FLOAT_DTYPE), (2, 1)),
                np.tile(np.array([[-a, -q]], dtype=FLOAT_DTYPE), (2, 1)),
                np.tile(np.array([[1.0, 0.0]], dtype=FLOAT_DTYPE), (2, 1)),
            ],
            axis=0,
        ),
    }
    return {"good": good, "bad": bad}


def construct_x1_tasks() -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Return Task A / Task B layer-to-pattern assignments (protocol section 7.3)."""
    patterns = construct_x1_patterns()
    tasks: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for task, layer_patterns in X1_TASK_LAYER_PATTERNS.items():
        tasks[task] = {layer: patterns[name] for layer, name in layer_patterns.items()}
    return tasks


def pattern_features_labels(
    pattern: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a class-pattern dict into ``(features, labels)`` in class order."""
    classes = sorted(int(c) for c in pattern)
    features = np.concatenate(
        [np.asarray(pattern[c], dtype=FLOAT_DTYPE) for c in classes], axis=0
    )
    labels = np.concatenate(
        [np.full(len(pattern[c]), c, dtype=np.int64) for c in classes], axis=0
    )
    return features, labels


def symbolic_pattern_message(pattern_name: str) -> dict[int, dict[str, Any]]:
    """Exact rational class counts and coordinate sums of a frozen pattern.

    Coordinates are represented in the exact basis ``(a, q)``:
    ``x_sum_over_a`` is the exact X coordinate divided by ``a = 1/5`` and
    ``y_sum_over_q`` the exact Y coordinate divided by ``q = sqrt(24/25)``.
    """
    if pattern_name not in _SYMBOLIC_PATTERNS:
        raise KeyError(f"unknown frozen pattern: {pattern_name!r}")
    message: dict[int, dict[str, Any]] = {}
    for c in X1_CLASSES:
        items = _SYMBOLIC_PATTERNS[pattern_name][c]
        message[c] = {
            "count": sum(mult for _, _, mult in items),
            "x_sum_over_a": sum(Fraction(xc) * mult for xc, _, mult in items),
            "y_sum_over_q": sum(Fraction(yc) * mult for _, yc, mult in items),
        }
    return message


def symbolic_class_sums_equal() -> dict[str, Any]:
    """Exact (Fraction-level) check that Good and Bad class messages coincide."""
    good = symbolic_pattern_message("good")
    bad = symbolic_pattern_message("bad")
    counts_equal = all(good[c]["count"] == bad[c]["count"] for c in X1_CLASSES)
    sums_equal = all(
        good[c]["x_sum_over_a"] == bad[c]["x_sum_over_a"]
        and good[c]["y_sum_over_q"] == bad[c]["y_sum_over_q"]
        for c in X1_CLASSES
    )
    return {
        "class_counts_equal": bool(counts_equal),
        "class_sums_equal": bool(sums_equal),
        "good": good,
        "bad": bad,
    }


# --- One-shot kernel-prototype message and classifier ------------------------
def class_message(
    features: np.ndarray, labels: np.ndarray
) -> dict[int, dict[str, Any]]:
    """Per-class communication object ``(count, vector_sum)`` for one layer."""
    x = np.asarray(features, dtype=FLOAT_DTYPE)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("features must be a 2-D array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and labels must have the same length")
    message: dict[int, dict[str, Any]] = {}
    for c in sorted(set(int(v) for v in y.tolist())):
        mask = y == c
        message[int(c)] = {
            "count": int(np.sum(mask)),
            "vector_sum": np.sum(x[mask], axis=0, dtype=FLOAT_DTYPE),
        }
    return message


def pattern_message(pattern: Mapping[int, np.ndarray]) -> dict[int, dict[str, Any]]:
    """Per-class message of a frozen pattern dict."""
    features, labels = pattern_features_labels(pattern)
    return class_message(features, labels)


def reconstruct_prototypes(
    message: Mapping[int, Mapping[str, Any]],
) -> dict[int, np.ndarray]:
    """Server-side reconstruction ``mu_c = s_c / n_c``."""
    prototypes: dict[int, np.ndarray] = {}
    for c in sorted(int(k) for k in message):
        entry = message[c]
        count = int(entry["count"])
        if count <= 0:
            raise ValueError(f"class {c} has non-positive count")
        prototypes[c] = np.asarray(entry["vector_sum"], dtype=FLOAT_DTYPE) / float(count)
    return prototypes


def predict_prototype(x: np.ndarray, prototypes: Mapping[int, np.ndarray]) -> int:
    """Prototype classifier ``argmax_c <x, mu_c>``; ties use smallest class ID."""
    xv = np.asarray(x, dtype=FLOAT_DTYPE)
    classes = sorted(int(c) for c in prototypes)
    scores = np.array(
        [float(np.dot(xv, np.asarray(prototypes[c], dtype=FLOAT_DTYPE))) for c in classes],
        dtype=FLOAT_DTYPE,
    )
    return int(classes[int(np.argmax(scores))])


def predict_all(features: np.ndarray, prototypes: Mapping[int, np.ndarray]) -> np.ndarray:
    x = np.asarray(features, dtype=FLOAT_DTYPE)
    return np.array([predict_prototype(row, prototypes) for row in x], dtype=np.int64)


def balanced_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    prototypes: Mapping[int, np.ndarray],
) -> float:
    """Balanced accuracy: unweighted mean of per-class recall."""
    predictions = predict_all(features, prototypes)
    y = np.asarray(labels, dtype=np.int64)
    per_class: list[float] = []
    for c in sorted(set(int(v) for v in y.tolist())):
        mask = y == c
        per_class.append(float(np.mean(predictions[mask] == c)))
    return float(np.mean(np.asarray(per_class, dtype=FLOAT_DTYPE)))


# --- Downstream decision -----------------------------------------------------
def optimal_action_set(
    risk_by_action: Mapping[str, float],
    tolerance: float = OPTIMAL_SET_TOLERANCE,
) -> tuple[str, ...]:
    """Layer actions whose risk is within ``tolerance`` of the minimum risk."""
    actions = [a for a in X1_LAYERS if a in risk_by_action]
    if len(actions) != len(risk_by_action):
        raise ValueError("risk mapping must use the canonical X1 layer keys")
    best = min(float(risk_by_action[a]) for a in actions)
    return tuple(
        a for a in actions if float(risk_by_action[a]) - best <= tolerance
    )


def _canonical_actions(
    risk_a: Mapping[str, float], risk_b: Mapping[str, float]
) -> list[str]:
    if set(risk_a) != set(risk_b) or set(risk_a) != set(X1_LAYERS):
        raise ValueError("both risk mappings must use the canonical X1 layer keys")
    return [a for a in X1_LAYERS if a in risk_a]


def deterministic_pair_minimax_regret(
    risk_a: Mapping[str, float], risk_b: Mapping[str, float]
) -> dict[str, Any]:
    """Deterministic pair minimax regret (protocol section 6)."""
    actions = _canonical_actions(risk_a, risk_b)
    star_a = min(float(risk_a[a]) for a in actions)
    star_b = min(float(risk_b[a]) for a in actions)
    per_action: dict[str, dict[str, float]] = {}
    best_action = actions[0]
    best_value: float | None = None
    for a in actions:
        regret_a = float(risk_a[a]) - star_a
        regret_b = float(risk_b[a]) - star_b
        value = max(regret_a, regret_b)
        per_action[a] = {
            "regret_A": regret_a,
            "regret_B": regret_b,
            "max_regret": value,
        }
        if best_value is None or value < best_value:
            best_value = value
            best_action = a
    return {
        "value": float(best_value),
        "argmin_action": best_action,
        "optimal_risk_A": star_a,
        "optimal_risk_B": star_b,
        "per_action": per_action,
    }


def randomized_pair_minimax_regret(
    risk_a: Mapping[str, float], risk_b: Mapping[str, float]
) -> dict[str, Any]:
    """Minimax regret over randomized selectors, solved on the simplex.

    ``min over distributions p on actions of max(E_p regret_A, E_p regret_B)``.
    With two regret constraints the LP optimum is supported on at most two
    actions; each candidate is the exact equalization point of the two
    expected-regret constraints, so the returned value is derived from the
    risk matrix only (never hard-coded).
    """
    actions = _canonical_actions(risk_a, risk_b)
    star_a = min(float(risk_a[a]) for a in actions)
    star_b = min(float(risk_b[a]) for a in actions)
    regret_a = np.array([float(risk_a[a]) - star_a for a in actions], dtype=FLOAT_DTYPE)
    regret_b = np.array([float(risk_b[a]) - star_b for a in actions], dtype=FLOAT_DTYPE)
    m = len(actions)

    best: dict[str, Any] | None = None
    for i in range(m):
        value = float(max(regret_a[i], regret_b[i]))
        if best is None or value < best["value"]:
            best = {"value": value, "support_idx": [i], "probs": [1.0]}
    for i in range(m):
        for j in range(i + 1, m):
            den = float(
                (regret_a[i] - regret_a[j]) - (regret_b[i] - regret_b[j])
            )
            if den == 0.0:
                continue
            p = float((regret_b[j] - regret_a[j]) / den)
            if p < 0.0 or p > 1.0:
                continue
            expected_a = float(p * regret_a[i] + (1.0 - p) * regret_a[j])
            expected_b = float(p * regret_b[i] + (1.0 - p) * regret_b[j])
            value = max(expected_a, expected_b)
            if value < best["value"]:
                best = {
                    "value": value,
                    "support_idx": [i, j],
                    "probs": [p, 1.0 - p],
                }

    support = [actions[i] for i in best["support_idx"]]
    probabilities = [float(p) for p in best["probs"]]
    expected_a = float(
        sum(p * float(risk_a[a]) for p, a in zip(probabilities, support)) - star_a
    )
    expected_b = float(
        sum(p * float(risk_b[a]) for p, a in zip(probabilities, support)) - star_b
    )
    value = max(expected_a, expected_b)
    if abs(value - float(best["value"])) > X1_NUMERIC_TOLERANCE:
        raise RuntimeError("randomized minimax solver inconsistency")
    return {
        "value": value,
        "support": support,
        "probabilities": probabilities,
        "expected_regret_A": expected_a,
        "expected_regret_B": expected_b,
        "optimal_risk_A": star_a,
        "optimal_risk_B": star_b,
    }


# --- Full X1 evaluation ------------------------------------------------------
def evaluate_x1_exact_collision() -> dict[str, Any]:
    """Run the complete frozen X1 witness and return a JSON-safe result."""
    patterns = construct_x1_patterns()
    tasks = construct_x1_tasks()
    tolerance = X1_NUMERIC_TOLERANCE

    messages = {
        task: {
            layer: pattern_message(tasks[task][layer]) for layer in X1_LAYERS
        }
        for task in ("A", "B")
    }
    prototypes = {
        task: {
            layer: reconstruct_prototypes(messages[task][layer])
            for layer in X1_LAYERS
        }
        for task in ("A", "B")
    }

    counts_equal = True
    max_sum_diff = 0.0
    per_layer_sum_diff: dict[str, dict[int, float]] = {
        layer: {} for layer in X1_LAYERS
    }
    for layer in X1_LAYERS:
        for c in X1_CLASSES:
            entry_a = messages["A"][layer][c]
            entry_b = messages["B"][layer][c]
            if int(entry_a["count"]) != int(entry_b["count"]):
                counts_equal = False
            diff = float(
                np.max(
                    np.abs(
                        np.asarray(entry_a["vector_sum"], dtype=FLOAT_DTYPE)
                        - np.asarray(entry_b["vector_sum"], dtype=FLOAT_DTYPE)
                    )
                )
            )
            per_layer_sum_diff[layer][c] = diff
            max_sum_diff = max(max_sum_diff, diff)

    max_prototype_diff = 0.0
    per_layer_prototype_diff: dict[str, dict[int, float]] = {
        layer: {} for layer in X1_LAYERS
    }
    for layer in X1_LAYERS:
        for c in X1_CLASSES:
            diff = float(
                np.max(
                    np.abs(prototypes["A"][layer][c] - prototypes["B"][layer][c])
                )
            )
            per_layer_prototype_diff[layer][c] = diff
            max_prototype_diff = max(max_prototype_diff, diff)

    symbolic = symbolic_class_sums_equal()
    symbolic_equal = bool(
        symbolic["class_counts_equal"] and symbolic["class_sums_equal"]
    )

    bacc: dict[str, dict[str, float]] = {}
    risk: dict[str, dict[str, float]] = {}
    for task in ("A", "B"):
        bacc[task] = {}
        risk[task] = {}
        for layer in X1_LAYERS:
            features, labels = pattern_features_labels(tasks[task][layer])
            value = balanced_accuracy(features, labels, prototypes[task][layer])
            bacc[task][layer] = value
            risk[task][layer] = 1.0 - value

    optimal_sets = {task: optimal_action_set(risk[task]) for task in ("A", "B")}
    deterministic = deterministic_pair_minimax_regret(risk["A"], risk["B"])
    randomized = randomized_pair_minimax_regret(risk["A"], risk["B"])

    intersection = sorted(set(optimal_sets["A"]) & set(optimal_sets["B"]))

    gates = {
        "X1-G1": {
            "name": "message equality",
            "passed": bool(
                counts_equal and symbolic_equal and max_sum_diff <= tolerance
            ),
            "class_counts_equal": bool(counts_equal),
            "class_sums_symbolically_equal": symbolic_equal,
            "max_abs_vector_sum_diff": max_sum_diff,
            "numeric_tolerance": tolerance,
            "protocol_ref": "protocol section 7.4 conditions 1-3; gate matrix X1-G1",
        },
        "X1-G2": {
            "name": "prototype equality",
            "passed": bool(max_prototype_diff <= tolerance),
            "max_abs_prototype_diff": max_prototype_diff,
            "numeric_tolerance": tolerance,
            "protocol_ref": "protocol section 7.4 condition 4",
        },
        "X1-G3": {
            "name": "optimal-set conflict",
            "passed": bool(
                len(optimal_sets["A"]) >= 1
                and len(optimal_sets["B"]) >= 1
                and len(intersection) == 0
            ),
            "optimal_sets": {
                "A": list(optimal_sets["A"]),
                "B": list(optimal_sets["B"]),
            },
            "intersection": intersection,
            "protocol_ref": "protocol section 7.4 condition 5; gate matrix X1-G2",
        },
        "X1-G4": {
            "name": "deterministic regret",
            "passed": bool(
                abs(deterministic["value"] - X1_EXPECTED_DETERMINISTIC_REGRET)
                <= tolerance
            ),
            "value": deterministic["value"],
            "expected": X1_EXPECTED_DETERMINISTIC_REGRET,
            "numeric_tolerance": tolerance,
            "protocol_ref": "protocol section 7.4 condition 6; gate matrix X1-G3",
        },
        "X1-G5": {
            "name": "randomized regret",
            "passed": bool(
                abs(randomized["value"] - X1_EXPECTED_RANDOMIZED_REGRET)
                <= tolerance
            ),
            "value": randomized["value"],
            "expected": X1_EXPECTED_RANDOMIZED_REGRET,
            "support": randomized["support"],
            "probabilities": randomized["probabilities"],
            "numeric_tolerance": tolerance,
            "protocol_ref": "protocol section 7.4 condition 7; gate matrix X1-G4",
        },
    }

    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_status": PROTOCOL_STATUS,
        "float_dtype": FLOAT_DTYPE_NAME,
        "spec": {
            "mode": "exact_analytic",
            "feature_map": "[cos(theta), sin(theta)]",
            "a": X1_A_FLOAT,
            "q": X1_Q_FLOAT,
            "q_squared": float(X1_Q_SQUARED),
            "class_count": X1_CLASS_COUNT,
            "classes": list(X1_CLASSES),
            "layers": list(X1_LAYERS),
            "numeric_tolerance": tolerance,
            "task_layer_patterns": {
                task: dict(layer_patterns)
                for task, layer_patterns in X1_TASK_LAYER_PATTERNS.items()
            },
            "patterns": patterns,
        },
        "messages": messages,
        "prototypes": prototypes,
        "symbolic": symbolic,
        "message_equality": {
            "class_counts_equal": bool(counts_equal),
            "class_sums_symbolically_equal": symbolic_equal,
            "max_abs_vector_sum_diff": max_sum_diff,
            "per_layer_per_class_max_abs": per_layer_sum_diff,
            "numeric_tolerance": tolerance,
        },
        "prototype_equality": {
            "max_abs_diff": max_prototype_diff,
            "per_layer_per_class_max_abs": per_layer_prototype_diff,
            "numeric_tolerance": tolerance,
        },
        "balanced_accuracy": bacc,
        "risk": risk,
        "optimal_sets": {
            "A": list(optimal_sets["A"]),
            "B": list(optimal_sets["B"]),
        },
        "deterministic_regret": deterministic,
        "randomized_regret": randomized,
        "expected": {
            "bacc": {
                task: dict(layer_values)
                for task, layer_values in X1_EXPECTED_BACC.items()
            },
            "deterministic_regret": X1_EXPECTED_DETERMINISTIC_REGRET,
            "randomized_regret": X1_EXPECTED_RANDOMIZED_REGRET,
            "numeric_tolerance": tolerance,
        },
        "gates": gates,
        "core_gates_pass": bool(all(gate["passed"] for gate in gates.values())),
    }
    return to_jsonable(result)


# =============================================================================
# X2 - action relevance of the frozen representation-layer action family
# =============================================================================
# Frozen action family (protocol section 5.1). No layer may be added, removed or
# substituted after protocol freeze; these tuples are checked against the frozen
# protocol YAML at load time by :func:`load_x2_protocol_constants`.
X2_ENCODER_LAYERS: dict[str, tuple[str, ...]] = {
    "resnet18_imagenet1k_v1": ("layer1", "layer2", "layer3", "layer4"),
    "vit_b_16_imagenet1k_v1": ("B3", "B6", "B9", "B12"),
}
X2_DATASETS: tuple[str, ...] = ("cifar100", "eurosat", "pathmnist", "dermamnist")
X2_CONDITIONS: tuple[tuple[str, str], ...] = tuple(
    (dataset, encoder) for dataset in X2_DATASETS for encoder in X2_ENCODER_LAYERS
)

X2_DECISION_PASS = "X2_PASS_ACTION_FAMILY_DECISION_RELEVANT"
X2_DECISION_FAIL = "ACTION_FAMILY_NOT_DECISION_RELEVANT"
X2_FAIL_FINAL_CMR_DECISION = "CMR-D"

X2_RFF_MASTER_SEED = 20261001
X2_RFF_DIMENSION = 256
X2_RFF_SIGMA = 1.0
X2_AGGREGATION_TOLERANCE = 1e-8
X2_ACTION_RELEVANT_SPREAD_MIN = 0.01
X2_UNIQUE_BEST_MARGIN_MIN = 0.002

# Repository-relative frozen protocol authority (existence is asserted, never modified).
CMR_PROTOCOL_YAML_REL = "configs/cross_mechanism_replication_protocol_v1.yaml"


def rff_seed(encoder_id: str, layer_id: str, master_seed: int = X2_RFF_MASTER_SEED) -> int:
    """Frozen per-layer RFF seed derivation (protocol section 5.2).

    ``sha256(f"{master_seed}|{encoder_id}|{layer_id}")`` -> first 8 bytes ->
    big-endian -> modulo ``2**32``.
    """
    if encoder_id not in X2_ENCODER_LAYERS:
        raise ValueError(f"encoder outside frozen action family: {encoder_id!r}")
    if layer_id not in X2_ENCODER_LAYERS[encoder_id]:
        raise ValueError(f"layer outside frozen action family: {layer_id!r}")
    digest = hashlib.sha256(f"{master_seed}|{encoder_id}|{layer_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def generate_rff(
    encoder_id: str,
    layer_id: str,
    input_dim: int,
    master_seed: int = X2_RFF_MASTER_SEED,
    dimension: int = X2_RFF_DIMENSION,
    sigma: float = X2_RFF_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen Gaussian-RBF map: ``W ~ N(0, sigma^-2)``, ``b ~ U(0, 2*pi)``, float64."""
    if int(input_dim) <= 0:
        raise ValueError("input_dim must be positive")
    generator = np.random.Generator(np.random.PCG64(rff_seed(encoder_id, layer_id, master_seed)))
    w = generator.normal(0.0, float(sigma) ** -1.0, size=(int(dimension), int(input_dim)))
    b = generator.uniform(0.0, 2.0 * np.pi, size=int(dimension))
    return (
        np.ascontiguousarray(w, dtype=FLOAT_DTYPE),
        np.ascontiguousarray(b, dtype=FLOAT_DTYPE),
    )


def array_sha256(array: np.ndarray) -> str:
    """Content hash of a frozen numeric array (dtype, shape, C-order bytes)."""
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def normalize_layer_features(features: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization of frozen layer features in ``float64``."""
    x = np.asarray(features, dtype=FLOAT_DTYPE)
    if x.ndim != 2:
        raise ValueError("layer features must be a 2-D array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("non-finite or zero-norm layer feature")
    return x / norms


def apply_rff(
    features: np.ndarray, w: np.ndarray, b: np.ndarray, dimension: int = X2_RFF_DIMENSION
) -> np.ndarray:
    """Apply the frozen map: ``sqrt(2/m) * cos(W u + b)``."""
    x = np.asarray(features, dtype=FLOAT_DTYPE)
    w = np.asarray(w, dtype=FLOAT_DTYPE)
    b = np.asarray(b, dtype=FLOAT_DTYPE)
    if x.ndim != 2:
        raise ValueError("features must be a 2-D array")
    if w.shape != (int(dimension), x.shape[1]) or b.shape != (int(dimension),):
        raise ValueError(
            f"frozen RFF shape mismatch: W {w.shape} b {b.shape} for input {x.shape[1]}"
        )
    return np.sqrt(2.0 / float(dimension)) * np.cos(x @ w.T + b)


def aggregate_client_messages(
    features: np.ndarray,
    labels: np.ndarray,
    clients: np.ndarray,
    n_clients: int,
    tolerance: float = X2_AGGREGATION_TOLERANCE,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """One-shot client aggregation and its centralized recovery check.

    Only class counts and class vector sums are transmitted; the reconstruction
    of the global class sums from client messages must agree with the
    centralized class sums to ``tolerance``.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    clients = np.asarray(clients, dtype=np.int64).reshape(-1)
    x = np.asarray(features, dtype=FLOAT_DTYPE)
    if x.ndim != 2 or x.shape[0] != labels.shape[0]:
        raise ValueError("features/labels shape mismatch")
    if clients.shape != labels.shape:
        raise ValueError("client assignment must cover every record exactly once")
    if np.any((clients < 0) | (clients >= int(n_clients))):
        raise ValueError("invalid client assignment")
    centralized = class_message(x, labels)
    aggregated = {
        int(c): {"count": 0, "vector_sum": np.zeros_like(entry["vector_sum"])}
        for c, entry in centralized.items()
    }
    for client in range(int(n_clients)):
        mask = clients == client
        if not np.any(mask):
            continue
        for c, entry in class_message(x[mask], labels[mask]).items():
            aggregated[int(c)]["count"] += int(entry["count"])
            aggregated[int(c)]["vector_sum"] = (
                aggregated[int(c)]["vector_sum"] + entry["vector_sum"]
            )
    counts_equal = all(
        int(aggregated[int(c)]["count"]) == int(centralized[int(c)]["count"])
        for c in centralized
    )
    max_abs = max(
        float(
            np.max(
                np.abs(
                    aggregated[int(c)]["vector_sum"] - centralized[int(c)]["vector_sum"]
                )
            )
        )
        for c in centralized
    )
    recovery = {
        "counts_equal": bool(counts_equal),
        "max_abs": max_abs,
        "tolerance": float(tolerance),
        "passed": bool(counts_equal and max_abs <= float(tolerance)),
    }
    return aggregated, recovery


def predict_prototype_batch(
    features: np.ndarray,
    prototypes: Mapping[int, np.ndarray],
    batch_size: int = 4096,
) -> np.ndarray:
    """Prototype classifier ``argmax_c <phi(x), mu_c>``; ties use smallest class ID.

    Classes are evaluated in ascending canonical order, so ``np.argmax`` resolves
    an exact score tie to the smallest canonical class ID.
    """
    if not prototypes:
        raise ValueError("no prototypes supplied")
    classes = np.asarray(sorted(int(c) for c in prototypes), dtype=np.int64)
    means = np.stack(
        [np.asarray(prototypes[int(c)], dtype=FLOAT_DTYPE) for c in classes]
    ).astype(FLOAT_DTYPE)
    x = np.asarray(features, dtype=FLOAT_DTYPE)
    if x.ndim != 2 or x.shape[1] != means.shape[1]:
        raise ValueError("feature/prototype dimension mismatch")
    predictions = np.empty(x.shape[0], dtype=np.int64)
    for start in range(0, x.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), x.shape[0])
        scores = x[start:stop] @ means.T
        predictions[start:stop] = classes[np.argmax(scores, axis=1)]
    return predictions


def bacc_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Balanced accuracy = unweighted mean of per-class recall."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("labels/predictions shape mismatch")
    classes = sorted(set(int(v) for v in y.tolist()))
    if not classes:
        raise ValueError("no labels supplied")
    recalls = [float(np.mean(p[y == c] == c)) for c in classes]
    return float(np.mean(np.asarray(recalls, dtype=FLOAT_DTYPE)))


def summarize_x2_condition(
    dataset: str,
    encoder: str,
    bacc_by_layer: Mapping[str, float],
    recovery: Mapping[str, Any],
) -> dict[str, Any]:
    """Per-condition X2 summary: spread, best/second-best, UNIQUE_BEST, recovery."""
    if encoder not in X2_ENCODER_LAYERS:
        raise ValueError(f"encoder outside frozen action family: {encoder!r}")
    actions = X2_ENCODER_LAYERS[encoder]
    if set(bacc_by_layer) != set(actions):
        raise ValueError("incomplete frozen action family for the condition")
    bacc = {layer: float(bacc_by_layer[layer]) for layer in actions}
    # Deterministic ordering: descending BACC, then frozen action-family order.
    order = sorted(actions, key=lambda layer: (-bacc[layer], actions.index(layer)))
    best, second = order[0], order[1]
    margin = float(bacc[best] - bacc[second])
    spread = float(max(bacc.values()) - min(bacc.values()))
    finite = bool(np.isfinite(np.asarray(list(bacc.values()) + [float(recovery["max_abs"])])).all())
    return {
        "dataset": dataset,
        "encoder": encoder,
        "balanced_accuracy": bacc,
        "best_layer": best,
        "second_best_layer": second,
        "best_margin": margin,
        "layer_spread": spread,
        "action_relevant": bool(finite and spread >= X2_ACTION_RELEVANT_SPREAD_MIN),
        "unique_best": best if (finite and margin >= X2_UNIQUE_BEST_MARGIN_MIN) else None,
        "aggregation_recovery_max_abs": float(recovery["max_abs"]),
        "aggregation_recovery_pass": bool(recovery["passed"]),
        "finite_pass": finite,
    }


def evaluate_x2_gates(conditions: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    """Frozen X2 gate evaluation (protocol section 8.5).

    Conditions: X2-G1 >= 6/8 action-relevant, X2-G2 >= 2 distinct ResNet
    UNIQUE_BEST layers, X2-G3 >= 2 distinct ViT UNIQUE_BEST blocks, X2-G4 all
    aggregation recoveries pass, X2-G5 all values finite.
    """
    expected = set(X2_CONDITIONS)
    observed = {(str(c["dataset"]), str(c["encoder"])) for c in conditions}
    if len(conditions) != len(expected) or observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "X2 requires exactly the eight frozen structural conditions; "
            f"missing={missing} extra={extra}"
        )
    relevant = int(sum(1 for c in conditions if c["action_relevant"]))
    winners: dict[str, list[str]] = {}
    for encoder in X2_ENCODER_LAYERS:
        winners[encoder] = sorted(
            {
                str(c["unique_best"])
                for c in conditions
                if str(c["encoder"]) == encoder and c["unique_best"] is not None
            }
        )
    max_abs = max(float(c["aggregation_recovery_max_abs"]) for c in conditions)
    gates = {
        "X2-G1": {
            "name": "action-relevant structural conditions",
            "passed": bool(relevant >= 6),
            "action_relevant_count": relevant,
            "required": 6,
            "structural_conditions": len(expected),
            "spread_threshold": X2_ACTION_RELEVANT_SPREAD_MIN,
        },
        "X2-G2": {
            "name": "ResNet-18 UNIQUE_BEST layer breadth",
            "passed": bool(len(winners["resnet18_imagenet1k_v1"]) >= 2),
            "distinct_unique_best": winners["resnet18_imagenet1k_v1"],
            "required": 2,
            "margin_threshold": X2_UNIQUE_BEST_MARGIN_MIN,
        },
        "X2-G3": {
            "name": "ViT-B/16 UNIQUE_BEST block breadth",
            "passed": bool(len(winners["vit_b_16_imagenet1k_v1"]) >= 2),
            "distinct_unique_best": winners["vit_b_16_imagenet1k_v1"],
            "required": 2,
            "margin_threshold": X2_UNIQUE_BEST_MARGIN_MIN,
        },
        "X2-G4": {
            "name": "aggregation recovery",
            "passed": bool(all(bool(c["aggregation_recovery_pass"]) for c in conditions)),
            "max_abs": max_abs,
            "tolerance": X2_AGGREGATION_TOLERANCE,
        },
        "X2-G5": {
            "name": "finiteness",
            "passed": bool(all(bool(c["finite_pass"]) for c in conditions)),
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "gates": gates,
        "all_gates_pass": passed,
        "decision": X2_DECISION_PASS if passed else X2_DECISION_FAIL,
        "final_cmr_decision": None if passed else X2_FAIL_FINAL_CMR_DECISION,
        "stop": (not passed),
    }


def load_x2_protocol_constants(protocol_yaml: PathLike | None = None) -> dict[str, Any]:
    """Read and cross-check the frozen X2 authority from the protocol YAML.

    Fails closed on any drift of datasets, encoders, layers, RFF settings,
    thresholds, client partition, primary metric or final decision matrix.
    """

    def _repo_root() -> Path:
        for candidate in Path(__file__).resolve().parents:
            if (candidate / CMR_PROTOCOL_YAML_REL).is_file():
                return candidate
        raise FileNotFoundError(f"cannot locate {CMR_PROTOCOL_YAML_REL}")

    path = Path(protocol_yaml) if protocol_yaml is not None else _repo_root() / CMR_PROTOCOL_YAML_REL
    if not path.is_file():
        raise FileNotFoundError(path)
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = document["protocol"]
    x2 = document["x2"]
    rff = document["rff"]
    encoders = document["encoders"]
    classifier = document["classifier"]
    target = document["scientific_target"]
    seed_derivation = rff["seed_derivation"]
    partition = x2["client_partition"]

    failures: list[str] = []
    if protocol["id"] != PROTOCOL_ID:
        failures.append("protocol.id")
    if protocol["version"] != PROTOCOL_VERSION:
        failures.append("protocol.version")
    if protocol["status"] != PROTOCOL_STATUS:
        failures.append("protocol.status")
    if tuple(x2["datasets"]) != X2_DATASETS:
        failures.append("x2.datasets")
    if tuple(x2["encoders"]) != tuple(X2_ENCODER_LAYERS):
        failures.append("x2.encoders")
    if int(x2["structural_conditions"]) != len(X2_CONDITIONS):
        failures.append("x2.structural_conditions")
    for encoder_id, layer_tuple in X2_ENCODER_LAYERS.items():
        if tuple(encoders[encoder_id]["layers"]) != layer_tuple:
            failures.append(f"encoders.{encoder_id}.layers")
    if encoders["normalization"] != "l2":
        failures.append("encoders.normalization")
    if int(rff["dimension"]) != X2_RFF_DIMENSION:
        failures.append("rff.dimension")
    if float(rff["sigma"]) != X2_RFF_SIGMA:
        failures.append("rff.sigma")
    if int(rff["master_seed"]) != X2_RFF_MASTER_SEED:
        failures.append("rff.master_seed")
    if rff["generation_dtype"] != FLOAT_DTYPE_NAME:
        failures.append("rff.generation_dtype")
    if rff["rng"] != "numpy_Generator_PCG64":
        failures.append("rff.rng")
    if seed_derivation["string_template"] != "20261001|<encoder_id>|<layer_id>":
        failures.append("rff.seed_derivation.string_template")
    if seed_derivation["hash"] != "sha256" or str(seed_derivation["bytes"]) != "first_8":
        failures.append("rff.seed_derivation.hash_or_bytes")
    if seed_derivation["byte_order"] != "big":
        failures.append("rff.seed_derivation.byte_order")
    if int(seed_derivation["modulus"]) != 2**32:
        failures.append("rff.seed_derivation.modulus")
    if int(partition["n_clients"]) != 20:
        failures.append("x2.client_partition.n_clients")
    if float(partition["dirichlet_alpha"]) != 0.10:
        failures.append("x2.client_partition.dirichlet_alpha")
    if int(partition["seed"]) != 20260908:
        failures.append("x2.client_partition.seed")
    if bool(partition["scientific_axis"]) is not False:
        failures.append("x2.client_partition.scientific_axis")
    if float(x2["action_relevant_spread_min"]) != X2_ACTION_RELEVANT_SPREAD_MIN:
        failures.append("x2.action_relevant_spread_min")
    if float(x2["unique_best_margin_min"]) != X2_UNIQUE_BEST_MARGIN_MIN:
        failures.append("x2.unique_best_margin_min")
    if int(x2["gate"]["min_action_relevant_conditions"]) != 6:
        failures.append("x2.gate.min_action_relevant_conditions")
    if int(x2["gate"]["resnet_min_distinct_unique_best_layers"]) != 2:
        failures.append("x2.gate.resnet_min_distinct_unique_best_layers")
    if int(x2["gate"]["vit_min_distinct_unique_best_layers"]) != 2:
        failures.append("x2.gate.vit_min_distinct_unique_best_layers")
    if bool(x2["gate"]["require_all_aggregation_recovery"]) is not True:
        failures.append("x2.gate.require_all_aggregation_recovery")
    if bool(x2["gate"]["require_all_finite"]) is not True:
        failures.append("x2.gate.require_all_finite")
    if x2["fail_decision"] != X2_DECISION_FAIL:
        failures.append("x2.fail_decision")
    if bool(x2["fail_stops_protocol"]) is not True:
        failures.append("x2.fail_stops_protocol")
    if target["downstream_action"] != "representation_layer":
        failures.append("scientific_target.downstream_action")
    if target["primary_metric"] != "balanced_accuracy":
        failures.append("scientific_target.primary_metric")
    if float(document["message"]["aggregation_recovery_abs_tolerance"]) != X2_AGGREGATION_TOLERANCE:
        failures.append("message.aggregation_recovery_abs_tolerance")
    if classifier["class_tie_break"] != "smallest_canonical_class_id":
        failures.append("classifier.class_tie_break")
    if classifier["score"] != "inner_product":
        failures.append("classifier.score")
    if "X2_FAIL" not in document["final_decision"]["CMR-D"]["require_any"][0]:
        failures.append("final_decision.CMR-D.require_any")
    if failures:
        raise RuntimeError(f"frozen CMR-V1 x2 authority drift: {sorted(set(failures))}")
    return {
        "protocol_id": protocol["id"],
        "protocol_version": protocol["version"],
        "protocol_status": protocol["status"],
        "protocol_yaml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "datasets": list(X2_DATASETS),
        "encoders": {k: list(v) for k, v in X2_ENCODER_LAYERS.items()},
        "rff": {
            "dimension": X2_RFF_DIMENSION,
            "sigma": X2_RFF_SIGMA,
            "master_seed": X2_RFF_MASTER_SEED,
            "generation_dtype": FLOAT_DTYPE_NAME,
            "rng": rff["rng"],
        },
        "x2": {
            "action_relevant_spread_min": X2_ACTION_RELEVANT_SPREAD_MIN,
            "unique_best_margin_min": X2_UNIQUE_BEST_MARGIN_MIN,
            "min_action_relevant_conditions": 6,
            "aggregation_tolerance": X2_AGGREGATION_TOLERANCE,
            "client_partition": {
                "n_clients": 20,
                "dirichlet_alpha": 0.10,
                "seed": 20260908,
            },
        },
        "prohibited_after_x2_outcome_access": list(
            document["prohibited_after_x2_outcome_access"]
        ),
    }
