"""CMR-V1 X1 - exact kernel-prototype representation-layer collision.

Frozen authority (read-only; never tuned by this implementation):

- ``docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md`` (section 7);
- ``configs/cross_mechanism_replication_protocol_v1.yaml`` (key ``x1``);
- ``docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv`` (stage X1).

The X1 witness is analytic and exact: two class-conditional finite Fourier-feature
patterns whose class counts and class vector sums collide exactly, while the
downstream representation-layer risk is inverted between the two tasks.

Scope guard: this module implements only X1. It touches no real dataset, no
encoder, no random Fourier map and no X2-X5 machinery. All arithmetic uses
``float64``; the frozen numeric tolerance ``1e-12`` must never be relaxed.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
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
