"""CMR-V1 X1 unit tests (T1-T9) for the exact kernel-prototype collision.

Frozen authority: ``docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md``
section 7 and ``configs/cross_mechanism_replication_protocol_v1.yaml``.

The tolerance is the frozen ``1e-12``; it must never be relaxed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.cmr_kernel_layer import (
    X1_A_FLOAT,
    X1_CLASS_COUNT,
    X1_CLASSES,
    X1_EXPECTED_DETERMINISTIC_REGRET,
    X1_EXPECTED_RANDOMIZED_REGRET,
    X1_LAYERS,
    X1_NUMERIC_TOLERANCE,
    X1_Q_FLOAT,
    balanced_accuracy,
    canonical_dumps,
    class_message,
    construct_x1_patterns,
    construct_x1_tasks,
    deterministic_pair_minimax_regret,
    evaluate_x1_exact_collision,
    optimal_action_set,
    pattern_features_labels,
    pattern_message,
    predict_prototype,
    randomized_pair_minimax_regret,
    reconstruct_prototypes,
    symbolic_class_sums_equal,
)

TOL = X1_NUMERIC_TOLERANCE


def _bacc_by_task_layer() -> dict:
    """Recompute the four BACC values directly from the frozen objects."""
    tasks = construct_x1_tasks()
    out = {}
    for task in ("A", "B"):
        out[task] = {}
        for layer in X1_LAYERS:
            prototypes = reconstruct_prototypes(pattern_message(tasks[task][layer]))
            features, labels = pattern_features_labels(tasks[task][layer])
            out[task][layer] = balanced_accuracy(features, labels, prototypes)
    return out


# -- T1: class counts ---------------------------------------------------------
def test_t01_frozen_pattern_class_counts():
    patterns = construct_x1_patterns()
    assert set(patterns) == {"good", "bad"}
    for name, pattern in patterns.items():
        features, labels = pattern_features_labels(pattern)
        message = class_message(features, labels)
        assert sorted(message) == list(X1_CLASSES), name
        for c in X1_CLASSES:
            assert pattern[c].shape == (X1_CLASS_COUNT, 2), name
            assert pattern[c].dtype == np.float64, name
            assert message[c]["count"] == X1_CLASS_COUNT, name
            assert message[c]["vector_sum"].dtype == np.float64, name


# -- T2: exact class vector sums (float + symbolic) ---------------------------
def test_t02_good_bad_class_vector_sums_match():
    patterns = construct_x1_patterns()
    message_good = pattern_message(patterns["good"])
    message_bad = pattern_message(patterns["bad"])
    for c in X1_CLASSES:
        diff = float(
            np.max(
                np.abs(
                    message_good[c]["vector_sum"] - message_bad[c]["vector_sum"]
                )
            )
        )
        assert diff <= TOL, (c, diff)
    symbolic = symbolic_class_sums_equal()
    assert symbolic["class_counts_equal"] is True
    assert symbolic["class_sums_equal"] is True
    # Exact rational witness: class 0 sum is (-6a, 0), class 1 sum is (+6a, 0).
    for c in X1_CLASSES:
        assert symbolic["good"][c]["x_sum_over_a"] == symbolic["bad"][c]["x_sum_over_a"]
        assert symbolic["good"][c]["y_sum_over_q"] == symbolic["bad"][c]["y_sum_over_q"]
    assert symbolic["good"][0]["x_sum_over_a"] == -6
    assert symbolic["good"][1]["x_sum_over_a"] == 6


# -- T3: Task A / Task B communication object equality ------------------------
def test_t03_task_communication_objects_equal():
    tasks = construct_x1_tasks()
    for layer in X1_LAYERS:
        message_a = pattern_message(tasks["A"][layer])
        message_b = pattern_message(tasks["B"][layer])
        for c in X1_CLASSES:
            assert message_a[c]["count"] == message_b[c]["count"] == X1_CLASS_COUNT
            diff = float(
                np.max(np.abs(message_a[c]["vector_sum"] - message_b[c]["vector_sum"]))
            )
            assert diff <= TOL, (layer, c, diff)


# -- T4: reconstructed prototype equality -------------------------------------
def test_t04_reconstructed_prototypes_identical():
    tasks = construct_x1_tasks()
    for layer in X1_LAYERS:
        prototypes_a = reconstruct_prototypes(pattern_message(tasks["A"][layer]))
        prototypes_b = reconstruct_prototypes(pattern_message(tasks["B"][layer]))
        for c in X1_CLASSES:
            diff = float(np.max(np.abs(prototypes_a[c] - prototypes_b[c])))
            assert diff <= TOL, (layer, c, diff)
            assert prototypes_a[c].dtype == np.float64
    # Analytic authority: mu_0 = (-a, 0), mu_1 = (+a, 0).
    prototypes = reconstruct_prototypes(pattern_message(construct_x1_patterns()["good"]))
    assert abs(prototypes[0][0] - (-X1_A_FLOAT)) <= TOL
    assert abs(prototypes[0][1]) <= TOL
    assert abs(prototypes[1][0] - X1_A_FLOAT) <= TOL
    assert abs(prototypes[1][1]) <= TOL


# -- T5: four BACC values ------------------------------------------------------
def test_t05_four_bacc_values():
    bacc = _bacc_by_task_layer()
    assert abs(bacc["A"]["layer1"] - 1.0) <= TOL
    assert abs(bacc["A"]["layer2"] - 1.0 / 3.0) <= TOL
    assert abs(bacc["B"]["layer1"] - 1.0 / 3.0) <= TOL
    assert abs(bacc["B"]["layer2"] - 1.0) <= TOL
    evaluated = evaluate_x1_exact_collision()["balanced_accuracy"]
    for task in ("A", "B"):
        for layer in X1_LAYERS:
            assert abs(evaluated[task][layer] - bacc[task][layer]) <= TOL


# -- T6: optimal sets conflict -------------------------------------------------
def test_t06_optimal_sets_disjoint():
    bacc = _bacc_by_task_layer()
    risk_a = {layer: 1.0 - bacc["A"][layer] for layer in X1_LAYERS}
    risk_b = {layer: 1.0 - bacc["B"][layer] for layer in X1_LAYERS}
    set_a = optimal_action_set(risk_a)
    set_b = optimal_action_set(risk_b)
    assert set_a == ("layer1",)
    assert set_b == ("layer2",)
    assert set(set_a) & set(set_b) == set()
    evaluated = evaluate_x1_exact_collision()
    assert evaluated["optimal_sets"] == {"A": ["layer1"], "B": ["layer2"]}


# -- T7: deterministic minimax regret ------------------------------------------
def test_t07_deterministic_minimax_regret():
    evaluated = evaluate_x1_exact_collision()
    value = evaluated["deterministic_regret"]["value"]
    assert abs(value - 2.0 / 3.0) <= TOL
    direct = deterministic_pair_minimax_regret(
        evaluated["risk"]["A"], evaluated["risk"]["B"]
    )
    assert abs(direct["value"] - X1_EXPECTED_DETERMINISTIC_REGRET) <= TOL
    for action in X1_LAYERS:
        assert abs(direct["per_action"][action]["max_regret"] - 2.0 / 3.0) <= TOL


# -- T8: randomized minimax regret ---------------------------------------------
def test_t08_randomized_minimax_regret():
    evaluated = evaluate_x1_exact_collision()
    randomized = evaluated["randomized_regret"]
    assert abs(randomized["value"] - 1.0 / 3.0) <= TOL
    assert set(randomized["support"]) == set(X1_LAYERS)
    probabilities = randomized["probabilities"]
    assert abs(sum(probabilities) - 1.0) <= TOL
    assert all(-TOL <= p <= 1.0 + TOL for p in probabilities)
    assert abs(randomized["expected_regret_A"] - 1.0 / 3.0) <= TOL
    assert abs(randomized["expected_regret_B"] - 1.0 / 3.0) <= TOL
    direct = randomized_pair_minimax_regret(
        evaluated["risk"]["A"], evaluated["risk"]["B"]
    )
    assert abs(direct["value"] - X1_EXPECTED_RANDOMIZED_REGRET) <= TOL
    # The randomized optimum cannot do worse than any deterministic selector.
    assert direct["value"] <= evaluated["deterministic_regret"]["value"] + TOL


# -- T9: deterministic rerun ----------------------------------------------------
def test_t09_rerun_is_fully_deterministic():
    first = evaluate_x1_exact_collision()
    second = evaluate_x1_exact_collision()
    assert canonical_dumps(first) == canonical_dumps(second)
    assert first == second


# -- Supporting unit checks ------------------------------------------------------
def test_tie_break_uses_smallest_canonical_class_id():
    prototypes = reconstruct_prototypes(pattern_message(construct_x1_patterns()["good"]))
    tie_point = np.array([0.0, 1.0], dtype=np.float64)
    assert predict_prototype(tie_point, prototypes) == 0
    point_0 = np.array([-X1_A_FLOAT, X1_Q_FLOAT], dtype=np.float64)
    assert predict_prototype(point_0, prototypes) == 0
    assert predict_prototype(-point_0, prototypes) == 1


def test_core_gates_pass_and_expected_records():
    evaluated = evaluate_x1_exact_collision()
    assert evaluated["core_gates_pass"] is True
    for gate_id, gate in evaluated["gates"].items():
        assert gate["passed"] is True, gate_id
    assert evaluated["message_equality"]["max_abs_vector_sum_diff"] <= TOL
    assert evaluated["prototype_equality"]["max_abs_diff"] <= TOL
    assert evaluated["expected"]["deterministic_regret"] == X1_EXPECTED_DETERMINISTIC_REGRET
    assert evaluated["expected"]["randomized_regret"] == X1_EXPECTED_RANDOMIZED_REGRET


def test_class_message_rejects_mismatched_lengths():
    try:
        class_message(
            np.zeros((3, 2), dtype=np.float64), np.array([0, 1], dtype=np.int64)
        )
    except ValueError:
        return
    raise AssertionError("class_message must reject mismatched features/labels")
