
import numpy as np

from frontier.phase3_moment_collision import (
    WitnessConfig,
    task_distributions,
    summary_signature,
    exact_collision_check,
    risk_curves,
    deterministic_minimax_regret,
    randomized_minimax_regret,
    third_order_escape,
)


def test_exact_empirical_second_order_collision():
    cfg = WitnessConfig(epsilon=0.08, alpha0=0.10)
    x = exact_collision_check(cfg, 100)
    assert x["class0_exact"]
    assert x["class1_exact"]
    assert x["class0_A"]["counts"] == (17, 49, 1, 33)
    assert x["class0_B"]["counts"] == (33, 1, 49, 17)


def test_population_summary_collision():
    cfg = WitnessConfig()
    tasks = task_distributions(cfg)
    a = np.array(summary_signature(tasks["A"]), dtype=float)
    b = np.array(summary_signature(tasks["B"]), dtype=float)
    assert np.max(np.abs(a - b)) < 1e-12


def test_frozen_primary_decision_gap():
    cfg = WitnessConfig(epsilon=0.08, alpha0=0.10)
    taus = [i / 10 for i in range(11)]
    curves = risk_curves(cfg, taus)
    optA = {taus[i] for i, x in enumerate(curves["A"]) if abs(x - curves["A"].max()) < 1e-12}
    optB = {taus[i] for i, x in enumerate(curves["B"]) if abs(x - curves["B"].max()) < 1e-12}
    assert optA.isdisjoint(optB)
    assert abs(deterministic_minimax_regret(curves) - 0.16) < 1e-12
    assert abs(randomized_minimax_regret(curves)["value"] - 0.08) < 1e-12


def test_third_order_escape():
    cfg = WitnessConfig()
    x = third_order_escape(cfg)
    assert x["task_identifiable_by_sign_pattern"]
