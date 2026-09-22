from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_phase5_nature_sensitivity.py"
CFG = ROOT / "configs" / "phase5_nature_sensitivity_addendum.yaml"

spec = importlib.util.spec_from_file_location("phase5_nature_sensitivity_runner", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def load_cfg():
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


def test_dense_grid_has_51_actions():
    cfg = load_cfg()
    tau = runner.dense_tau_grid(cfg)
    assert len(tau) == 51
    assert tau[0] == 0.0
    assert tau[-1] == 1.0
    assert tau[1] == 0.02


def test_unique_scenario_plan():
    cfg = load_cfg()
    s = runner.build_scenarios(cfg)
    assert len(s) == 7
    assert len({x.scenario_id for x in s}) == 7
    assert "C1_dense_primary" in {x.scenario_id for x in s}


def test_gate_aggregation_passes_expected_fixture():
    cfg = load_cfg()
    scenarios = runner.build_scenarios(cfg)
    retained = {
        "C1_dense_primary": 10,
        "C2_alpha_0p03": 8,
        "C2_alpha_0p3": 6,
        "C3_projection_20260921": 7,
        "C3_projection_20260922": 8,
        "C4_split_20260921": 7,
        "C4_split_20260922": 6,
    }
    rows = []
    for s in scenarios:
        for i in range(56):
            primary = i < 11
            rows.append({
                "scenario_id": s.scenario_id,
                "primary_robust": primary,
                "supported": True,
                "retained_primary_robust": primary and i < retained[s.scenario_id],
            })
    g = runner.summarize_gates(pd.DataFrame(rows), cfg)
    assert g["C1_dense_grid"]["pass"]
    assert g["C2_regularization"]["pass"]
    assert g["C3_projection"]["pass"]
    assert g["C4_group_split"]["pass"]
    assert g["overall_pass"]


def test_gate_aggregation_fails_dense_grid():
    cfg = load_cfg()
    scenarios = runner.build_scenarios(cfg)
    rows = []
    for s in scenarios:
        n = 8 if s.scenario_id == "C1_dense_primary" else 11
        for i in range(56):
            primary = i < 11
            rows.append({
                "scenario_id": s.scenario_id,
                "primary_robust": primary,
                "supported": True,
                "retained_primary_robust": primary and i < n,
            })
    g = runner.summarize_gates(pd.DataFrame(rows), cfg)
    assert not g["C1_dense_grid"]["pass"]
    assert not g["overall_pass"]
